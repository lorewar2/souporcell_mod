#[macro_use]
extern crate clap;
extern crate hashbrown;
extern crate rand;
extern crate statrs;
extern crate itertools;
extern crate rayon;
extern crate vcf;
extern crate flate2;

use flate2::read::GzDecoder;
use flate2::read::MultiGzDecoder;
use vcf::*;


use rayon::prelude::*;

use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand::seq::IteratorRandom;

use clap::App;
use std::f32;

use std::f32::MIN;
use std::ffi::OsStr;
use std::io::Read;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::u32::MAX;
use statrs::function::{beta};

use hashbrown::{HashMap,HashSet};
use itertools::izip;

fn main() {
    let params = load_params();
    let cell_barcodes = load_barcodes(&params); 
    let (loci_used, total_cells, cell_data, index_to_locus, locus_to_index) = load_cell_data(&params);
    souporcell_main(loci_used, cell_data, &params, cell_barcodes, locus_to_index);
}

struct ThreadData {
    best_log_probabilities: Vec<Vec<f32>>,
    best_total_log_probability: f32,
    rng: StdRng,
    solves_per_thread: usize,
    thread_num: usize,
}

impl ThreadData {
    fn from_seed(seed: [u8; 32], solves_per_thread: usize, thread_num: usize) -> ThreadData {
        ThreadData {
            best_log_probabilities: Vec::new(),
            best_total_log_probability: f32::NEG_INFINITY,
            rng: SeedableRng::from_seed(seed),
            solves_per_thread: solves_per_thread,
            thread_num: thread_num,
        }
    }
}

fn souporcell_main(loci_used: usize, cell_data: Vec<CellData>, params: &Params, barcodes: Vec<String>, locus_to_index: HashMap<usize, usize>) {
    let seed = [params.seed; 32];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let mut threads: Vec<ThreadData> = Vec::new();
    let solves_per_thread = ((params.restarts as f32)/(params.threads as f32)).ceil() as usize;
    for i in 0..params.threads {
        threads.push(ThreadData::from_seed(new_seed(&mut rng), solves_per_thread, i));
    }
    threads.par_iter_mut().for_each(|thread_data| {
        for iteration in 0..thread_data.solves_per_thread {
            // the main steps here, multiple restarts with threads 
            // INITIALIZING CLUSTERS
            let cluster_centers: Vec<Vec<f32>> = init_cluster_centers(loci_used, &cell_data, params, &mut thread_data.rng, &locus_to_index);
            // EXPRECTATION MAXIMIZATION WITH TEMP ANNEALING
            let (log_loss, log_probabilities) = EM(loci_used, cluster_centers, &cell_data ,params, iteration, thread_data.thread_num);
            if log_loss > thread_data.best_total_log_probability {
                thread_data.best_total_log_probability = log_loss;
                thread_data.best_log_probabilities = log_probabilities;
            }
            eprintln!("thread {} iteration {} done with {}, best so far {}", 
                thread_data.thread_num, iteration, log_loss, thread_data.best_total_log_probability);
        }
    });
    let mut best_log_probability = f32::NEG_INFINITY;
    let mut best_log_probabilities: Vec<Vec<f32>> = Vec::new();
    for thread_data in threads {
        if thread_data.best_total_log_probability > best_log_probability {
            best_log_probability = thread_data.best_total_log_probability;
            best_log_probabilities = thread_data.best_log_probabilities;
        }
    }
    eprintln!("best total log probability = {}", best_log_probability);
    //println!("finished with {}",best_log_probability);
    for (bc, log_probs) in barcodes.iter().zip(best_log_probabilities.iter()) {
        let mut best = 0;
        let mut best_lp = f32::NEG_INFINITY;
        for index in 0..log_probs.len() {
            if log_probs[index] > best_lp {
                best = index;
                best_lp = log_probs[index];
            }
        }
        print!("{}\t{}\t",bc, best);
        for index in 0..log_probs.len() {
            print!("{}",log_probs[index]);
            if index < log_probs.len() - 1 { print!("\t"); } 
        } print!("\n");
    }

}

fn EM_return_cluster_centers (loci: usize, mut cluster_centers: Vec<Vec<f32>>, cell_data: &Vec<CellData>, params: &Params, epoch: usize, thread_num: usize) -> (f32, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut sums: Vec<Vec<f32>> = Vec::new();
    let mut denoms: Vec<Vec<f32>> = Vec::new();
    for cluster in 0..params.num_clusters {
        sums.push(Vec::new());
        denoms.push(Vec::new());
        for index in 0..loci {
            sums[cluster].push(1.0);
            denoms[cluster].push(2.0); // psuedocounts
        }
    }

    let log_prior: f32 = (1.0/(params.num_clusters as f32)).ln();

    let mut change = 1000.0;
    let mut iterations = 0;
    //let mut cell_probabilities: Vec<Vec<f32>> = Vec::new();
    //for _cell in cell_data {
    //    cell_probabilities.push(Vec::new());
    //}
    let mut total_log_loss = f32::NEG_INFINITY;
    let mut total_log_loss_binom = f32::NEG_INFINITY;
    let mut final_log_probabilities = Vec::new();
    for _cell in 0..cell_data.len() {
        final_log_probabilities.push(Vec::new());
    }
    let log_loss_change_limit = 0.01*(cell_data.len() as f32);
    let temp_steps = 9;
    let mut last_log_loss = f32::NEG_INFINITY;
    for temp_step in 0..temp_steps {
        //eprintln!("temp step {}",temp_step);
        let mut log_loss_change = 10000.0;
        //let mut cluster_cells_weighted: Vec<f32> = Vec::new();
        //for cluster in 0..params.num_clusters { cluster_cells_weighted.push(0.0); }
        while (log_loss_change > log_loss_change_limit && iterations < 1000) {
            //for cluster in 0..params.num_clusters { cluster_cells_weighted[cluster] = 0.0; } 
            //let mut log_loss = 0.0;
            let mut log_binom_loss = 0.0;
            reset_sums_denoms(loci, &mut sums, &mut denoms, &cluster_centers, params.num_clusters);
            for (celldex, cell) in cell_data.iter().enumerate() {
                //let log_probabilities = sum_of_squares_loss(cell, &cluster_centers, log_prior, celldex);
                let log_binoms = binomial_loss(cell, &cluster_centers, log_prior, celldex);
                log_binom_loss += log_sum_exp(&log_binoms);
                //eprintln!("cell {} loci {} total_alleles {}", celldex, cell.loci.len(), cell.total_alleles);
                //log_loss += log_sum_exp(&log_binoms);
                let mut temp = (cell.total_alleles/(20.0 * 2.0f32.powf((temp_step as f32)))).max(1.0);
                if temp_step == temp_steps - 1 { temp = 1.0; }
                //if temp_step > 0 { temp = 1.0; }
                let probabilities = normalize_in_log_with_temp(&log_binoms, temp);
                //for cluster in 0..params.num_clusters { cluster_cells_weighted[cluster] += probabilities[cluster]; }
                update_centers_average(&mut sums, &mut denoms, cell, &probabilities);
            
                //println!("normalized probabilities {:?}", probabilities);
                //cell_probabilities[celldex] = probabilities;
                final_log_probabilities[celldex] = log_binoms;//log_probabilities;
            }

            total_log_loss = log_binom_loss;
            log_loss_change = log_binom_loss - last_log_loss;//log_loss - last_log_loss;
            last_log_loss = log_binom_loss;//log_loss;

            update_final(loci, &sums, &denoms, &mut cluster_centers);
            iterations += 1;
            eprintln!("binomial\t{}\t{}\t{}\t{}\t{}\t{}", thread_num, epoch, iterations, temp_step, log_binom_loss, log_loss_change);//, cluster_cells_weighted);
        }
    }
    //for (celldex, probabilities) in cell_probabilities.iter().enumerate() {
    //    println!("cell {} with {} loci, cluster probabilities {:?}", celldex, cell_data[celldex].loci.len(), probabilities);
    //}
    //for center in 0..cluster_centers.len() {
    //    for locus in 0..cluster_centers[0].len() {
    //        println!("cluster {} locus {} {}", center, locus, cluster_centers[center][locus]);
    //    }
    //}
    //println!("total log probability = {}",total_log_loss);

    (total_log_loss, final_log_probabilities, cluster_centers)
}

fn EM(loci: usize, mut cluster_centers: Vec<Vec<f32>>, cell_data: &Vec<CellData>, params: &Params, epoch: usize, thread_num: usize) -> (f32, Vec<Vec<f32>>) {
    let mut sums: Vec<Vec<f32>> = Vec::new();
    let mut denoms: Vec<Vec<f32>> = Vec::new();
    for cluster in 0..params.num_clusters {
        sums.push(Vec::new());
        denoms.push(Vec::new());
        for index in 0..loci {
            sums[cluster].push(1.0);
            denoms[cluster].push(2.0); // psuedocounts
        }
    }

    let log_prior: f32 = (1.0/(params.num_clusters as f32)).ln();

    let mut change = 1000.0;
    let mut iterations = 0;
    //let mut cell_probabilities: Vec<Vec<f32>> = Vec::new();
    //for _cell in cell_data {
    //    cell_probabilities.push(Vec::new());
    //}
    let mut total_log_loss = f32::NEG_INFINITY;
    let mut total_log_loss_binom = f32::NEG_INFINITY;
    let mut final_log_probabilities = Vec::new();
    for _cell in 0..cell_data.len() {
        final_log_probabilities.push(Vec::new());
    }
    let log_loss_change_limit = 0.01*(cell_data.len() as f32);
    let temp_steps = 9;
    let mut last_log_loss = f32::NEG_INFINITY;
    for temp_step in 0..temp_steps {
        //eprintln!("temp step {}",temp_step);
        let mut log_loss_change = 10000.0;
        //let mut cluster_cells_weighted: Vec<f32> = Vec::new();
        //for cluster in 0..params.num_clusters { cluster_cells_weighted.push(0.0); }
        while (log_loss_change > log_loss_change_limit && iterations < 1000) {
            //for cluster in 0..params.num_clusters { cluster_cells_weighted[cluster] = 0.0; } 
            //let mut log_loss = 0.0;
            let mut log_binom_loss = 0.0;
            reset_sums_denoms(loci, &mut sums, &mut denoms, &cluster_centers, params.num_clusters);
            for (celldex, cell) in cell_data.iter().enumerate() {
                //let log_probabilities = sum_of_squares_loss(cell, &cluster_centers, log_prior, celldex);
                let log_binoms = binomial_loss(cell, &cluster_centers, log_prior, celldex);
                log_binom_loss += log_sum_exp(&log_binoms);
                //eprintln!("cell {} loci {} total_alleles {}", celldex, cell.loci.len(), cell.total_alleles);
                //log_loss += log_sum_exp(&log_binoms);
                let mut temp = (cell.total_alleles/(20.0 * 2.0f32.powf((temp_step as f32)))).max(1.0);
                if temp_step == temp_steps - 1 { temp = 1.0; }
                //if temp_step > 0 { temp = 1.0; }
                let probabilities = normalize_in_log_with_temp(&log_binoms, temp);
                //for cluster in 0..params.num_clusters { cluster_cells_weighted[cluster] += probabilities[cluster]; }
                update_centers_average(&mut sums, &mut denoms, cell, &probabilities);
            
                //println!("normalized probabilities {:?}", probabilities);
                //cell_probabilities[celldex] = probabilities;
                final_log_probabilities[celldex] = log_binoms;//log_probabilities;
            }

            total_log_loss = log_binom_loss;
            log_loss_change = log_binom_loss - last_log_loss;//log_loss - last_log_loss;
            last_log_loss = log_binom_loss;//log_loss;

            update_final(loci, &sums, &denoms, &mut cluster_centers);
            iterations += 1;
            eprintln!("binomial\t{}\t{}\t{}\t{}\t{}\t{}", thread_num, epoch, iterations, temp_step, log_binom_loss, log_loss_change);//, cluster_cells_weighted);
        }
    }
    //for (celldex, probabilities) in cell_probabilities.iter().enumerate() {
    //    println!("cell {} with {} loci, cluster probabilities {:?}", celldex, cell_data[celldex].loci.len(), probabilities);
    //}
    //for center in 0..cluster_centers.len() {
    //    for locus in 0..cluster_centers[0].len() {
    //        println!("cluster {} locus {} {}", center, locus, cluster_centers[center][locus]);
    //    }
    //}
    //println!("total log probability = {}",total_log_loss);

    (total_log_loss, final_log_probabilities)
}

fn sum_of_squares_loss(cell_data: &CellData, cluster_centers: &Vec<Vec<f32>>, log_prior: f32, cellnum: usize) -> Vec<f32> {
    let mut log_probabilities: Vec<f32> = Vec::new();
    for (cluster, center) in cluster_centers.iter().enumerate() {
        log_probabilities.push(log_prior);
        for (locus_index, locus) in cell_data.loci.iter().enumerate() {
            log_probabilities[cluster] -= (cell_data.allele_fractions[locus_index] - center[*locus]).powf(2.0);
        }
    }
    log_probabilities 
}

fn binomial_loss(cell_data: &CellData, cluster_centers: &Vec<Vec<f32>>, log_prior: f32, cellnum: usize) -> Vec<f32> {
    let mut log_probabilities: Vec<f32> = Vec::new();
    let mut sum = 0.0;
    for (cluster, center) in cluster_centers.iter().enumerate() {
        log_probabilities.push(log_prior);
        for (locus_index, locus) in cell_data.loci.iter().enumerate() {
            log_probabilities[cluster] += cell_data.log_binomial_coefficient[locus_index] + 
                (cell_data.alt_counts[locus_index] as f32) * center[*locus].ln() + 
                (cell_data.ref_counts[locus_index] as f32) * (1.0 - center[*locus]).ln();
        }
        sum += log_probabilities[cluster];
    }
    
    log_probabilities
}

fn log_sum_exp(p: &Vec<f32>) -> f32{
    let max_p: f32 = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_rst: f32 = p.iter().map(|x| (x - max_p).exp()).sum();
    max_p + sum_rst.ln()
}

fn normalize_in_log(log_probs: &Vec<f32>) -> Vec<f32> { // takes in a log_probability vector and converts it to a normalized probability
    let mut normalized_probabilities: Vec<f32> = Vec::new();
    let sum = log_sum_exp(log_probs);
    for i in 0..log_probs.len() {
        normalized_probabilities.push((log_probs[i]-sum).exp());
    }
    normalized_probabilities
}

fn normalize_in_log_with_temp(log_probs: &Vec<f32>, temp: f32) -> Vec<f32> {
    let mut normalized_probabilities: Vec<f32> = Vec::new();
    let mut new_log_probs: Vec<f32> = Vec::new();
    for log_prob in log_probs {
        new_log_probs.push(log_prob/temp);
    }
    let sum = log_sum_exp(&new_log_probs);
    for i in 0..log_probs.len() {
        normalized_probabilities.push((new_log_probs[i]-sum).exp());
    }
    normalized_probabilities 
}

fn update_final(loci: usize, sums: &Vec<Vec<f32>>, denoms: &Vec<Vec<f32>>, cluster_centers: &mut Vec<Vec<f32>>) {
    for locus in 0..loci {
        for cluster in 0..sums.len() {
            let update = sums[cluster][locus]/denoms[cluster][locus];
            cluster_centers[cluster][locus] = update.min(0.99).max(0.01);//max(0.0001, min(0.9999, update));
        }
    }
}

fn reset_sums_denoms(loci: usize, sums: &mut Vec<Vec<f32>>, 
    denoms: &mut Vec<Vec<f32>>, cluster_centers: &Vec<Vec<f32>>, num_clusters: usize) {
    for cluster in 0..num_clusters {
        for index in 0..loci {
            sums[cluster][index] = 1.0;
            denoms[cluster][index] = 2.0;
        }
    }
}


fn update_centers_flat(sums: &mut Vec<Vec<f32>>, denoms: &mut Vec<Vec<f32>>, cell: &CellData, probabilities: &Vec<f32>) {
    for locus in 0..cell.loci.len() {
        for (cluster, probability) in probabilities.iter().enumerate() {
            sums[cluster][cell.loci[locus]] += probabilities[cluster] * cell.allele_fractions[locus];
            denoms[cluster][cell.loci[locus]] += probabilities[cluster];
        }
    }
}

fn update_centers_average(sums: &mut Vec<Vec<f32>>, denoms: &mut Vec<Vec<f32>>, cell: &CellData, probabilities: &Vec<f32>) {
    for locus in 0..cell.loci.len() {
        for (cluster, probability) in probabilities.iter().enumerate() {
            sums[cluster][cell.loci[locus]] += probabilities[cluster] * (cell.alt_counts[locus] as f32);
            denoms[cluster][cell.loci[locus]] += probabilities[cluster] * ((cell.alt_counts[locus] + cell.ref_counts[locus]) as f32);
        }
    }
}

fn init_cluster_centers(loci_used: usize, cell_data: &Vec<CellData>, params: &Params, rng: &mut StdRng, locus_to_index: &HashMap<usize, usize>) -> Vec<Vec<f32>> {
    if let Some(known_genotypes) = &params.known_genotypes {
        return init_cluster_centers_known_genotypes(loci_used, params, rng, locus_to_index);
    } else if let Some(assigned_cells) = &params.known_cell_assignments {
        return init_cluster_centers_known_cells(loci_used, &cell_data, params, rng);
    } else {
        match params.initialization_strategy {
            ClusterInit::KmeansPP => init_cluster_centers_kmeans_pp(loci_used, &cell_data, params, rng),
            ClusterInit::Overclustering => init_cluster_centers_overclustering(loci_used, &cell_data, params, rng),
            ClusterInit::RandomUniform => init_cluster_centers_uniform(loci_used, params, rng),
            ClusterInit::RandomAssignment => init_cluster_centers_random_assignment(loci_used, &cell_data, params, rng),
            ClusterInit::MiddleVariance => init_cluster_centers_middle_variance(loci_used, &cell_data, params, rng),
        }
    }
}

pub fn reader(filename: &str) -> Box<dyn BufRead> {
    let path = Path::new(filename);
    let file = match File::open(&path) {
        Err(why) => panic!("couldn't open file {}", filename),
        Ok(file) => file,
    };
    if path.extension() == Some(OsStr::new("gz")) {
        Box::new(BufReader::with_capacity(128 * 1024, MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::with_capacity(128 * 1024, file))
    }
}


fn init_cluster_centers_known_genotypes(loci: usize, params: &Params, rng: &mut StdRng, locus_to_index: &HashMap<usize, usize>) -> Vec<Vec<f32>> {
    let mut centers: Vec<Vec<f32>> = Vec::new();
    for cluster in 0..params.num_clusters {
        centers.push(Vec::new());
        for _ in 0..loci {
            centers[cluster].push(0.5);
        }
    }
    let mut vcf_reader = VCFReader::new(reader(params.known_genotypes.as_ref().unwrap())).unwrap();
    let mut locus_id: usize = 0;
    for record in vcf_reader {
        let record = record.unwrap();
        if let Some(loci_index) = locus_to_index.get(&locus_id) {
            if params.known_genotypes_sample_names.len() > 0 {
                for (sample_index, sample) in params.known_genotypes_sample_names.iter().enumerate() {
                    let gt = record.call[sample]["GT"][0].to_string();
                    // complicated way of getting the haplotype to numbers
                    let hap0 = gt.chars().nth(0).unwrap().to_string();
                    if hap0 == "." { continue; }
                    let hap0 = hap0.parse::<u32>().unwrap().min(1);
                    let hap1 = gt.chars().nth(2).unwrap().to_string().parse::<u32>().unwrap().min(1);
                    centers[sample_index][*loci_index] = (((hap0 + hap1) as f32)/2.0).min(0.99).max(0.01);
                }
            } else { assert!(false, "currently requiring known_genotypes_sample_names if known_genotypes set"); }
        }
        locus_id += 1;
    }
    centers
}

fn init_cluster_centers_known_cells(loci: usize, cell_data: &Vec<CellData>, params: &Params, rng: &mut StdRng) -> Vec<Vec<f32>> {
    assert!(false, "known cell assignments not yet implemented");
    Vec::new()
}

// Done for now
fn init_cluster_centers_kmeans_pp(loci: usize, cell_data: &Vec<CellData>, params: &Params, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let mut original_centers: Vec<Vec<f32>> = vec![];
    // new cluster centers with alpha and beta // initialize with ones?
    let mut centers: Vec<Vec<(f32, f32)>> = vec![vec![(1.0, 1.0); loci]; params.num_clusters];
    // select a random cell and update the first cluster center with its ref and alt
    let first_cluster_cell = cell_data.get(rng.gen_range(0, cell_data.len())).unwrap();
    update_cluster_using_cell(first_cluster_cell, &mut centers, 0);
    for current_cluster_index in 1..params.num_clusters + 1 {
        // calculate the Beta-Binomial loss from each cell to the nearest center
        let mut loss_vec: Vec<f32> = vec![];
        let mut preferred_cluster: Vec<usize> = vec![]; //for assigning each cell after cluster selection ends
        // go through all the known centers which is 0..current_cluster_index and get the min loss one for each cell
        for current_cell in cell_data {
            let mut min_loss_index: (f32, usize) = (f32::MAX, 0);
            for loop_2_current_cluster_index in 0..current_cluster_index {
                let loss = beta_binomial_loss(current_cell, &centers[loop_2_current_cluster_index]);
                if loss < min_loss_index.0 {
                    min_loss_index = (loss, loop_2_current_cluster_index);
                }
            }
            loss_vec.push(min_loss_index.0);
            preferred_cluster.push(min_loss_index.1);
        }
        // select the cell as cluster center
        if current_cluster_index < params.num_clusters {
            // get sum and divide
            let loss_sum: f32 = loss_vec.iter().sum();
            for value in loss_vec.iter_mut() {
                *value /= loss_sum;
            }
            // get a random value between 0 and 1
            let r: f32 = rng.gen();
            let mut cumulative_probability = 0.0;
            for (selected_cell, &probability) in cell_data.iter().zip(loss_vec.iter()) {
                cumulative_probability += probability;
                if r < cumulative_probability {
                    // the selected cell, update the current_cluster_index
                    update_cluster_using_cell(selected_cell, &mut centers, current_cluster_index);
                    break;
                }
            }
        }
        // using the preferred_cluster for each cell on the final iteration, make the original cluster centers, 
        // known cell assignment
        else {
            // Conversion code
            let mut sums: Vec<Vec<f32>> = Vec::new();
            let mut denoms: Vec<Vec<f32>> = Vec::new();
            // put some random values in sums and 0.01 in denoms for each cluster
            for cluster in 0..params.num_clusters {
                sums.push(Vec::new());
                denoms.push(Vec::new());
                for _ in 0..loci {
                    sums[cluster].push(rng.gen::<f32>()*0.01);
                    denoms[cluster].push(0.01);
                }
            }
            // go through each cell
            for (index, cell) in cell_data.iter().enumerate() {
                // choose the preferred
                //let cluster = rng.gen_range(0,params.num_clusters);
                let cluster = preferred_cluster[index];
                // go thorugh the cell locations
                for locus in 0..cell.loci.len() {
                    let alt_c = cell.alt_counts[locus] as f32;
                    let total = alt_c + (cell.ref_counts[locus] as f32);
                    let locus_index = cell.loci[locus];
                    // update sum and denoms for locus index
                    sums[cluster][locus_index] += alt_c;
                    denoms[cluster][locus_index] += total;
                }
            }
            for cluster in 0..params.num_clusters {
                for locus in 0..loci {
                    sums[cluster][locus] = sums[cluster][locus]/denoms[cluster][locus] + (rng.gen::<f32>()/2.0 - 0.25);
                    sums[cluster][locus] = sums[cluster][locus].min(0.9999).max(0.0001);
                }
            }
            original_centers = sums;
        }
    }
    original_centers
}

fn init_cluster_centers_overclustering(loci: usize, cell_data: &Vec<CellData>, params: &Params, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let mut original_centers: Vec<Vec<f32>> = vec![];
    // random initialization of clusters initialize twice the number of required clusters
    for cluster in 0..params.num_clusters * 2 {
        original_centers.push(Vec::new());
        for _ in 0..loci {
            original_centers[cluster].push(rng.gen::<f32>().min(0.9999).max(0.0001));
        }
    }
    // go through the clusters and get the two clusters which are closest with each other
    while original_centers.len() != params.num_clusters {
        eprintln!("Current number of clusters {}", original_centers.len());   
        let mut closest_2_clusters = (0, 0);
        let mut fartherst_2_clusters = (0, 0);
        let mut min_dist = f32::MAX;
        let mut max_dist = f32::MIN;
        // go through each cluster and compare it to each other 2 loops i guess
        for (index1, cluster1) in original_centers.iter().enumerate() {
            for (index2, cluster2) in original_centers.iter().enumerate().skip(index1) {
                if index1 != index2 {
                    let curr_dist = cluster_compare(cluster1, cluster2);
                    if curr_dist < min_dist {
                        closest_2_clusters = (index1, index2);
                        min_dist = curr_dist;
                    }
                    if curr_dist > max_dist {
                        fartherst_2_clusters = (index1, index2);
                        max_dist = curr_dist;
                    }
                }
            }
        }
        eprintln!("Min distance {} closeset clusters {:?};;; Max distance {} farthest clusters {:?}", min_dist, closest_2_clusters, max_dist, fartherst_2_clusters); 
        // save the two clusters which are closest and merge them together, del on for now
        original_centers.remove(closest_2_clusters.0);
    }
    original_centers
}

fn cluster_compare (cluster1: &Vec<f32>, cluster2: &Vec<f32>) -> f32 {
    let mut squared_dist = 0.0;
    // go thorugh each loci of the two clusters, and get the squared distance between them
    assert!(cluster1.len() == cluster2.len());
    for locus in 0..cluster1.len() {
        squared_dist += (cluster1[locus].powi(2) - cluster2[locus].powi(2)).abs();
    }
    squared_dist
}

fn cluster_merge (cluster1: Vec<f32>, cluster2: Vec<f32>) {
    // lets just delete one for now
}

fn init_cluster_centers_overclustering_old(loci: usize, cell_data: &Vec<CellData>, params: &Params, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let mut original_centers: Vec<Vec<f32>> = vec![];
    // get 500 cluster centers using kmeanspp
    let mut centers: Vec<Vec<(f32, f32)>> = vec![vec![(1.0, 1.0); loci]; 500];
    // select a random cell and update the first cluster center with its ref and alt
    let first_cluster_cell = cell_data.get(rng.gen_range(0, cell_data.len())).unwrap();
    update_cluster_using_cell(first_cluster_cell, &mut centers, 0);
    for current_cluster_index in 1..500 + 1{
        // calculate the Beta-Binomial loss from each cell to the nearest center
        let mut loss_vec: Vec<f32> = vec![];
        let mut preferred_cluster: Vec<usize> = vec![]; //for assigning each cell after cluster selection ends
        // go through all the known centers which is 0..current_cluster_index and get the min loss one for each cell
        for current_cell in cell_data {
            let mut min_loss_index: (f32, usize) = (f32::MAX, 0);
            for loop_2_current_cluster_index in 0..current_cluster_index {
                let loss = beta_binomial_loss(current_cell, &centers[loop_2_current_cluster_index]);
                if loss < min_loss_index.0 {
                    min_loss_index = (loss, loop_2_current_cluster_index);
                }
            }
            loss_vec.push(min_loss_index.0);
            preferred_cluster.push(min_loss_index.1);
        }
        // select the cell as cluster center
        if current_cluster_index < params.num_clusters {
            // get sum and divide
            let loss_sum: f32 = loss_vec.iter().sum();
            for value in loss_vec.iter_mut() {
                *value /= loss_sum;
            }
            // get a random value between 0 and 1
            let r: f32 = rng.gen();
            let mut cumulative_probability = 0.0;
            for (selected_cell, &probability) in cell_data.iter().zip(loss_vec.iter()) {
                cumulative_probability += probability;
                if r < cumulative_probability {
                    // the selected cell, update the current_cluster_index
                    update_cluster_using_cell(selected_cell, &mut centers, current_cluster_index);
                    break;
                }
            }
        }
        else {
            // Conversion code
            let mut sums: Vec<Vec<f32>> = Vec::new();
            let mut denoms: Vec<Vec<f32>> = Vec::new();
            // put some random values in sums and 0.01 in denoms for each cluster
            for cluster in 0..params.num_clusters {
                sums.push(Vec::new());
                denoms.push(Vec::new());
                for _ in 0..loci {
                    sums[cluster].push(rng.gen::<f32>()*0.01);
                    denoms[cluster].push(0.01);
                }
            }
            // go through each cell
            for (index, cell) in cell_data.iter().enumerate() {
                // choose the preferred
                //let cluster = rng.gen_range(0,params.num_clusters);
                let cluster = preferred_cluster[index];
                // go thorugh the cell locations
                for locus in 0..cell.loci.len() {
                    let alt_c = cell.alt_counts[locus] as f32;
                    let total = alt_c + (cell.ref_counts[locus] as f32);
                    let locus_index = cell.loci[locus];
                    // update sum and denoms for locus index
                    sums[cluster][locus_index] += alt_c;
                    denoms[cluster][locus_index] += total;
                }
            }
            for cluster in 0..params.num_clusters {
                for locus in 0..loci {
                    sums[cluster][locus] = sums[cluster][locus]/denoms[cluster][locus] + (rng.gen::<f32>()/2.0 - 0.25);
                    sums[cluster][locus] = sums[cluster][locus].min(0.9999).max(0.0001);
                }
            }
            original_centers = sums;
        }
    }
    let mut cluster_centers_as_cell = vec![];
    // Conversion code // convert the cluster centers to cells
    for cluster_center in &centers {
        let mut temp_cell_data: CellData = CellData{
            log_binomial_coefficient: vec![],
            allele_fractions: vec![],
            alt_counts: vec![],
            ref_counts: vec![],
            loci: vec![],
            total_alleles: 0.0
        };
        for loci in 0..cluster_center.len() {
            // make all the required stuff for the current cell data
            let value = cluster_center[loci];
            let alt_count = value.0;
            let ref_count = value.1;
            temp_cell_data.alt_counts.push(alt_count as u32);
            temp_cell_data.ref_counts.push(ref_count as u32);
            temp_cell_data.loci.push(loci);
            temp_cell_data.allele_fractions.push((alt_count as f32)/((ref_count + alt_count) as f32));
            temp_cell_data.log_binomial_coefficient.push(
                statrs::function::factorial::ln_binomial((ref_count + alt_count) as u64, alt_count as u64) as f32);
            temp_cell_data.total_alleles += (ref_count + alt_count) as f32;
        }
        cluster_centers_as_cell.push(temp_cell_data);
    }
    // do EM and get the lowest cluster center stuff
    let sub_sampling_attempts = 10;
    let mut best_loss = f32::MAX;
    let mut best_cluster = vec![];
    
    for _sub_sample_attempt in 0..sub_sampling_attempts {
        // get random required cluster centers and save if loss is the lowest
        let current_cluster_center = original_centers.choose_multiple(rng, params.num_clusters).cloned().collect();
        let (loss, _, final_cluster_centers) = EM_return_cluster_centers (loci, current_cluster_center, &cluster_centers_as_cell, params, 0, 0);
        if best_loss > loss {
            best_loss = loss;
            best_cluster = final_cluster_centers.clone();
        }
        eprintln!("@@@@ best loss for subsampled clusters {}", best_loss);
    }
    best_cluster
}

fn init_cluster_centers_ksubsampling(loci: usize, cell_data: &Vec<CellData>, params: &Params, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let sub_sampling_attempts: usize = 10;
    let sub_samples_to_get = cell_data.len() * (10 / 100);
    let mut aggregated_cluster_centers: Vec<Vec<Vec<f32>>> = vec![];
    let mut cluster_centers_as_cell: Vec<CellData> = vec![];
    // subsample the cell data j times (j subsamples  from cell_data)
    for sub_sample_attempt in 0..sub_sampling_attempts {
        // select random cells
        let selected_cells: Vec<_> = cell_data.iter().choose_multiple(rng, sub_samples_to_get).into_iter().cloned().collect();
        // get random cluster centers using selected cells
        let cluster_centers = init_cluster_centers_uniform(loci, params, rng);
        // EM with initial cluster centers
        let (_, _, final_cluster_centers) = EM_return_cluster_centers (loci, cluster_centers, &selected_cells, params, 0, 0);
        // convert the cluster centers to cell data
        for cluster_center in &final_cluster_centers {
            let mut temp_cell_data: CellData = CellData{
                log_binomial_coefficient: vec![],
                allele_fractions: vec![],
                alt_counts: vec![],
                ref_counts: vec![],
                loci: vec![],
                total_alleles: 0.0
            };
            for loci in 0..cluster_center.len() {
                // make all the required stuff for the current cell data
                let value = cluster_center[loci];
                let mut alt_count = 1;
                if value > 1.0 {
                    alt_count = ((value as usize) * 10) as usize;
                }
                let mut ref_count: usize = 1;
                if alt_count < 10 {
                    ref_count = 10 - alt_count;
                }
                temp_cell_data.alt_counts.push(alt_count as u32);
                temp_cell_data.ref_counts.push(ref_count as u32);
                temp_cell_data.loci.push(loci);
                temp_cell_data.allele_fractions.push((alt_count as f32)/((ref_count + alt_count) as f32));
                temp_cell_data.log_binomial_coefficient.push(
                    statrs::function::factorial::ln_binomial((ref_count + alt_count) as u64, alt_count as u64) as f32);
                temp_cell_data.total_alleles += (ref_count + alt_count) as f32;
            }
            cluster_centers_as_cell.push(temp_cell_data);
        }
        aggregated_cluster_centers.push(final_cluster_centers);
    }
    // rerun em using the collected cell data and cluster centers as cluster centers then get the one with best loss (min)
    let mut best_loss = f32::MAX;
    let mut best_cluster = vec![];
    for sub_sample_attempt in 0..sub_sampling_attempts {
        let current_cluster_center = aggregated_cluster_centers[sub_sample_attempt].clone();
        let (loss, _, final_cluster_centers) = EM_return_cluster_centers (loci, current_cluster_center, &cluster_centers_as_cell, params, 0, 0);
        if best_loss > loss {
            best_loss = loss;
            best_cluster = final_cluster_centers.clone();
        }
    }
    best_cluster
}

// Update the cluster based on the cell data
fn update_cluster_using_cell(cell_data: &CellData, cluster_centers: &mut Vec<Vec<(f32, f32)>>, cluster_index: usize) {
    for (locus_index, locus) in cell_data.loci.iter().enumerate() {
        cluster_centers[cluster_index][*locus].0 += cell_data.alt_counts[locus_index] as f32;
        cluster_centers[cluster_index][*locus].1 += cell_data.ref_counts[locus_index] as f32;
    }
}
// Beta binomial function to take the loss between a cluster center and a cell
fn beta_binomial_loss(cell_data: &CellData, cluster_center: &Vec<(f32, f32)>) -> f32 {
    let mut log_probability: f32 = 0.0;
    for (locus_index, locus) in cell_data.loci.iter().enumerate() {
        let center_locus_alpha = cluster_center[*locus].0 as f64;
        let center_locus_beta = cluster_center[*locus].1 as f64;
        let cell_locus_bino_coeff = cell_data.log_binomial_coefficient[locus_index] as f64;
        let cell_locus_ref = cell_data.ref_counts[locus_index] as f64;
        let cell_locus_alt = cell_data.alt_counts[locus_index] as f64;
        let log_loss_locus = cell_locus_bino_coeff
                        + beta::ln_beta(cell_locus_ref + center_locus_alpha, cell_locus_alt + center_locus_beta)
                        - beta::ln_beta(center_locus_alpha,center_locus_beta);
        log_probability += log_loss_locus as f32;
    }
    log_probability
}

fn init_cluster_centers_uniform(loci: usize, params: &Params, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let mut centers: Vec<Vec<f32>> = Vec::new();
    for cluster in 0..params.num_clusters {
        centers.push(Vec::new());
        for _ in 0..loci {
            centers[cluster].push(rng.gen::<f32>().min(0.9999).max(0.0001));
        }
    }
    centers
}

fn init_cluster_centers_random_assignment(loci: usize, cell_data: &Vec<CellData>, params: &Params, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let mut sums: Vec<Vec<f32>> = Vec::new();
    let mut denoms: Vec<Vec<f32>> = Vec::new();
    // put some random values in sums and 0.01 in denoms for each cluster
    for cluster in 0..params.num_clusters {
        sums.push(Vec::new());
        denoms.push(Vec::new());
        for _ in 0..loci {
            sums[cluster].push(rng.gen::<f32>()*0.01);
            denoms[cluster].push(0.01);
        }
    }
    // go through each cell
    for cell in cell_data {
        // choose a random cluster
        let cluster = rng.gen_range(0,params.num_clusters);
        // go thorugh the cell locations
        for locus in 0..cell.loci.len() {
            let alt_c = cell.alt_counts[locus] as f32;
            let total = alt_c + (cell.ref_counts[locus] as f32);
            let locus_index = cell.loci[locus];
            // update sum and denoms for locus index
            sums[cluster][locus_index] += alt_c;
            denoms[cluster][locus_index] += total;
        }
    }
    for cluster in 0..params.num_clusters {
        for locus in 0..loci {
            sums[cluster][locus] = sums[cluster][locus]/denoms[cluster][locus] + (rng.gen::<f32>()/2.0 - 0.25);
            sums[cluster][locus] = sums[cluster][locus].min(0.9999).max(0.0001);
        }
    }
    let centers = sums;
    centers
}

fn init_cluster_centers_middle_variance(loci: usize, cell_data: &Vec<CellData>, params: &Params, rng: &mut StdRng) -> Vec<Vec<f32>> {
    assert!(false, "middle variance not yet implemented");
    Vec::new()
}

fn load_cell_data(params: &Params) -> (usize, usize, Vec<CellData>, Vec<usize>, HashMap<usize, usize>) {
    let alt_reader = File::open(params.alt_mtx.to_string()).expect("cannot open alt mtx file");

    let alt_reader = BufReader::new(alt_reader);
    let ref_reader = File::open(params.ref_mtx.to_string()).expect("cannot open ref mtx file");
    
    let ref_reader = BufReader::new(ref_reader);
    let mut used_loci: HashSet<usize> = HashSet::new();
    let mut line_number = 0;
    let mut total_loci = 0;
    let mut total_cells = 0;
    let mut all_loci: HashSet<usize> = HashSet::new();
    let mut locus_cell_counts: HashMap<usize, [u32; 2]> = HashMap::new();
    let mut locus_umi_counts: HashMap<usize, [u32; 2]> = HashMap::new();
    let mut locus_counts: HashMap<usize, HashMap<usize, [u32; 2]>> = HashMap::new();
    for (alt_line, ref_line) in izip!(alt_reader.lines(), ref_reader.lines()) {
        let alt_line = alt_line.expect("cannot read alt mtx");
        let ref_line = ref_line.expect("cannot read ref mtx");
        if line_number > 2 {
            let alt_tokens: Vec<&str> = alt_line.split_whitespace().collect();
            let ref_tokens: Vec<&str> = ref_line.split_whitespace().collect();
            let locus = alt_tokens[0].to_string().parse::<usize>().unwrap() - 1;
            all_loci.insert(locus);
            let cell = alt_tokens[1].to_string().parse::<usize>().unwrap() - 1;
            let ref_count = ref_tokens[2].to_string().parse::<u32>().unwrap();
            let alt_count = alt_tokens[2].to_string().parse::<u32>().unwrap();
            assert!(locus < total_loci);
            assert!(cell < total_cells);
            let cell_counts = locus_cell_counts.entry(locus).or_insert([0; 2]);
            let umi_counts = locus_umi_counts.entry(locus).or_insert([0; 2]);
            if ref_count > 0 { cell_counts[0] += 1; umi_counts[0] += ref_count; }
            if alt_count > 0 { cell_counts[1] += 1; umi_counts[1] += alt_count; }
            let cell_counts = locus_counts.entry(locus).or_insert(HashMap::new());
            cell_counts.insert(cell, [ref_count, alt_count]);
        } else if line_number == 2 {
            let tokens: Vec<&str> = alt_line.split_whitespace().collect();
            total_loci = tokens[0].to_string().parse::<usize>().unwrap();
            total_cells = tokens[1].to_string().parse::<usize>().unwrap();
        }
        line_number += 1;
    }
    let mut all_loci2: Vec<usize> = Vec::new();
    for loci in all_loci {
        all_loci2.push(loci);
    }
    let mut all_loci = all_loci2;

    all_loci.sort();
    let mut index_to_locus: Vec<usize> = Vec::new();
    let mut locus_to_index: HashMap<usize, usize> = HashMap::new();
    let mut cell_data: Vec<CellData> = Vec::new();
    for _cell in 0..total_cells {
        cell_data.push(CellData::new());
    }
    let mut locus_index = 0;
    for locus in all_loci {
        let cell_counts = locus_cell_counts.get(&locus).unwrap();
        let umi_counts = locus_umi_counts.get(&locus).unwrap();
        if cell_counts[0] >= params.min_ref && cell_counts[1] >= params.min_alt && umi_counts[0] >= params.min_ref_umis && umi_counts[1] >= params.min_alt_umis {
            used_loci.insert(locus);
            index_to_locus.push(locus);
            locus_to_index.insert(locus, locus_index);
            for (cell, counts) in locus_counts.get(&locus).unwrap() {
                if counts[0]+counts[1] == 0 { continue; }
                cell_data[*cell].alt_counts.push(counts[1]);
                cell_data[*cell].ref_counts.push(counts[0]);
                cell_data[*cell].loci.push(locus_index);
                cell_data[*cell].allele_fractions.push((counts[1] as f32)/((counts[0] + counts[1]) as f32));
                cell_data[*cell].log_binomial_coefficient.push(
                     statrs::function::factorial::ln_binomial((counts[1]+counts[0]) as u64, counts[1] as u64) as f32);
                cell_data[*cell].total_alleles += (counts[0] + counts[1]) as f32;
                //println!("cell {} locus {} alt {} ref {} fraction {}",*cell, locus_index, counts[1], counts[0], 
                //    (counts[1] as f32)/((counts[0] + counts[1]) as f32));
            }
            locus_index += 1;
        }
    }
    eprintln!("total loci used {}",used_loci.len());
    
    (used_loci.len(), total_cells, cell_data, index_to_locus, locus_to_index)
}
#[derive(Clone)]
struct CellData {
    allele_fractions: Vec<f32>,
    log_binomial_coefficient: Vec<f32>,
    alt_counts: Vec<u32>,
    ref_counts: Vec<u32>,
    loci: Vec<usize>,
    total_alleles: f32,
}

impl CellData {
    fn new() -> CellData {
        CellData{
            allele_fractions: Vec::new(),
            log_binomial_coefficient: Vec::new(),
            alt_counts: Vec::new(),
            ref_counts: Vec::new(),
            loci: Vec::new(),
            total_alleles: 0.0,
        }
    }
}



fn load_barcodes(params: &Params) -> Vec<String> {
    //let reader = File::open(params.barcodes.to_string()).expect("cannot open barcode file");
    //let reader = BufReader::new(reader);
    let reader = reader(&params.barcodes);
    let mut cell_barcodes: Vec<String> = Vec::new();
    for line in reader.lines() {
        let line = line.expect("Unable to read line");
        cell_barcodes.push(line.to_string());
    }
    cell_barcodes
}


#[derive(Clone)]
struct Params {
    ref_mtx: String,
    alt_mtx: String,
    barcodes: String,
    num_clusters: usize,
    min_alt: u32,
    min_ref: u32,
    min_alt_umis: u32,
    min_ref_umis: u32,
    restarts: u32,
    known_cell_assignments: Option<String>,
    known_genotypes: Option<String>,
    known_genotypes_sample_names: Vec<String>,
    initialization_strategy: ClusterInit,
    threads: usize,
    seed: u8,
}

#[derive(Clone)]
enum ClusterInit {
    KmeansPP,
    Overclustering,
    RandomUniform,
    RandomAssignment,
    MiddleVariance,
}

fn load_params() -> Params {
    let yaml = load_yaml!("params.yml");
    let params = App::from_yaml(yaml).get_matches();
    let ref_mtx = params.value_of("ref_matrix").unwrap();
    let alt_mtx = params.value_of("alt_matrix").unwrap();
    let barcodes = params.value_of("barcodes").unwrap();
    let num_clusters = params.value_of("num_clusters").unwrap();
    let num_clusters = num_clusters.to_string().parse::<usize>().unwrap();
    let min_alt = params.value_of("min_alt").unwrap_or("4");
    let min_alt = min_alt.to_string().parse::<u32>().unwrap();
    let min_ref = params.value_of("min_ref").unwrap_or("4");
    let min_ref = min_ref.to_string().parse::<u32>().unwrap();
    let restarts = params.value_of("restarts").unwrap_or("100");
    let restarts = restarts.to_string().parse::<u32>().unwrap();
    let known_cell_assignments = params.value_of("known_cell_assignments");
    let known_cell_assignments = match known_cell_assignments {
        Some(x) => Some(x.to_string()),
        None => None,
    };
    let known_genotypes = params.value_of("known_genotypes");
    let known_genotypes = match known_genotypes {
        Some(x) => {
            assert!(known_cell_assignments == None, "Cannot set both known_genotypes and known_cell_assignments");
            Some(x.to_string())
        },
        None => None,
    };
    let known_genotypes_sample_names = params.values_of("known_genotypes_sample_names");
    let known_genotypes_sample_names: Vec<&str> = match known_genotypes_sample_names {
        Some(x) => x.collect(),
        None => Vec::new(),
    };
    let mut sample_names: Vec<String> = Vec::new();
    for name in known_genotypes_sample_names {
        sample_names.push(name.to_string());
    }

    //let initialization_strategy = params.value_of("initialization_strategy").unwrap_or("random_uniform");
    let initialization_strategy = params.value_of("initialization_strategy").unwrap_or("overcluster");
    let initialization_strategy = match initialization_strategy {
        "kmeans++" => ClusterInit::KmeansPP,
        "random_uniform" => ClusterInit::RandomUniform,
        "random_cell_assignment" => ClusterInit::RandomAssignment,
        "middle_variance" => ClusterInit::MiddleVariance,
        "overcluster" => ClusterInit::Overclustering,
        _ => {
            assert!(false, "initialization strategy must be one of kmeans++, random_uniform, random_cell_assignment, middle_variance");
            ClusterInit::RandomAssignment
        },
    };

    let threads = params.value_of("threads").unwrap_or("1");
    let threads = threads.to_string().parse::<usize>().unwrap();

    let seed = params.value_of("seed").unwrap_or("4");
    let seed = seed.to_string().parse::<u8>().unwrap();

    let min_ref_umis = params.value_of("min_ref_umis").unwrap_or("0");
    let min_ref_umis = min_ref_umis.to_string().parse::<u32>().unwrap();
    
    
    let min_alt_umis = params.value_of("min_alt_umis").unwrap_or("0");
    let min_alt_umis = min_alt_umis.to_string().parse::<u32>().unwrap();

    Params{
        ref_mtx: ref_mtx.to_string(),
        alt_mtx: alt_mtx.to_string(),
        barcodes: barcodes.to_string(),
        num_clusters: num_clusters,
        min_alt: min_alt,
        min_ref: min_ref,
        restarts: restarts,
        known_cell_assignments: known_cell_assignments,
        known_genotypes: known_genotypes,
        known_genotypes_sample_names: sample_names,
        initialization_strategy: initialization_strategy,
        threads: threads,
        seed: seed,
        min_alt_umis: min_alt_umis,
        min_ref_umis: min_ref_umis,
    }
}

fn new_seed(rng: &mut StdRng) -> [u8; 32] {
    let mut seed = [0; 32];
    for i in 0..32 {
        seed[i] = rng.gen::<u8>();
    }
    seed
}
