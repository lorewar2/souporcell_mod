#[macro_use]
extern crate clap;
extern crate hashbrown;
extern crate rand;
extern crate statrs;
extern crate itertools;
extern crate rayon;
extern crate vcf;
extern crate flate2;

use flate2::read::MultiGzDecoder;
use rayon::prelude::*;
use rand::{Rng, rngs::StdRng, SeedableRng};
use std::{f32, ffi::OsStr, fs::File, io::{BufRead, BufReader}, path::Path};
use statrs::function::beta;
use hashbrown::{HashMap,HashSet};
use itertools::izip;
use clap::App;

const TEMP: f32 = 20.0;
const MULTIPLY_CLUS: usize = 10;
const READ_ALT_REF_MIN: &str = "4";
const USE_KHM: bool = true;
const P_DIM: f32 = 2.0;

fn main() {
    let params = load_params();
    let cell_barcodes = load_barcodes(&params); 
    let (loci_used, _total_cells, cell_data, _index_to_locus, _locus_to_index) = load_cell_data(&params);
    souporcell_main(loci_used, cell_data, &params, cell_barcodes);
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

fn souporcell_main(loci_used: usize, cell_data: Vec<CellData>, params: &Params, barcodes: Vec<String>) {
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
            let cluster_centers: Vec<Vec<f32>> = init_cluster_centers(loci_used, &cell_data, params, &mut thread_data.rng);
            let (log_loss, log_probabilities);
            // Main method
            if USE_KHM {
                (log_loss, log_probabilities) = khm(loci_used, cluster_centers, &cell_data ,params, iteration, thread_data.thread_num);
            }
            else {
                (log_loss, log_probabilities) = em(loci_used, cluster_centers, &cell_data ,params, iteration, thread_data.thread_num);
            }
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

fn khm(loci: usize, mut cluster_centers: Vec<Vec<f32>>, cell_data: &Vec<CellData>, params: &Params, epoch: usize, thread_num: usize) -> (f32, Vec<Vec<f32>>) {
    // sums and denoms for likelihood calculation
    let mut sums: Vec<Vec<f32>> = Vec::new();
    let mut denoms: Vec<Vec<f32>> = Vec::new();
    for cluster in 0..params.num_clusters {
        sums.push(Vec::new());
        denoms.push(Vec::new());
        for _index in 0..loci {
            sums[cluster].push(1.0);
            denoms[cluster].push(2.0); // psuedocounts
        }
    }
    let mut total_log_loss = f32::NEG_INFINITY;
    let mut final_log_probabilities = Vec::new();
    let mut iterations = 0;
    let log_prior: f32 = (1.0/(params.num_clusters as f32)).ln();

    for _cell in 0..cell_data.len() {
        final_log_probabilities.push(Vec::new());
    }
    let mut last_log_loss = f32::NEG_INFINITY;
    let mut log_loss_change;
    while iterations < 20 {
        // should prob calcualte the performance metric too, to quit
        let mut log_binom_loss = 0.0;
        // reset sum and denoms
        reset_sums_denoms(loci, &mut sums, &mut denoms, params.num_clusters);
        let mut cell_khm_perfs = vec![];
        for (celldex, cell) in cell_data.iter().enumerate() {
            // both log loss and min loss clus
            let (log_binoms, min_clus) = binomial_loss_with_min_index(cell, &cluster_centers, log_prior);
            // calculate the cell khm perf function
            cell_khm_perfs.push((params.num_clusters as f32).ln() - calculate_khm_perf_for_cell(&log_binoms));
            // for total loss
            log_binom_loss += log_sum_exp(&log_binoms);
            // calculate the q and q sum for cell wrt each cluster
            let (q_vec, q_sum) = calculate_q_for_current_cell(&log_binoms, min_clus);
            // adjust with temp
            update_centers_hm(&mut sums, &mut denoms, cell, &q_vec, q_sum);
            final_log_probabilities[celldex] = log_binoms;
        }
        let khm_log_loss = log_sum_exp(&cell_khm_perfs);
        total_log_loss = log_binom_loss;
        log_loss_change = log_binom_loss - last_log_loss;
        last_log_loss = log_binom_loss;
        update_final(loci, &sums, &denoms, &mut cluster_centers);
        iterations += 1;
        eprintln!("binomial\t{}\t{}\t{}\t{}\t{}\tkhm_loss: {}", thread_num, epoch, iterations, log_binom_loss, log_loss_change, khm_log_loss);
    }
    (total_log_loss, final_log_probabilities)
}

fn calculate_khm_perf_for_cell (log_binoms: &Vec<f32>) -> f32 {
    let mut inverse_log_binoms = vec![];
    for log_binom in log_binoms {
        inverse_log_binoms.push(-P_DIM * log_binom);
    }
    log_sum_exp(&inverse_log_binoms)
}

fn update_centers_hm(sums: &mut Vec<Vec<f32>>, denoms: &mut Vec<Vec<f32>>, cell: &CellData, q_vec: &Vec<f32>, q_sum: f32) {
    for locus in 0..cell.loci.len() {
        for (cluster, log_probability) in q_vec.iter().enumerate() {
            let probability = (log_probability - q_sum).exp();
            sums[cluster][cell.loci[locus]] += probability * (cell.alt_counts[locus] as f32);
            denoms[cluster][cell.loci[locus]] += probability * ((cell.alt_counts[locus] + cell.ref_counts[locus]) as f32);
        }
    }
}

fn calculate_q_for_current_cell (log_loss_vec: &Vec<f32>, min_clus: usize) -> (Vec<f32>, f32){
    // we need the sum as well
    let q_sum: f32;
    let mut q_vec = vec![];
    let log_winner_cluster_loss = log_loss_vec[min_clus];
    // first calculate the common denom
    let mut log_loss_winner_sub_current = vec![];
    for (index, log_loss) in log_loss_vec.iter().enumerate() {
        if index != min_clus {
            log_loss_winner_sub_current.push(P_DIM * (log_winner_cluster_loss - log_loss));
        }
        else {
            log_loss_winner_sub_current.push(0.0);
        }
    }
    let q_denom = log_sum_exp(&log_loss_winner_sub_current);
    // calculate q
    for (_index, log_loss) in log_loss_vec.iter().enumerate() {
        let q_for_cluster = ((2.0 * P_DIM * log_winner_cluster_loss) - ((P_DIM + 2.0) * log_loss)) - (2.0 * q_denom);
        q_vec.push(q_for_cluster);
    }
    // get the sum
    q_sum = log_sum_exp(&q_vec);
    (q_vec, q_sum)
}

fn binomial_loss_with_min_index(cell_data: &CellData, cluster_centers: &Vec<Vec<f32>>, log_prior: f32) -> (Vec<f32>, usize) {
    let mut log_probabilities: Vec<f32> = Vec::new();
    let mut min_log: f32 = f32::MIN;
    let mut min_index: usize = 0;
    for (cluster, center) in cluster_centers.iter().enumerate() {
        log_probabilities.push(log_prior);
        for (locus_index, locus) in cell_data.loci.iter().enumerate() {
            log_probabilities[cluster] += cell_data.log_binomial_coefficient[locus_index] + 
                (cell_data.alt_counts[locus_index] as f32) * center[*locus].ln() + 
                (cell_data.ref_counts[locus_index] as f32) * (1.0 - center[*locus]).ln();
        }
        if log_probabilities[cluster] > min_log {
            min_log = log_probabilities[cluster];
            min_index = cluster;
        }
    }
    (log_probabilities, min_index)
}

fn em(loci: usize, mut cluster_centers: Vec<Vec<f32>>, cell_data: &Vec<CellData>, params: &Params, epoch: usize, thread_num: usize) -> (f32, Vec<Vec<f32>>) {
    // sums and denoms for likelihood calculation
    let mut sums: Vec<Vec<f32>> = Vec::new();
    let mut denoms: Vec<Vec<f32>> = Vec::new();
    for cluster in 0..params.num_clusters {
        sums.push(Vec::new());
        denoms.push(Vec::new());
        for _index in 0..loci {
            sums[cluster].push(1.0);
            denoms[cluster].push(2.0); // psuedocounts
        }
    }
    //e qual prior
    let log_prior: f32 = (1.0/(params.num_clusters as f32)).ln();
    // current iteration
    let mut iterations = 0;
    let mut total_log_loss = f32::NEG_INFINITY;
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
        while (log_loss_change > log_loss_change_limit) && (iterations < 1000) {
            let mut log_binom_loss = 0.0;
            // reset sum and denoms
            reset_sums_denoms(loci, &mut sums, &mut denoms, params.num_clusters);
            let mut cell_khm_perfs = vec![];
            for (celldex, cell) in cell_data.iter().enumerate() {
                // get the bionomial loss for the cell with current cluster centers
                let log_binoms = binomial_loss(cell, &cluster_centers, log_prior);
                cell_khm_perfs.push((params.num_clusters as f32).ln() - calculate_khm_perf_for_cell(&log_binoms));
                // for total loss 
                log_binom_loss += log_sum_exp(&log_binoms);
                // temp determinstic annealing
                let mut temp = (cell.total_alleles / (TEMP * 2.0f32.powf(temp_step as f32))).max(1.0);
                if temp_step == temp_steps - 1 { temp = 1.0; }
                // apply temp for loss
                let adjusted_log_binoms = normalize_in_log_with_temp(&log_binoms, temp);
                // update sums and denoms
                update_centers_average(&mut sums, &mut denoms, cell, &adjusted_log_binoms);
                final_log_probabilities[celldex] = log_binoms;
            }
            let khm_log_loss = log_sum_exp(&cell_khm_perfs);
            total_log_loss = log_binom_loss;
            log_loss_change = log_binom_loss - last_log_loss;
            last_log_loss = log_binom_loss;
            // sums and denoms to update cluster centers
            update_final(loci, &sums, &denoms, &mut cluster_centers);
            iterations += 1;
            eprintln!("binomial\t{}\t{}\t{}\t{}\t{}\t{}\tkhmloss: {}", thread_num, epoch, iterations, temp_step, log_binom_loss, log_loss_change, khm_log_loss);
        }
    }
    (total_log_loss, final_log_probabilities)
}

fn binomial_loss(cell_data: &CellData, cluster_centers: &Vec<Vec<f32>>, log_prior: f32) -> Vec<f32> {
    let mut log_probabilities: Vec<f32> = Vec::new();
    for (cluster, center) in cluster_centers.iter().enumerate() {
        log_probabilities.push(log_prior);
        for (locus_index, locus) in cell_data.loci.iter().enumerate() {
            log_probabilities[cluster] += cell_data.log_binomial_coefficient[locus_index] + 
                (cell_data.alt_counts[locus_index] as f32) * center[*locus].ln() + 
                (cell_data.ref_counts[locus_index] as f32) * (1.0 - center[*locus]).ln();
        }
    }
    log_probabilities
}

fn log_sum_exp(p: &Vec<f32>) -> f32{
    let max_p: f32 = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_rst: f32 = p.iter().map(|x| (x - max_p).exp()).sum();
    max_p + sum_rst.ln()
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
            cluster_centers[cluster][locus] = update.min(0.99).max(0.01);
        }
    }
}

fn reset_sums_denoms(loci: usize, sums: &mut Vec<Vec<f32>>, 
    denoms: &mut Vec<Vec<f32>>, num_clusters: usize) {
    for cluster in 0..num_clusters {
        for index in 0..loci {
            sums[cluster][index] = 1.0;
            denoms[cluster][index] = 2.0;
        }
    }
}

fn update_centers_average(sums: &mut Vec<Vec<f32>>, denoms: &mut Vec<Vec<f32>>, cell: &CellData, probabilities: &Vec<f32>) {
    for locus in 0..cell.loci.len() {
        for (cluster, probability) in probabilities.iter().enumerate() {
            sums[cluster][cell.loci[locus]] += probability * (cell.alt_counts[locus] as f32);
            denoms[cluster][cell.loci[locus]] += probability * ((cell.alt_counts[locus] + cell.ref_counts[locus]) as f32);
        }
    }
}

fn init_cluster_centers(loci_used: usize, cell_data: &Vec<CellData>, params: &Params, rng: &mut StdRng) -> Vec<Vec<f32>> {
    match params.initialization_strategy {
        ClusterInit::KmeansPP => init_cluster_centers_kmeans_pp(loci_used, &cell_data, params, rng),
        ClusterInit::Overclustering => init_cluster_centers_overclustering(loci_used, &cell_data, params, rng),
        ClusterInit::RandomUniform => init_cluster_centers_uniform(loci_used, params, rng),
        ClusterInit::RandomAssignment => init_cluster_centers_random_assignment(loci_used, &cell_data, params, rng),
    }
}

pub fn reader(filename: &str) -> Box<dyn BufRead> {
    let path = Path::new(filename);
    let file = match File::open(&path) {
        Err(_why) => panic!("couldn't open file {}", filename),
        Ok(file) => file,
    };
    if path.extension() == Some(OsStr::new("gz")) {
        Box::new(BufReader::with_capacity(128 * 1024, MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::with_capacity(128 * 1024, file))
    }
}

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
    for cluster in 0..params.num_clusters * MULTIPLY_CLUS {
        original_centers.push(Vec::new());
        for _ in 0..loci {
            original_centers[cluster].push(rng.gen::<f32>().min(0.9999).max(0.0001));
        }
    }
    // weights for distance calculation
    let mut loci_weights = vec![0.0; loci];
    // go thorugh the 
    for (_index, cell) in cell_data.iter().enumerate() {
        // go thorugh the cell locations
        for locus in 0..cell.loci.len() {
            let alt_c = cell.alt_counts[locus] as f32;
            let total = alt_c + (cell.ref_counts[locus] as f32);
            let locus_index = cell.loci[locus];
            loci_weights[locus_index] += total;
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
                    let curr_dist = cluster_compare(cluster1, cluster2, &loci_weights);
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

fn cluster_compare (cluster1: &Vec<f32>, cluster2: &Vec<f32>, loci_weights: &Vec<f32>) -> f32 {
    let mut squared_dist = 0.0;
    // go thorugh each loci of the two clusters, and get the squared distance between them multiplied by the locus weight
    assert!(cluster1.len() == cluster2.len());
    for locus in 0..cluster1.len() {
        squared_dist += ((cluster1[locus] - cluster2[locus]) * loci_weights[locus]).powi(2);
    }
    squared_dist
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
}

fn load_params() -> Params {
    let yaml = load_yaml!("params.yml");
    let params = App::from_yaml(yaml).get_matches();
    let ref_mtx = params.value_of("ref_matrix").unwrap();
    let alt_mtx = params.value_of("alt_matrix").unwrap();
    let barcodes = params.value_of("barcodes").unwrap();
    let num_clusters = params.value_of("num_clusters").unwrap();
    let num_clusters = num_clusters.to_string().parse::<usize>().unwrap();
    let min_alt = params.value_of("min_alt").unwrap_or(READ_ALT_REF_MIN);
    let min_alt = min_alt.to_string().parse::<u32>().unwrap();
    let min_ref = params.value_of("min_ref").unwrap_or(READ_ALT_REF_MIN);
    let min_ref = min_ref.to_string().parse::<u32>().unwrap();
    let restarts = params.value_of("restarts").unwrap_or("100");
    let restarts = restarts.to_string().parse::<u32>().unwrap();
    let initialization_strategy = params.value_of("initialization_strategy").unwrap_or("random_uniform");
    //let initialization_strategy = params.value_of("initialization_strategy").unwrap_or("overcluster");
    let initialization_strategy = match initialization_strategy {
        "kmeans++" => ClusterInit::KmeansPP,
        "random_uniform" => ClusterInit::RandomUniform,
        "random_cell_assignment" => ClusterInit::RandomAssignment,
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
