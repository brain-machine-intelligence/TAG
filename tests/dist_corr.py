import numpy as np
from modules.env import MazeEnvironment
import parmap
import multiprocessing as mp
from utils.env import traverse_env
import os
from scipy.stats import spearmanr
from tqdm import tqdm
from itertools import permutations, product
import scipy.linalg as LA
from modules.base import SRMB


def calc_diffusion_dist(state_1, state_2, diffusion_map):
    '''
    diffusion_map: num_states x num_dims
    '''
    return np.sqrt(np.sum(np.power((diffusion_map[state_1] - diffusion_map[state_2]), 2)))


def calc_diffusion_map(eigenvecs, eigenvals_t, t, num_dims):
    '''
    eigenvecs: num_states x num_dims (full_dim)
    eigenvals_t: num_dims
    '''
    # Sort eigenvalues in descending order and get the sorted indices
    sorted_indices = np.argsort(eigenvals_t)[::-1]

    # Sort eigenvalues and eigenvectors accordingly
    eigenvals_t = eigenvals_t[sorted_indices]
    eigenvecs = eigenvecs[:, sorted_indices]
    return np.matmul(eigenvecs[:, 1:1 + num_dims], np.diag(np.power(eigenvals_t[1:1 + num_dims], t)))


def calc_successor_map(eigenvecs, eigenvals_t, gamma, num_dims):
    '''
    eigenvecs: num_states x num_dims (full_dim)
    eigenvals_t: num_dims
    '''
    # Sort eigenvalues in descending order and get the sorted indices
    sorted_indices = np.argsort(eigenvals_t)[::-1]

    # Sort eigenvalues and eigenvectors accordingly
    eigenvals_t = eigenvals_t[sorted_indices]
    eigenvecs = eigenvecs[:, sorted_indices]
    return np.matmul(eigenvecs[:, 1:1 + num_dims], np.diag(np.sqrt(1 / (np.ones(num_dims) - gamma * eigenvals_t[1:1 + num_dims]))))


def traverse(index, ratio_index):
    env = MazeEnvironment(10, 10, auto_reset=False)
    env.set_index("{}_{}".format(ratio_index, index))
    env.load_map("data/env/train/")
    gt_results, _ = traverse_env(env)
    return gt_results


def calculate_optimal_path_length():
    os.makedirs("data/dist_corr/raw_data/path_lengths/", exist_ok=True)
    index_list = list(range(1, 101))
    num_cores = mp.cpu_count()
    splitted_index_list = np.array_split(index_list, num_cores)
    for ratio_index in range(1, 11):
        results = parmap.map(
            traverse,
            splitted_index_list,
            ratio_index,
            pm_pbar=True,
            pm_processes=num_cores,
        )
        for splitted_index_index in range(num_cores):
            for index_index in range(len(splitted_index_list[splitted_index_index])):
                index = splitted_index_list[splitted_index_index][index_index]
                gt_results = results[splitted_index_index][index_index]
                np.save(
                    "data/dist_corr/raw_data/path_lengths/{}_{}".format(
                        ratio_index, index
                    ),
                    gt_results,
                )


def calc_optimal_geodist_per_topology(gamma=0.995):
    os.makedirs("data/dist_corr/raw_data/optimal/", exist_ok=True)
    env = MazeEnvironment(10, 10, auto_reset=False)
    for ratio_index in tqdm(range(1, 11)):
        for index in range(1, 101):
            env.set_index('{}_{}'.format(ratio_index, index))
            env.load_map('data/env/train/')
            optimal_value = np.eye(100)  # s, g
            optimal_geodesic_distance = np.zeros((100, 100)) # s, g
            gt_results = np.load('data/dist_corr/raw_data/path_lengths/{}_{}.npy'.format(ratio_index, index))
            for s, g, dist in gt_results:
                optimal_geodesic_distance[s, g] = dist
                optimal_value[s, g] = gamma ** dist
            max_dist = np.max(optimal_geodesic_distance)
            for b in env.blocks:
                optimal_geodesic_distance[b, :] = max_dist + 1
                optimal_geodesic_distance[:, b] = max_dist + 1
            np.save('data/dist_corr/raw_data/optimal/values_{}_{}.npy'.format(ratio_index, index), optimal_value)
            np.save('data/dist_corr/raw_data/optimal/geodesic_dists_{}_{}.npy'.format(
                ratio_index, index
            ), optimal_geodesic_distance)


def calc_diffusion_dist_per_topology(diffdist_functions_and_params):
    sr = SRMB(10, 10)
    env = MazeEnvironment(10, 10, auto_reset=False)
    for ratio_index in tqdm(range(1, 11)):
        for index in tqdm(range(1, 101)):
            for biased in [False, True]:
                biased_str = 'biased' if biased else 'unbiased'
                env.set_index('{}_{}'.format(ratio_index, index))
                env.load_map('data/env/train/')
                sr.set_index('{}_{}'.format(ratio_index, index))
                sr.load_map_structure('data/env/train/', rd_prob=0.5 if not biased else 0.9)
                transition_matrix = sr.transition_matrix
                eigenvals, eigenvecs = LA.eig(transition_matrix)
                eigenvals_t = np.real(eigenvals)
                eigenvecs = np.real(eigenvecs)

                for func, params_list in diffdist_functions_and_params:
                    for params in params_list:
                        diffusion_map = func(eigenvecs, eigenvals_t, *params) 
                        diffusion_distance = np.zeros((100, 100))  # s, g
                        for state_1, state_2 in permutations(env.nonblocks, 2):
                            diffusion_distance[state_1, state_2] = calc_diffusion_dist(
                                state_1, state_2, diffusion_map
                            )
                        max_dist = np.max(diffusion_distance)
                        for b in env.blocks:
                            diffusion_distance[b, :] = max_dist + 1
                            diffusion_distance[:, b] = max_dist + 1
                        params_str = [f'{p:.2f}' if isinstance(p, float) else str(p) for p in params]
                        folder_path = f'data/dist_corr/raw_data/diffusion_dists/{func.__name__}/{params_str[0]}_{params_str[1]}'

                        # 폴더가 존재하지 않으면 생성
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                        np.save(folder_path + f'/{ratio_index}_{index}_{biased_str}.npy', diffusion_distance)


def postprocess_value_eucliddist_corr():
    os.makedirs("data/dist_corr/", exist_ok=True)
    value_eucliddist_spearmanrs = []
    epsilon = 1e-10 
    env = MazeEnvironment(10, 10)
    optimal_eucliddist = np.zeros((100, 100))
    for start in range(100):
        for goal in range(100):
            start_x, start_y = env.state_to_index(start)
            goal_x, goal_y = env.state_to_index(goal)
            distance = np.sqrt((start_x - goal_x) ** 2 + (start_y - goal_y) ** 2)
            optimal_eucliddist[start, goal] = distance
    optimal_eucliddist += epsilon
    for ratio_index in tqdm(range(1, 11)):
        for index in range(1, 101):
            blocks = np.load(
                "data/env/train/blocks_{}_{}.npy".format(ratio_index, index)
            )
            nonblocks = np.setdiff1d(np.arange(100), blocks)
            optimal_value = np.load(
                "data/dist_corr/raw_data/optimal/values_{}_{}.npy".format(
                    ratio_index, index
                )
            )  # s, g
            value_eucliddist_spearmanr = 0
            for g in nonblocks:
                value_eucliddist_spearmanr += spearmanr(
                    optimal_value[:, g], 1 / optimal_eucliddist[:, g]
                )[0]
            value_eucliddist_spearmanrs.append(value_eucliddist_spearmanr / len(nonblocks))
    value_eucliddist_spearmanr_mean = np.mean(value_eucliddist_spearmanrs)
    value_eucliddist_spearmanr_std = np.std(value_eucliddist_spearmanrs)
    np.save(
        "data/dist_corr/value_eucliddist_spearmanr_mean.npy",
        value_eucliddist_spearmanr_mean,
    )
    np.save(
        "data/dist_corr/value_eucliddist_spearmanr_std.npy",
        value_eucliddist_spearmanr_std,
    )


def postprocess_value_diffdist_corr(diffdist_functions_and_params):
    for biased in [False, True]:
        biased_str = 'biased' if biased else 'unbiased'
        for func, params_list in diffdist_functions_and_params:
            for params in params_list:
                params_str = [f'{p:.2f}' if isinstance(p, float) else str(p) for p in params]
                folder_path = f'data/dist_corr/raw_data/diffusion_dists/{func.__name__}/{params_str[0]}_{params_str[1]}'
                value_diffdist_spearmanrs = []
                for ratio_index in tqdm(range(1, 11)):
                    for index in range(1, 101):
                        blocks = np.load('data/env/train/blocks_{}_{}.npy'.format(ratio_index, index))
                        nonblocks = np.setdiff1d(np.arange(100), blocks)
                        optimal_value = np.load('data/dist_corr/raw_data/optimal/values_{}_{}.npy'.format(ratio_index, index))  # s, g

                        diffdist = np.load(
                            folder_path + f"/{ratio_index}_{index}_{biased_str}.npy"
                        )
                        value_diffdist_spearmanr = 0
                        with np.errstate(divide='ignore', invalid='ignore'):
                            inv_diffdist = np.where(diffdist == 0, 100000.0, 1 / diffdist)
                        for g in nonblocks:
                            value_diffdist_spearmanr += spearmanr(
                                optimal_value[:, g], inv_diffdist[:, g]
                            )[0]
                        value_diffdist_spearmanrs.append(
                            value_diffdist_spearmanr / len(nonblocks)
                        )

                value_diffdist_spearmanr_mean = np.mean(value_diffdist_spearmanrs)
                value_diffdist_spearmanr_std = np.std(value_diffdist_spearmanrs)
                params_str = [f'{p:.2f}' if isinstance(p, float) else str(p) for p in params]
                np.save(f'data/dist_corr/value_diffdist_spearmanr_mean_{biased_str}_{func.__name__}_{params_str[0]}_{params_str[1]}.npy', value_diffdist_spearmanr_mean)
                np.save(
                    f"data/dist_corr/value_diffdist_spearmanr_std_{biased_str}_{func.__name__}_{params_str[0]}_{params_str[1]}.npy",
                    value_diffdist_spearmanr_std,
                )


def postprocess_value_sr_corr():
    sr = SRMB(10, 10)
    gamma_list = [0.1, 0.2, 0.3, 1/np.exp(1), 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.995]
    for biased in [False, True]:
        value_sr_spearmanrs = []
        biased_str = "biased" if biased else "unbiased"
        for gamma in gamma_list:
            sr.gamma = gamma
            for ratio_index in tqdm(range(1, 11)):
                for index in range(1, 101):
                    blocks = np.load('data/env/train/blocks_{}_{}.npy'.format(ratio_index, index))
                    nonblocks = np.setdiff1d(np.arange(100), blocks)
                    optimal_value = np.load(
                        "data/topology_value_corr/optimal/values_{}_{}.npy".format(
                            ratio_index, index
                        )
                    )  # s, g
                    sr.set_index("{}_{}".format(ratio_index, index))
                    sr.load_map_structure(
                        "data/env/train/",
                        rd_prob=0.5 if not biased else 0.9,
                    )
                    value_sr_spearmanr = 0
                    for g in nonblocks:
                        value_sr_spearmanr += spearmanr(
                            optimal_value[:, g], sr.sr[:, g]
                        )[0]
                    value_sr_spearmanrs.append(value_sr_spearmanr / len(nonblocks))
            value_sr_spearmanr_mean = np.mean(value_sr_spearmanrs)
            value_sr_spearmanr_std = np.std(value_sr_spearmanrs)
            gamma_str = f"{gamma:.2f}"
            np.save(
                f"data/topology_value_corr/value_sr_spearmanr_mean_{biased_str}_{gamma_str}.npy",
                value_sr_spearmanr_mean,
            )
            np.save(
                f"data/topology_value_corr/value_sr_spearmanr_std_{biased_str}_{gamma_str}.npy",
                value_sr_spearmanr_std,
            )


def check_optimal_path_length_data():
    """Check if optimal path length data files exist"""
    base_dir = "data/dist_corr/raw_data/path_lengths/"
    
    # Check if directory exists
    if not os.path.exists(base_dir):
        return False
    
    # Check for all required path length files (10 ratios x 100 indices = 1000 files)
    for ratio_index in range(1, 11):
        for index in range(1, 101):
            file_path = os.path.join(base_dir, f"{ratio_index}_{index}.npy")
            if not os.path.exists(file_path):
                return False
    
    return True


def check_optimal_geodist_data():
    """Check if optimal geodesic distance data files exist"""
    base_dir = "data/dist_corr/raw_data/optimal/"
    
    # Check if directory exists
    if not os.path.exists(base_dir):
        return False
    
    # Check for all required optimal files (10 ratios x 100 indices = 1000 files each)
    for ratio_index in range(1, 11):
        for index in range(1, 101):
            # Check values file
            values_file_path = os.path.join(base_dir, f"values_{ratio_index}_{index}.npy")
            if not os.path.exists(values_file_path):
                return False
            
            # Check geodesic distances file
            geodist_file_path = os.path.join(base_dir, f"geodesic_dists_{ratio_index}_{index}.npy")
            if not os.path.exists(geodist_file_path):
                return False
    
    return True


def check_data_availability_fig7():
    base_dir = "data/dist_corr/"
    
    # SR (Successor Representation) files - unbiased
    sr_filenames_mean = [
        "value_sr_spearmanr_mean_unbiased_0.99.npy",
        "value_sr_spearmanr_mean_unbiased_0.90.npy",
        "value_sr_spearmanr_mean_unbiased_0.80.npy",
        "value_sr_spearmanr_mean_unbiased_0.70.npy",
        "value_sr_spearmanr_mean_unbiased_0.60.npy",
        "value_sr_spearmanr_mean_unbiased_0.50.npy",
        "value_sr_spearmanr_mean_unbiased_0.37.npy",
        "value_sr_spearmanr_mean_unbiased_0.20.npy",
        "value_sr_spearmanr_mean_unbiased_0.10.npy",
    ]
    sr_filenames_std = [
        "value_sr_spearmanr_std_unbiased_0.99.npy",
        "value_sr_spearmanr_std_unbiased_0.90.npy",
        "value_sr_spearmanr_std_unbiased_0.80.npy",
        "value_sr_spearmanr_std_unbiased_0.70.npy",
        "value_sr_spearmanr_std_unbiased_0.60.npy",
        "value_sr_spearmanr_std_unbiased_0.50.npy",
        "value_sr_spearmanr_std_unbiased_0.37.npy",
        "value_sr_spearmanr_std_unbiased_0.20.npy",
        "value_sr_spearmanr_std_unbiased_0.10.npy",
    ]
    
    # Diffusion Distance files - unbiased successor map
    diff_filenames_mean = [
        "value_diffdist_spearmanr_mean_unbiased_calc_successor_map_0.99_99.npy",
        "value_diffdist_spearmanr_mean_unbiased_calc_successor_map_0.90_99.npy",
        "value_diffdist_spearmanr_mean_unbiased_calc_successor_map_0.80_99.npy",
        "value_diffdist_spearmanr_mean_unbiased_calc_successor_map_0.70_99.npy",
        "value_diffdist_spearmanr_mean_unbiased_calc_successor_map_0.60_99.npy",
        "value_diffdist_spearmanr_mean_unbiased_calc_successor_map_0.50_99.npy",
        "value_diffdist_spearmanr_mean_unbiased_calc_successor_map_0.37_99.npy",
        "value_diffdist_spearmanr_mean_unbiased_calc_successor_map_0.20_99.npy",
        "value_diffdist_spearmanr_mean_unbiased_calc_successor_map_0.10_99.npy",
    ]
    diff_filenames_std = [
        "value_diffdist_spearmanr_std_unbiased_calc_successor_map_0.99_99.npy",
        "value_diffdist_spearmanr_std_unbiased_calc_successor_map_0.90_99.npy",
        "value_diffdist_spearmanr_std_unbiased_calc_successor_map_0.80_99.npy",
        "value_diffdist_spearmanr_std_unbiased_calc_successor_map_0.70_99.npy",
        "value_diffdist_spearmanr_std_unbiased_calc_successor_map_0.60_99.npy",
        "value_diffdist_spearmanr_std_unbiased_calc_successor_map_0.50_99.npy",
        "value_diffdist_spearmanr_std_unbiased_calc_successor_map_0.37_99.npy",
        "value_diffdist_spearmanr_std_unbiased_calc_successor_map_0.20_99.npy",
        "value_diffdist_spearmanr_std_unbiased_calc_successor_map_0.10_99.npy",
    ]
    
    # Combine all required files
    required_files = (
        sr_filenames_mean + 
        sr_filenames_std + 
        diff_filenames_mean + 
        diff_filenames_std
    )
    
    # Check if all files exist
    for filename in required_files:
        file_path = os.path.join(base_dir, filename)
        if not os.path.exists(file_path):
            return False
    
    return True


def check_data_availability_fig1():
    eucliddist_mean_filename = 'value_eucliddist_spearmanr_mean.npy'
    eucliddist_std_filename = 'value_eucliddist_spearmanr_std.npy'

    succ_unbiased_filename_mean = "value_diffdist_spearmanr_mean_unbiased_calc_successor_map_0.99_99.npy"
    succ_unbiased_filename_std = "value_diffdist_spearmanr_std_unbiased_calc_successor_map_0.99_99.npy"

    diff_unbiased_filename_mean = (
        "value_diffdist_spearmanr_mean_unbiased_calc_diffusion_map_7_99.npy"
    )
    diff_unbiased_filename_std = (
        "value_diffdist_spearmanr_std_unbiased_calc_diffusion_map_7_99.npy"
    )
    geodist_mean_filename = 'value_geodist_spearmanr_mean.npy'
    geodist_std_filename = 'value_geodist_spearmanr_std.npy'

    sr_biased_mean_filename = 'value_sr_spearmanr_mean_biased_0.99.npy'
    sr_biased_std_filename = 'value_sr_spearmanr_std_biased_0.99.npy'

    sr_unbiased_mean_filename = 'value_sr_spearmanr_mean_unbiased_0.99.npy'
    sr_unbiased_std_filename = 'value_sr_spearmanr_std_unbiased_0.99.npy'

    succ_biased_mean_filename = 'value_diffdist_spearmanr_mean_biased_calc_successor_map_0.99_99.npy'
    succ_biased_std_filename = 'value_diffdist_spearmanr_std_biased_calc_successor_map_0.99_99.npy'

    base_dir = "data/dist_corr/"
    required_files = [
        eucliddist_mean_filename,
        eucliddist_std_filename,
        succ_unbiased_filename_mean,
        succ_unbiased_filename_std,
        diff_unbiased_filename_mean,
        diff_unbiased_filename_std,
        geodist_mean_filename,
        geodist_std_filename,
        sr_biased_mean_filename,
        sr_biased_std_filename,
        sr_unbiased_mean_filename,
        sr_unbiased_std_filename,
        succ_biased_mean_filename,
        succ_biased_std_filename
    ]
    
    for filename in required_files:
        file_path = os.path.join(base_dir, filename)
        if not os.path.exists(file_path):
            return False
    
    return True


def main_fig1():
    t_list = [7]
    num_dims_list = [99]
    gamma_list = [0.99]
    diffdist_functions_and_params = [
        (
            calc_successor_map,
            [
                (gamma_list[i], num_dims_list[j])
                for i, j in product(
                    range(len(gamma_list)), range(len(num_dims_list))
                )
            ],
        ),  # gamma, num_dims
        (
            calc_diffusion_map,
            [
                (t_list[i], num_dims_list[j])
                for i, j in product(range(len(t_list)), range(len(num_dims_list)))
            ],
        ),  # t, num_dims
    ]
    if not check_data_availability_fig1():
        if not check_optimal_path_length_data():
            calculate_optimal_path_length()
        if not check_optimal_geodist_data():
            calc_optimal_geodist_per_topology()
        calc_diffusion_dist_per_topology(diffdist_functions_and_params)
        postprocess_value_diffdist_corr(diffdist_functions_and_params)
        postprocess_value_eucliddist_corr()



def main_fig7():
    num_dims_list = [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    gamma_list = [0.1, 0.2, 0.3, 1/np.exp(1), 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    diffdist_functions_and_params = [
        (
            calc_successor_map,
            [
                (gamma_list[i], num_dims_list[j])
                for i, j in product(
                    range(len(gamma_list)), range(len(num_dims_list))
                )
            ],
        )
    ]
    if not check_data_availability_fig7():
        calc_diffusion_dist_per_topology(diffdist_functions_and_params)
        postprocess_value_diffdist_corr(diffdist_functions_and_params)
        postprocess_value_sr_corr()
