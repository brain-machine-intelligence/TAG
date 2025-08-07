import numpy as np
from modules.base import SRMB
from modules.env import MazeEnvironment
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau

# from utils.ot import ot_cost
from utils.ot import (
    calc_gromov_wasserstein_dist_alignment,
    calc_gromov_wasserstein_dist_graph,
    interpolate_transport_adjacent,
    calc_transport_dist_alignment,
    center_and_scale_diffusionmap,
)
from utils.topology import (
    calc_gh_dist_graph,
    calc_gh_dist_grid,
    create_matchers,
    generate_permutation_matrices,
    return_adjacent_nodes,
    avg_nodewise,
)
import networkx as nx
import argparse
from utils.skeletonize import skeletonize_env
import os, re, pickle
from collections import defaultdict
from math import ceil

EPSILON = 1e-9


def convert_to_integer_graph(G):
    # Step 1: Create a mapping from original nodes to integers
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}

    # Step 2: Create a new MultiGraph to maintain structure
    new_G = nx.MultiGraph()

    # Step 3: Add nodes with new integer labels
    new_G.add_nodes_from(node_mapping.values())

    # Step 4: Add edges with the new integer nodes and maintain original edge attributes
    for u, v, key, data in G.edges(data=True, keys=True):
        new_G.add_edge(node_mapping[u], node_mapping[v], key=key, **data)

    return new_G


def calc_gh_dist_graph_basis_10(overall_index_1):
    gh_dist_vec = np.ones(1000) * (-1)
    ratio_index_1 = (overall_index_1 - 1) // 100 + 1
    index_1 = (overall_index_1 - 1) % 100 + 1
    G1 = nx.MultiGraph(
        nx.read_graphml(
            f"data/skeletonize/skeleton_graph/train/{ratio_index_1}_{index_1}.graphml"
        )
    )
    G1 = convert_to_integer_graph(G1)
    for overall_index_2 in tqdm(range(1, 1001)):
        if overall_index_1 == overall_index_2:
            gh_dist_vec[overall_index_2 - 1] = 0
        else:
            ratio_index_2 = (overall_index_2 - 1) // 100 + 1
            index_2 = (overall_index_2 - 1) % 100 + 1
            G2 = nx.MultiGraph(
                nx.read_graphml(
                    f"data/skeletonize/skeleton_graph/train/{ratio_index_2}_{index_2}.graphml"
                )
            )
            G2 = convert_to_integer_graph(G2)
            try:
                gh_dist = calc_gh_dist_graph(G1, G2)
            except UnboundLocalError:
                gh_dist = np.nan
            gh_dist_vec[overall_index_2 - 1] = gh_dist
    np.save(
        f"data/isomorphism_distinguish/raw_data/gh_dist_graph_vec_{overall_index_1}.npy",
        gh_dist_vec,
    )


def calc_gw_dist_grid_basis_10(overall_index_1, dim=10):
    gw_dist_vec = np.ones(1000) * (-1)
    skel_gw_dist_vec = np.ones(1000) * (-1)
    srmb1 = SRMB(10, 10)
    ratio_index_1 = (overall_index_1 - 1) // 100 + 1
    index_1 = (overall_index_1 - 1) % 100 + 1
    srmb1.set_index(f"{ratio_index_1}_{index_1}")
    srmb1.load_map_structure("data/env/train/")
    eigenmap1 = srmb1.return_successormap()[:, :dim]
    env1 = MazeEnvironment(10, 10)
    env1.set_index(f"{ratio_index_1}_{index_1}")
    env1.load_map("data/env/train/")
    vertex_nodes1, deadend_nodes1, _, _, _, _, _, _, _ = skeletonize_env(env1)
    for overall_index_2 in tqdm(range(1, 1001)):
        if overall_index_1 == overall_index_2:
            skel_gw_dist_vec[overall_index_2 - 1] = 0
        else:
            ratio_index_2 = (overall_index_2 - 1) // 100 + 1
            index_2 = (overall_index_2 - 1) % 100 + 1
            srmb2 = SRMB(10, 10)
            srmb2.set_index(f"{ratio_index_2}_{index_2}")
            srmb2.load_map_structure("data/env/train/")
            eigenmap2 = srmb2.return_successormap()[:, :dim]
            env2 = MazeEnvironment(10, 10)
            env2.set_index(f"{ratio_index_2}_{index_2}")
            env2.load_map("data/env/train/")
            normalized_eigenmap1, eigenmap1_direct_scale_factor, eigenmap1_norms = (
                center_and_scale_diffusionmap(eigenmap1)
            )
            normalized_eigenmap2, eigenmap2_direct_scale_factor, eigenmap2_norms = (
                center_and_scale_diffusionmap(eigenmap2)
            )

            vertex_nodes2, deadend_nodes2, _, _, _, _, _, _, _ = skeletonize_env(env2)
            skel_eigenmap1 = avg_nodewise(eigenmap1, vertex_nodes1, deadend_nodes1)

            skel_eigenmap2 = avg_nodewise(eigenmap2, vertex_nodes2, deadend_nodes2)
            skel_gw_dist, _, _, _, _, _ = calc_gromov_wasserstein_dist_alignment(
                center_and_scale_diffusionmap(
                    skel_eigenmap1, direct_scale_factor=eigenmap1_direct_scale_factor
                )[0],
                center_and_scale_diffusionmap(
                    skel_eigenmap2, direct_scale_factor=eigenmap2_direct_scale_factor
                )[0],
            )
            skel_gw_dist_vec[overall_index_2 - 1] = skel_gw_dist
    np.save(
        f"data/isomorphism_distinguish/raw_data/gw_dist_grid_vec_{overall_index_1}.npy",
        gw_dist_vec,
    )
    np.save(
        f"data/isomorphism_distinguish/raw_data/skel_gw_dist_grid_vec_{overall_index_1}.npy",
        skel_gw_dist_vec,
    )


def postprocess_gw_dist_basis_10():
    gw_dist_grid_mat = np.zeros((1000, 1000))
    skel_gw_dist_grid_mat = np.zeros((1000, 1000))
    for overall_index_1 in tqdm(range(1, 1001)):
        gw_dist_grid_vec = np.load(
            f"data/isomorphism_distinguish/raw_data/gw_dist_grid_vec_{overall_index_1}.npy"
        )
        gw_dist_grid_mat[overall_index_1 - 1] = gw_dist_grid_vec
        skel_gw_dist_grid_vec = np.load(
            f"data/isomorphism_distinguish/raw_data/skel_gw_dist_grid_vec_{overall_index_1}.npy"
        )
        skel_gw_dist_grid_mat[overall_index_1 - 1] = skel_gw_dist_grid_vec
    np.save("data/isomorphism_distinguish/gw_dist_grid_mat.npy", gw_dist_grid_mat)
    np.save("data/isomorphism_distinguish/skel_gw_dist_grid_mat.npy", skel_gw_dist_grid_mat)


def postprocess_gh_dist_basis_10():
    gh_dist_graph_mat = np.zeros((1000, 1000))
    for overall_index_1 in tqdm(range(1, 1001)):
        gh_dist_graph_vec = np.load(
            f"data/isomorphism_distinguish/raw_data/gh_dist_graph_vec_{overall_index_1}.npy"
        )
        gh_dist_graph_mat[overall_index_1 - 1] = gh_dist_graph_vec
    np.save("data/isomorphism_distinguish/gh_dist_graph_mat.npy", gh_dist_graph_mat)


def check_data_availability():
    import os
    
    # Check for postprocessed output files
    required_files = [
        "data/isomorphism_distinguish/gw_dist_grid_mat.npy",
        "data/isomorphism_distinguish/skel_gw_dist_grid_mat.npy",
        "data/isomorphism_distinguish/gh_dist_graph_mat.npy"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            return False
    
    return True


def main():
    if not check_data_availability():
        for overall_index_1 in tqdm(range(1, 1001)):
            calc_gh_dist_graph_basis_10(overall_index_1)
            calc_gw_dist_grid_basis_10(overall_index_1)
        postprocess_gw_dist_basis_10()
        postprocess_gh_dist_basis_10()
    



def postprocess_gw_dist_graph_basis_10():
    # Create a single 3D tensor for all max_iter values
    gw_dist_graph_parametric_tensor = np.zeros((1000, 1000, 20))

    for overall_index_1 in tqdm(range(1, 1001)):
        try:
            # Load the vector data for this index that contains all max_iter values
            max_iter = 20
            gw_dist_graph_vec = np.load(
                f"data/isomorphism_distinguish/raw_data/gw_dist_graph_vec_{overall_index_1}_{max_iter}.npy"
            )
            # Store all max_iter values for this pair in the tensor
            # The max_iter dimension is zero-indexed (0-19)
            for iter in range(max_iter): 
                gw_dist_graph_parametric_tensor[
                    overall_index_1 - 1, :, iter
                ] = gw_dist_graph_vec[:, iter]
        except (FileNotFoundError, IndexError) as e:
            print(f"Error processing index {overall_index_1}: {e}")
            pass

    # Save the entire 3D tensor as a single file
    np.save(
        "data/isomorphism_distinguish/gw_dist_graph_parametric_tensor.npy",
        gw_dist_graph_parametric_tensor,
    )
    print(
        "Processed and combined all parametric GW distance matrices into a single tensor"
    )





def decompose_matrix(t_diff):
    # Get the shape of the matrix
    rows, cols = t_diff.shape
    
    # Find the row indices where there is any nonzero element in the corresponding vector for the row with a margin (epsilon 1e-10)
    row_indices = np.where(np.any(np.abs(t_diff) > EPSILON, axis=1))[0]
    C = np.array([[1 if x == i else 0 for x in range(cols)] for i in row_indices]).T
    R = np.array([t_diff[i] for i in row_indices])

    return C, R


def calculate_por(t, t_os, sr_os):
    C, R = decompose_matrix(t - t_os)
    assert np.sum(np.abs(t - t_os - (C @ R))) < EPSILON
    por = R @ sr_os @ C
    return np.linalg.inv(np.eye(por.shape[0]) - por)


def calculate_por_whole(t, t_os, sr_os):
    R = t - t_os
    C = np.eye(t.shape[1])
    por = R @ sr_os @ C
    return np.linalg.inv(np.eye(por.shape[0]) - por)


def calculate_pors_basis_10_train():
    sr_os = SRMB(10, 10)
    sr_os.update_map_structure()
    t_os = sr_os.transition_matrix
    sr = SRMB(10, 10)
    por_list = []
    for ratio_index in tqdm(range(1, 11)):
        for index in tqdm(range(1, 101)):
            sr.set_index(f"{ratio_index}_{index}")
            sr.load_map_structure('data/env/train/')
            t = sr.transition_matrix
            por = calculate_por_whole(t, t_os, sr_os.sr)
            por_list.append(por)
    np.save('data/isomorphism_distinguish/pors.npy', por_list)


def matrix_correlation(A, B, method='pearson'):
    """
    Compute the average correlation between corresponding row vectors of two matrices.

    Parameters:
        A (np.ndarray): First matrix (MxN).
        B (np.ndarray): Second matrix (MxN).
        method (str): Correlation method to use. Options: 
                      'pearson' (default), 'spearman', 'kendall'.

    Returns:
        float: Average correlation coefficient over all rows.
    """
    # 입력 검증
    if A.shape != B.shape:
        raise ValueError("Input matrices must have the same shape.")
    
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Input matrices must be 2D.")

    num_rows = A.shape[0]  # 행(row) 개수
    correlation_values = []

    for i in range(num_rows):
        A_row = A[i, :]
        B_row = B[i, :]

        if method == 'pearson':  # Pearson Correlation
            corr, _ = pearsonr(A_row, B_row)
        elif method == 'spearman':  # Spearman Rank Correlation
            corr, _ = spearmanr(A_row, B_row)
        elif method == 'kendall':  # Kendall Tau Correlation
            corr, _ = kendalltau(A_row, B_row)
        else:
            raise ValueError("Unsupported correlation method. Choose from 'pearson', 'spearman', or 'kendall'.")
        
        correlation_values.append(corr)

    # 평균 상관계수 반환
    return np.mean(correlation_values)


def matrix_distance(A, B, norm_type='fro'):
    """
    Compute the distance between two matrices A and B using different norms.
    
    Parameters:
        A (np.ndarray): First matrix.
        B (np.ndarray): Second matrix.
        norm_type (str): Type of norm to use. Options: 
                         'fro' (Frobenius, default), 
                         'l1' (L1 norm), 
                         'l2' (L2 norm), 
                         'linf' (L-infinity norm).
    
    Returns:
        float: Distance between A and B based on the chosen norm.
    """
    # 입력 검증
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")

    # 차이 행렬 계산
    diff = A - B
    
    # Norm 계산
    if norm_type == 'fro':  # Frobenius norm
        return np.linalg.norm(diff, 'fro')
    elif norm_type == 'l1':  # L1 norm (Manhattan norm)
        return np.linalg.norm(diff, 1)
    elif norm_type == 'l2':  # L2 norm (Spectral norm)
        return np.linalg.norm(diff, 2)
    elif norm_type == 'linf':  # L-infinity norm (Maximum norm)
        return np.linalg.norm(diff, np.inf)
    else:
        raise ValueError("Unsupported norm type. Choose from 'fro', 'l1', 'l2', or 'linf'.")


def calculate_por_distances_basis_10_train(overall_index_1):
    por_list = np.load('data/isomorphism_distinguish/pors.npy')
    por_dist_pearsonr = np.zeros(1000)
    por_dist_spearmanr = np.zeros(1000)
    por_dist_kendalltau = np.zeros(1000)
    por_dist_fro = np.zeros(1000)
    por_dist_l1 = np.zeros(1000)
    por_dist_l2 = np.zeros(1000)
    por_dist_linf = np.zeros(1000)
    size = 10  # 10x10 maze
    rotated_indices = np.array([(9 - (i % size)) * size + (i // size) for i in range(size * size)])  # rotate 90 degrees CCW
    for overall_index_2 in tqdm(range(1000)):
        por_i = por_list[overall_index_1 - 1]
        por_j = por_list[overall_index_2]
        max_pearsonr = -1
        max_spearmanr = -1
        max_kendalltau = -1
        min_fro = float('inf')
        min_l1 = float('inf')
        min_l2 = float('inf')
        min_linf = float('inf')
        for i in range(4):
            max_pearsonr = max(max_pearsonr, matrix_correlation(por_i, por_j, method='pearson'))
            max_spearmanr = max(max_spearmanr, matrix_correlation(por_i, por_j, method='spearman'))
            max_kendalltau = max(max_kendalltau, matrix_correlation(por_i, por_j, method='kendall'))
            min_fro = min(min_fro, matrix_distance(por_i, por_j, norm_type='fro'))
            min_l1 = min(min_l1, matrix_distance(por_i, por_j, norm_type='l1'))
            min_l2 = min(min_l2, matrix_distance(por_i, por_j, norm_type='l2'))
            min_linf = min(min_linf, matrix_distance(por_i, por_j, norm_type='linf'))
            if i < 3:
                por_j = por_j[np.ix_(rotated_indices, rotated_indices)]
        por_dist_pearsonr[overall_index_2] = max_pearsonr
        por_dist_spearmanr[overall_index_2] = max_spearmanr
        por_dist_kendalltau[overall_index_2] = max_kendalltau
        por_dist_fro[overall_index_2] = min_fro
        por_dist_l1[overall_index_2] = min_l1
        por_dist_l2[overall_index_2] = min_l2
        por_dist_linf[overall_index_2] = min_linf
    np.save(f'data/isomorphism_distinguish/raw_data/por_dist_pearsonr_{overall_index_1}.npy', por_dist_pearsonr)
    np.save(f'data/isomorphism_distinguish/raw_data/por_dist_spearmanr_{overall_index_1}.npy', por_dist_spearmanr)
    np.save(f'data/isomorphism_distinguish/raw_data/por_dist_kendalltau_{overall_index_1}.npy', por_dist_kendalltau)
    np.save(f'data/isomorphism_distinguish/raw_data/por_dist_fro_{overall_index_1}.npy', por_dist_fro)
    np.save(f'data/isomorphism_distinguish/raw_data/por_dist_l1_{overall_index_1}.npy', por_dist_l1)
    np.save(f'data/isomorphism_distinguish/raw_data/por_dist_l2_{overall_index_1}.npy', por_dist_l2)
    np.save(f'data/isomorphism_distinguish/raw_data/por_dist_linf_{overall_index_1}.npy', por_dist_linf)


def calc_por_dist_basis_10_cluster(start_index, end_index):
    for overall_index_1 in tqdm(range(start_index, end_index)):
        calculate_por_distances_basis_10_train(overall_index_1)


def postprocess_por_dist_basis_10():
    por_dist_pearsonr_mat = np.zeros((1000, 1000))
    por_dist_spearmanr_mat = np.zeros((1000, 1000))
    por_dist_kendalltau_mat = np.zeros((1000, 1000))
    por_dist_fro_mat = np.zeros((1000, 1000))
    por_dist_l1_mat = np.zeros((1000, 1000))
    por_dist_l2_mat = np.zeros((1000, 1000))
    por_dist_linf_mat = np.zeros((1000, 1000))
    for overall_index_1 in tqdm(range(1, 1001)):
        por_dist_pearsonr_vec = np.load('data/isomorphism_distinguish/raw_data/por_dist_pearsonr_{}.npy'.format(overall_index_1))
        por_dist_spearmanr_vec = np.load('data/isomorphism_distinguish/raw_data/por_dist_spearmanr_{}.npy'.format(overall_index_1))
        por_dist_kendalltau_vec = np.load('data/isomorphism_distinguish/raw_data/por_dist_kendalltau_{}.npy'.format(overall_index_1))
        por_dist_fro_vec = np.load('data/isomorphism_distinguish/raw_data/por_dist_fro_{}.npy'.format(overall_index_1))
        por_dist_l1_vec = np.load('data/isomorphism_distinguish/raw_data/por_dist_l1_{}.npy'.format(overall_index_1))
        por_dist_l2_vec = np.load('data/isomorphism_distinguish/raw_data/por_dist_l2_{}.npy'.format(overall_index_1))
        por_dist_linf_vec = np.load('data/isomorphism_distinguish/raw_data/por_dist_linf_{}.npy'.format(overall_index_1))
        por_dist_pearsonr_mat[overall_index_1 - 1] = por_dist_pearsonr_vec
        por_dist_spearmanr_mat[overall_index_1 - 1] = por_dist_spearmanr_vec
        por_dist_kendalltau_mat[overall_index_1 - 1] = por_dist_kendalltau_vec
        por_dist_fro_mat[overall_index_1 - 1] = por_dist_fro_vec
        por_dist_l1_mat[overall_index_1 - 1] = por_dist_l1_vec
        por_dist_l2_mat[overall_index_1 - 1] = por_dist_l2_vec
        por_dist_linf_mat[overall_index_1 - 1] = por_dist_linf_vec
    np.save('data/isomorphism_distinguish/por_dist_pearsonr_mat.npy', por_dist_pearsonr_mat)
    np.save('data/isomorphism_distinguish/por_dist_spearmanr_mat.npy', por_dist_spearmanr_mat)
    np.save('data/isomorphism_distinguish/por_dist_kendalltau_mat.npy', por_dist_kendalltau_mat)
    np.save('data/isomorphism_distinguish/por_dist_fro_mat.npy', por_dist_fro_mat)
    np.save('data/isomorphism_distinguish/por_dist_l1_mat.npy', por_dist_l1_mat)
    np.save('data/isomorphism_distinguish/por_dist_l2_mat.npy', por_dist_l2_mat)
    np.save('data/isomorphism_distinguish/por_dist_linf_mat.npy', por_dist_linf_mat)

