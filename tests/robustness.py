from modules.env import MazeEnvironment
from modules.base import SRMB
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.stats import spearmanr
from tqdm import tqdm
from itertools import product
import networkx as nx
from utils.skeletonize import skeletonize_env
from utils.ot import calc_ot_post_procrustes, calc_gromov_wasserstein_dist_alignment, calc_ot_without_rotation, center_and_scale_diffusionmap
from utils.topology import avg_nodewise
from utils.plot import scatter_multiple
import os
from scipy.stats import ttest_rel
from utils.topology import identify_env_pairs_basis10_train_wo_subgraph


def reindex_matrix_column_wise(matrix, rotation_factor, env):
    matrix_original = copy.deepcopy(matrix)
    num_states = matrix.shape[0]
    for _ in range(rotation_factor):
        matrix_rotated = np.zeros((num_states, matrix.shape[1]))
        for s in range(num_states):
            srow, scol = env.state_to_index(s)
            new_s = env.index_to_state(env.num_column - 1 - scol, srow, env.num_row)
            matrix_rotated[new_s, :] = matrix_original[s, :]
        matrix_original = matrix_rotated
    return matrix_original


def interpolate_matrix_column_wise(matrix, scale_factor, original_env):
    rows, cols = matrix.shape
    new_rows = rows * (scale_factor**2)
    new_matrix = np.zeros((new_rows, cols), dtype=matrix.dtype)

    # Precompute row areas
    row_areas = []
    for s in range(rows):
        srow, scol = original_env.state_to_index(s)
        row_area = [
            original_env.index_to_state(
                srow * scale_factor + r_diff,
                scol * scale_factor + c_diff,
                original_env.num_column * scale_factor,
            )
            for r_diff in range(scale_factor)
            for c_diff in range(scale_factor)
        ]
        row_areas.append(np.array(row_area))

    # Fill the new matrix using precomputed row areas
    for s in range(rows):
        for d in range(cols):
            new_matrix[row_areas[s], d] = matrix[s, d]

    return new_matrix


def original_env_states_in_shifted_env(env_original, row_shift, column_shift):
    states_in_shifted_env = []
    for s in range(env_original.num_states):
        r, c = env_original.state_to_index(s)
        row_start = max(0, row_shift)
        column_start = max(0, column_shift)
        num_column_shifted = env_original.num_column + abs(column_shift)
        states_in_shifted_env.append(
            env_original.index_to_state(
                r + row_start, c + column_start, num_column_shifted
            )
        )
    return np.array(states_in_shifted_env)


def shift_topology_state_list(node_list, row_shift, column_shift, env, env_shifted):
    new_node_list = []
    for n in node_list:
        nr, nc = env.state_to_index(n)
        nnr = nr + row_shift if row_shift > 0 else nr
        nnc = nc + column_shift if column_shift > 0 else nc
        new_node_list.append(
            env_shifted.index_to_state(
                nnr, nnc
            )
        )
        if nr == 0 and 0 < nc < env.num_column - 1:
            if row_shift > 0:
                for i in range(1, row_shift + 1):
                    new_node_list.append(
                        env_shifted.index_to_state(
                            nnr - i, nnc
                        )
                    )
        elif nr == env.num_row - 1 and 0 < nc < env.num_column - 1:
            if row_shift < 0:
                for i in range(1, abs(row_shift) + 1):
                    new_node_list.append(
                        env_shifted.index_to_state(
                            nnr + i, nnc
                        )
                    )
        elif 0 < nr < env.num_row - 1 and nc == 0:
            if column_shift > 0:
                for i in range(1, column_shift + 1):
                    new_node_list.append(
                        env_shifted.index_to_state(
                            nnr, nnc - i
                        )
                    )
        elif 0 < nr < env.num_row - 1 and nc == env.num_column - 1:
            if column_shift < 0:
                for i in range(1, abs(column_shift) + 1):
                    new_node_list.append(
                        env_shifted.index_to_state(
                            nnr, nnc + i
                        )
                    )
        elif nr == 0 and nc == 0:
            if row_shift >= 0 and column_shift >= 0:
                for i in range(row_shift + 1):
                    for j in range(column_shift + 1):
                        if i == 0 and j == 0:
                            continue
                        new_node_list.append(
                            env_shifted.index_to_state(
                                nnr - i, nnc - j
                            )
                        )
        elif nr == 0 and nc == env.num_column - 1:
            if row_shift >= 0 and column_shift <= 0:
                for i in range(row_shift + 1):
                    for j in range(abs(column_shift) + 1):
                        if i == 0 and j == 0:
                            continue
                        new_node_list.append(
                            env_shifted.index_to_state(
                                nnr - i, nnc + j
                            )
                        )
        elif nr == env.num_row - 1 and nc == 0:
            if row_shift <= 0 and column_shift >= 0:
                for i in range(abs(row_shift) + 1):
                    for j in range(column_shift + 1):
                        new_node_list.append(
                            env_shifted.index_to_state(
                                nnr + i, nnc - j
                            )
                        )
        elif nr == env.num_row - 1 and nc == env.num_column - 1:
            if row_shift <= 0 and column_shift <= 0:
                for i in range(abs(row_shift) + 1):
                    for j in range(abs(column_shift) + 1):
                        new_node_list.append(
                            env_shifted.index_to_state(
                                nnr + i, nnc + j
                            )
                        )
        else:
            pass

    return new_node_list


def process_shift(ratio_index, index, max_shift, dim):
    srmb = SRMB(10, 10)
    env = MazeEnvironment(10, 10, auto_reset=False)

    grid_cost_gw_list = np.zeros((2 * max_shift + 1, 2 * max_shift + 1))
    skel_grid_cost_gw_list = np.zeros((2 * max_shift + 1, 2 * max_shift + 1))

    env.set_index("{}_{}".format(ratio_index, index))
    env.load_map("data/env/train/")
    srmb.update_map_structure(env.blocks)

    (
        vertex_nodes, deadend_nodes, edge_nodes, G, image_original, 
        skeleton, vertex_corresp, deadend_corresp, edge_corresp, edge_dict, 
        original_G, closest_skeleton_indices
    ) = skeletonize_env(env)
    if G is None:
        return 0

    eigenmap = srmb.return_successormap()

    skel_eigenmap = avg_nodewise(eigenmap, vertex_nodes, deadend_nodes)
    sr_avg_nodewise = avg_nodewise(
        avg_nodewise(srmb.sr, vertex_nodes, deadend_nodes).T,
        vertex_nodes,
        deadend_nodes,
    ).T

    for row_shift in range(-max_shift, max_shift + 1):
        for column_shift in range(-max_shift, max_shift + 1):
            if row_shift == 0 and column_shift == 0:
                continue

            env_shifted = copy.deepcopy(env)
            env_shifted.shift_map_w_expansion((row_shift, column_shift))


            vertex_nodes_shifted = []
            for vertex_node_list in vertex_nodes:
                new_vertex_node_list = shift_topology_state_list(vertex_node_list, row_shift, column_shift, env, env_shifted)
                vertex_nodes_shifted.append(new_vertex_node_list)
            
            deadend_nodes_shifted = []
            for deadend_node_list in deadend_nodes:
                new_deadend_node_list = shift_topology_state_list(deadend_node_list, row_shift, column_shift, env, env_shifted)
                deadend_nodes_shifted.append(new_deadend_node_list)

            vertex_corresp_shifted = []
            for vcr, vcc in vertex_corresp:
                nvcr = vcr + row_shift if row_shift > 0 else vcr
                nvcc = vcc + column_shift if column_shift > 0 else vcc
                vertex_corresp_shifted.append((nvcr, nvcc))
            vertex_corresp_shifted = tuple(vertex_corresp_shifted)

            deadend_corresp_shifted = []
            for dcr, dcc in deadend_corresp:
                ndcr = dcr + row_shift if row_shift > 0 else dcr
                ndcc = dcc + column_shift if column_shift > 0 else dcc
                deadend_corresp_shifted.append((ndcr, ndcc))
            deadend_corresp_shifted = tuple(deadend_corresp_shifted)

            blocks = env_shifted.return_blocks()
            srmb_shifted = SRMB(env_shifted.num_row, env_shifted.num_column)
            srmb_shifted.update_map_structure(blocks=blocks)


            eigenmap_shifted = srmb_shifted.return_successormap()

            original_states = original_env_states_in_shifted_env(
                env, row_shift, column_shift
            )

            skel_eigenmap_shifted = avg_nodewise(
                eigenmap_shifted, vertex_nodes_shifted, deadend_nodes_shifted
            )
            sr_shifted_avg_nodewise = avg_nodewise(
                avg_nodewise(
                    srmb_shifted.sr, vertex_nodes_shifted, deadend_nodes_shifted
                ).T,
                vertex_nodes_shifted,
                deadend_nodes_shifted,
            ).T


            eigenmap_normalized, eigenmap_direct_scale_factor, eigenmap_scaling_norms = center_and_scale_diffusionmap(eigenmap)
            eigenmap_shifted_normalized, eigenmap_shifted_direct_scale_factor, eigenmap_shifted_scaling_norms = center_and_scale_diffusionmap(eigenmap_shifted[original_states], provided_norms=eigenmap_scaling_norms)

            gw_cost, transport_plan, _, _, _, _ = (
                calc_gromov_wasserstein_dist_alignment(
                    eigenmap_normalized, eigenmap_shifted_normalized
                )
            )
            grid_cost_gw_list[row_shift + max_shift][column_shift + max_shift] = gw_cost
            # grid_transport_plan_gw_list.append(transport_plan)



            skel_gw_cost, skel_transport_plan, _, _, _, _ = (
                calc_gromov_wasserstein_dist_alignment(
                    center_and_scale_diffusionmap(skel_eigenmap, direct_scale_factor=eigenmap_direct_scale_factor)[0], center_and_scale_diffusionmap(skel_eigenmap_shifted, direct_scale_factor=eigenmap_shifted_direct_scale_factor)[0]
                )
            )
            skel_grid_cost_gw_list[row_shift + max_shift][
                column_shift + max_shift
            ] = skel_gw_cost
            # skel_grid_transport_plan_gw_list.append(skel_transport_plan)

    np.save(
        f"data/robustness/shift/raw_data/grid_cost_gw_{ratio_index}_{index}.npy",
        grid_cost_gw_list,
    )
    np.save(
        f"data/robustness/shift/raw_data/skel_grid_cost_gw_{ratio_index}_{index}.npy",
        skel_grid_cost_gw_list,
    )



def test_shift(max_shift=2):
    dim = 10
    if max_shift != 2:
        raise NotImplementedError("Shift test on other than 2 is not yet implemented.")
    ratio_index_list = list(range(1, 11))
    index_list = list(range(1, 101))
    pairs = list(product(ratio_index_list, index_list))
    for pair in tqdm(pairs):
        process_shift(pair[0], pair[1], max_shift, dim)


def is_valid_list(lst):
    try:
        return not np.any(np.isnan(lst)) and len(lst) > 0
    except TypeError:
        for l in lst:
            if np.any(np.isnan(l)) or len(l) == 0:
                return False
        return True

def postprocess_shift():
    grid_cost_gw_list_list = []
    skel_grid_cost_gw_list_list = []
    for ratio_index in tqdm(range(1, 11)):
        for index in range(1, 101):
            G = nx.MultiGraph(
                nx.read_graphml(
                    f"data/skeletonize/skeleton_graph/train/{ratio_index}_{index}.graphml"
                )
            )
            if G.number_of_nodes() == 1:
                continue
            grid_cost_gw_list = np.load(
                "data/robustness/shift/raw_data/grid_cost_gw_{}_{}.npy".format(
                    ratio_index, index
                )
            )

            skel_grid_cost_gw_list = np.load(
                "data/robustness/shift/raw_data/skel_grid_cost_gw_{}_{}.npy".format(
                    ratio_index, index
                )
            )
            if is_valid_list(grid_cost_gw_list):
                grid_cost_gw_list_list.append(grid_cost_gw_list)

            if is_valid_list(skel_grid_cost_gw_list):
                skel_grid_cost_gw_list_list.append(skel_grid_cost_gw_list)


    grid_cost_gw_list_list = np.array(grid_cost_gw_list_list)
    skel_grid_cost_gw_list_list = np.array(skel_grid_cost_gw_list_list)


    def calculate_group_index(j, k):
        value = (j - 2) ** 2 + (k - 2) ** 2
        group_mapping = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 8: 5}
        return group_mapping[value]


    # Initialize the arrays to process
    arrays_to_process = [
        grid_cost_gw_list_list,
        skel_grid_cost_gw_list_list,
    ]

    # Corresponding names for the arrays
    array_names = [
        "grid_cost_gw",
        "skel_grid_cost_gw",
    ]

    all_group_means = []
    all_group_stds = []
    all_group_sems = []

    for array in arrays_to_process:
        grouped_values = {i: [] for i in range(6)}
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                for k in range(array.shape[2]):
                    group_index = calculate_group_index(j, k)
                    grouped_values[group_index].append(array[i, j, k])

        # Calculate the mean and standard deviation for each group
        group_means = [np.mean(grouped_values[i], axis=0) for i in range(6)]
        groups_stds = [np.std(grouped_values[i], axis=0) for i in range(6)]
        group_sems = [np.std(grouped_values[i], axis=0, ddof=1) / np.sqrt(np.shape(grouped_values[i])[0]) for i in range(6)]

        # Store the results
        all_group_means.append(group_means)
        all_group_stds.append(groups_stds)
        all_group_sems.append(group_sems)
    
    for idx, array_name in enumerate(array_names):
        np.save(
            f"data/robustness/shift/{array_name}_mean.npy",
            all_group_means[idx],
        )
        np.save(
            f"data/robustness/shift/{array_name}_std.npy",
            all_group_stds[idx],
        )
        np.save(
            f"data/robustness/shift/{array_name}_sem.npy",
            all_group_sems[idx],
        )


def process_scale(ratio_index, index, max_scale, dim):
    srmb = SRMB(10, 10)
    env = MazeEnvironment(10, 10, auto_reset=False)

    grid_cost_normalized_gw_list = np.zeros(max_scale)
    grid_cost_gw_list = np.zeros(max_scale)
    skel_grid_cost_normalized_gw_list = np.zeros(max_scale)
    skel_grid_cost_gw_list = np.zeros(max_scale)

    env.set_index("{}_{}".format(ratio_index, index))
    env.load_map("data/env/train/")
    srmb.update_map_structure(env.blocks)

    (
        vertex_nodes, deadend_nodes, edge_nodes, G, image_original, skeleton, 
        vertex_corresp, deadend_corresp, edge_corresp, edge_dict, original_G, closest_skeleton_indices
    ) = skeletonize_env(env)
    if G is None:
        return 0

    eigenmap = srmb.return_successormap()

    skel_eigenmap = avg_nodewise(eigenmap, vertex_nodes, deadend_nodes)
    sr_avg_nodewise = avg_nodewise(
        avg_nodewise(srmb.sr, vertex_nodes, deadend_nodes).T,
        vertex_nodes,
        deadend_nodes,
    ).T

    for scale_factor in range(1, max_scale + 1):
        if scale_factor == 1:
            continue

        else:
            env_scaled = copy.deepcopy(env)
            env_scaled.scale_map(scale_factor)

            vertex_nodes_scaled = []
            for vertex_node_list in vertex_nodes:
                new_vertex_node_list = []
                for v in vertex_node_list:
                    vr, vc = env.state_to_index(v)
                    for dvr in range(scale_factor):
                        for dvc in range(scale_factor):
                            new_vertex_node_list.append(
                                env_scaled.index_to_state(
                                    vr * scale_factor + dvr, vc * scale_factor + dvc
                                )
                            )
                vertex_nodes_scaled.append(new_vertex_node_list)

            deadend_nodes_scaled = []
            for deadend_node_list in deadend_nodes:
                new_deadend_node_list = []
                for d in deadend_node_list:
                    dr, dc = env.state_to_index(d)
                    for ddr in range(scale_factor):
                        for ddc in range(scale_factor):
                            new_deadend_node_list.append(
                                env_scaled.index_to_state(
                                    dr * scale_factor + ddr, dc * scale_factor + ddc
                                )
                            )
                deadend_nodes_scaled.append(new_deadend_node_list)

            vertex_corresp_scaled = []
            for vcr, vcc in vertex_corresp:
                vertex_corresp_scaled.append(
                    (vcr * scale_factor, vcc * scale_factor)
                )
            vertex_corresp_scaled = tuple(vertex_corresp_scaled)

            deadend_corresp_scaled = []
            for dcr, dcc in deadend_corresp:
                deadend_corresp_scaled.append(
                    (dcr * scale_factor, dcc * scale_factor)
                )
            deadend_corresp_scaled = tuple(deadend_corresp_scaled)

            blocks = env_scaled.return_blocks()
            srmb_scaled = SRMB(env_scaled.num_row, env_scaled.num_column)
            srmb_scaled.update_map_structure(blocks=blocks)
            eigenmap_scaled = srmb_scaled.return_successormap()

            skel_eigenmap_scaled = avg_nodewise(
                eigenmap_scaled, vertex_nodes_scaled, deadend_nodes_scaled
            )

            gw_cost, transport_plan, _, _, _, _ = (
                calc_gromov_wasserstein_dist_alignment(eigenmap, eigenmap_scaled)
            )

            grid_cost_gw_list[scale_factor - 1] = gw_cost
            # grid_transport_plan_gw_list.append(transport_plan)
            eigenmap_normalized, eigenmap_direct_scale_factor, eigenmap_scaling_norms = center_and_scale_diffusionmap(eigenmap)
            eigenmap_scaled_normalized, eigenmap_scaled_direct_scale_factor, eigenmap_scaled_scaling_norms = center_and_scale_diffusionmap(eigenmap_scaled, provided_norms=eigenmap_scaling_norms)

            normalized_gw_cost, _, _, _, _, _ = (
                calc_gromov_wasserstein_dist_alignment(eigenmap_normalized, eigenmap_scaled_normalized)
            )
            grid_cost_normalized_gw_list[scale_factor - 1] = normalized_gw_cost



            skel_gw_cost, skel_transport_plan, _, _, _, _ = (
                calc_gromov_wasserstein_dist_alignment(
                    skel_eigenmap, skel_eigenmap_scaled
                )
            )
            skel_grid_cost_gw_list[scale_factor - 1] = skel_gw_cost
            # skel_grid_transport_plan_gw_list.append(skel_transport_plan)

            normalized_skel_gw_cost, _, _, _, _, _ = (
                calc_gromov_wasserstein_dist_alignment(
                    center_and_scale_diffusionmap(skel_eigenmap, direct_scale_factor=eigenmap_direct_scale_factor)[0], center_and_scale_diffusionmap(skel_eigenmap_scaled, direct_scale_factor=eigenmap_scaled_direct_scale_factor)[0]
                )
            )
            skel_grid_cost_normalized_gw_list[scale_factor - 1] = normalized_skel_gw_cost


    np.save(
        f"data/robustness/scale/raw_data/grid_cost_gw_{ratio_index}_{index}.npy",
        grid_cost_gw_list,
    )
    np.save(
        f"data/robustness/scale/raw_data/skel_grid_cost_gw_{ratio_index}_{index}.npy",
        skel_grid_cost_gw_list,
    )
    np.save(
        f"data/robustness/scale/raw_data/grid_cost_normalized_gw_{ratio_index}_{index}.npy",
        grid_cost_normalized_gw_list,
    )
    np.save(
        f"data/robustness/scale/raw_data/skel_grid_cost_normalized_gw_{ratio_index}_{index}.npy",
        skel_grid_cost_normalized_gw_list,
    )


def test_scale(max_scale=5):
    dim = 10
    if max_scale != 5:
        raise NotImplementedError("Shift test on other than 2 is not yet implemented.")
    ratio_index_list = list(range(1, 11))
    index_list = list(range(1, 101))
    pairs = list(product(ratio_index_list, index_list))
    for pair in tqdm(pairs):
        process_scale(pair[0], pair[1], max_scale, dim)



def postprocess_scale():

    grid_cost_gw_list_list = []
    grid_cost_normalized_gw_list_list = []
    skel_grid_cost_gw_list_list = []
    skel_grid_cost_normalized_gw_list_list = []

    corresponding_indices = []

    for ratio_index in tqdm(range(1, 11)):
        for index in range(1, 101):
            overall_index = (ratio_index - 1) * 100 + index
            G = nx.MultiGraph(
                nx.read_graphml(
                    f"data/skeletonize/skeleton_graph/train/{ratio_index}_{index}.graphml"
                )
            )
            if G.number_of_nodes() == 1:
                continue

            grid_cost_gw_list = np.load(
                "data/robustness/scale/raw_data/grid_cost_gw_{}_{}.npy".format(
                    ratio_index, index
                )
            )
            grid_cost_normalized_gw_list = np.load(
                "data/robustness/scale/raw_data/grid_cost_normalized_gw_{}_{}.npy".format(
                    ratio_index, index
                )
            )

            skel_grid_cost_gw_list = np.load(
                "data/robustness/scale/raw_data/skel_grid_cost_gw_{}_{}.npy".format(
                    ratio_index, index
                )
            )
            skel_grid_cost_normalized_gw_list = np.load(
                "data/robustness/scale/raw_data/skel_grid_cost_normalized_gw_{}_{}.npy".format(
                    ratio_index, index
                )
            )

            if is_valid_list(grid_cost_gw_list):
                grid_cost_gw_list_list.append(grid_cost_gw_list)
            if is_valid_list(grid_cost_normalized_gw_list):
                grid_cost_normalized_gw_list_list.append(grid_cost_normalized_gw_list)
            # grid_transport_plan_gw_list_list.append(grid_transport_plan_gw_list)
            if is_valid_list(skel_grid_cost_gw_list):
                skel_grid_cost_gw_list_list.append(skel_grid_cost_gw_list)
                corresponding_indices.append((ratio_index, index))
            if is_valid_list(skel_grid_cost_normalized_gw_list):
                skel_grid_cost_normalized_gw_list_list.append(
                    skel_grid_cost_normalized_gw_list
                )


    # Convert lists to numpy arrays
    grid_cost_gw_list_list = np.array(grid_cost_gw_list_list)
    grid_cost_normalized_gw_list_list = np.array(grid_cost_normalized_gw_list_list)
    # grid_transport_plan_gw_list_list = np.array(grid_transport_plan_gw_list_list)
    skel_grid_cost_gw_list_list = np.array(skel_grid_cost_gw_list_list)
    skel_grid_cost_normalized_gw_list_list = np.array(
        skel_grid_cost_normalized_gw_list_list
    )
    # skel_grid_transport_plan_gw_list_list = np.array(skel_grid_transport_plan_gw_list_list)

    # Initialize the arrays to process
    arrays_to_process = [
        grid_cost_gw_list_list,
        grid_cost_normalized_gw_list_list,
        skel_grid_cost_gw_list_list,
        skel_grid_cost_normalized_gw_list_list,
    ]

    # Corresponding names for the arrays
    array_names = [
        "grid_cost_gw",
        "grid_cost_normalized_gw",
        "skel_grid_cost_gw",
        "skel_grid_cost_normalized_gw",
    ]

    # Initialize lists to store the results
    all_group_means = []
    all_group_stds = []
    all_group_sems = []

    # Process each array
    for array in arrays_to_process:
        # Calculate the mean and standard deviation for each group
        group_means = np.mean(array, axis=0)
        group_stds = np.std(array, axis=0)
        group_sems = np.std(array, axis=0, ddof=1) / np.sqrt(np.shape(array)[0])

        # Store the results
        all_group_means.append(group_means)
        all_group_stds.append(group_stds)
        all_group_sems.append(group_sems)
 

    for idx, array_name in enumerate(array_names):
        np.save(
            f"data/robustness/scale/{array_name}_mean.npy",
            all_group_means[idx],
        )
        np.save(
            f"data/robustness/scale/{array_name}_std.npy",
            all_group_stds[idx],
        )
        np.save(
            f"data/robustness/scale/{array_name}_sem.npy",
            all_group_sems[idx],
        )


def process_rotation(ratio_index, index, dim):
    max_rotation = 4
    srmb = SRMB(10, 10)
    env = MazeEnvironment(10, 10, auto_reset=False)

    grid_cost_gw_list = np.zeros(max_rotation)
    skel_grid_cost_gw_list = np.zeros(max_rotation)

    env.set_index("{}_{}".format(ratio_index, index))
    env.load_map("data/env/train/")
    srmb.update_map_structure(env.blocks)

    (
        vertex_nodes, deadend_nodes, edge_nodes, G, image_original, skeleton, 
        vertex_corresp, deadend_corresp, edge_corresp, edge_dict, original_G, closest_skeleton_indices
    ) = skeletonize_env(env)
    if G is None:
        return 0

    eigenmap = srmb.return_successormap()

    skel_eigenmap = avg_nodewise(eigenmap, vertex_nodes, deadend_nodes)
    sr_avg_nodewise = avg_nodewise(
        avg_nodewise(srmb.sr, vertex_nodes, deadend_nodes).T,
        vertex_nodes,
        deadend_nodes,
    ).T

    for rotation_factor in range(max_rotation):
        if rotation_factor == 0:
            continue

        else:
            env_rotated = copy.deepcopy(env)
            vertex_nodes_rotated = []
            for vertex_node_list in vertex_nodes:
                new_vertex_node_list = []
                for v in vertex_node_list:
                    vr, vc = env.state_to_index(v)
                    for _ in range(rotation_factor):
                        nvr = env.num_row - 1 - vc
                        nvc = vr
                        vr, vc = nvr, nvc
                    new_vertex_node_list.append(env_rotated.index_to_state(vr, vc))
                vertex_nodes_rotated.append(new_vertex_node_list)

            deadend_nodes_rotated = []
            for deadend_node_list in deadend_nodes:
                new_deadend_node_list = []
                for d in deadend_node_list:
                    dr, dc = env.state_to_index(d)
                    for _ in range(rotation_factor):
                        ndr = env.num_row - 1 - dc
                        ndc = dr
                        dr, dc = ndr, ndc
                    new_deadend_node_list.append(env_rotated.index_to_state(dr, dc))
                deadend_nodes_rotated.append(new_deadend_node_list)

            vertex_corresp_rotated = []
            for vcr, vcc in vertex_corresp:
                for _ in range(rotation_factor):
                    nvr = env.num_row - 1 - vcc
                    nvc = vcr
                    vcr, vcc = nvr, nvc
                vertex_corresp_rotated.append((vcr, vcc))
            vertex_corresp_rotated = tuple(vertex_corresp_rotated)

            deadend_corresp_rotated = []
            for dcr, dcc in deadend_corresp:
                for _ in range(rotation_factor):
                    ndr = env.num_row - 1 - dcc
                    ndc = dcr
                    dcr, dcc = ndr, ndc
                deadend_corresp_rotated.append((dcr, dcc))
            deadend_corresp_rotated = tuple(deadend_corresp_rotated)

            env_rotated.rotate_map(rotation_factor)

            blocks = env_rotated.return_blocks()
            srmb_rotated = SRMB(env_rotated.num_row, env_rotated.num_column)
            srmb_rotated.update_map_structure(blocks=blocks)
            eigenmap_rotated = srmb_rotated.return_successormap()

            skel_eigenmap_rotated = avg_nodewise(
                eigenmap_rotated, vertex_nodes_rotated, deadend_nodes_rotated
            )
            sr_rotated_avg_nodewise = avg_nodewise(
                avg_nodewise(
                    srmb_rotated.sr, vertex_nodes_rotated, deadend_nodes_rotated
                ).T,
                vertex_nodes_rotated,
                deadend_nodes_rotated,
            ).T



            eigenmap_normalized, eigenmap_direct_scale_factor, eigenmap_scaling_norms = center_and_scale_diffusionmap(eigenmap)
            eigenmap_rotated_normalized, eigenmap_rotated_direct_scale_factor, eigenmap_rotated_scaling_norms = center_and_scale_diffusionmap(eigenmap_rotated, provided_norms=eigenmap_scaling_norms)
            gw_cost, transport_plan, _, _, _, _ = (
                calc_gromov_wasserstein_dist_alignment(eigenmap_normalized, eigenmap_rotated_normalized)
            )
            grid_cost_gw_list[rotation_factor] = gw_cost

            skel_gw_cost, skel_transport_plan, _, _, _, _ = (
                calc_gromov_wasserstein_dist_alignment(
                    center_and_scale_diffusionmap(skel_eigenmap, direct_scale_factor=eigenmap_direct_scale_factor)[0], center_and_scale_diffusionmap(skel_eigenmap_rotated, direct_scale_factor=eigenmap_rotated_direct_scale_factor)[0]
                )
            )
            skel_grid_cost_gw_list[rotation_factor] = skel_gw_cost
            # skel_grid_transport_plan_gw_list.append(skel_transport_plan)

    np.save(
        f"data/robustness/rotation/raw_data/grid_cost_gw_{ratio_index}_{index}.npy",
        grid_cost_gw_list,
    )
    np.save(
        f"data/robustness/rotation/raw_data/skel_grid_cost_gw_{ratio_index}_{index}.npy",
        skel_grid_cost_gw_list,
    )



def test_rotation():
    dim = 10
    ratio_index_list = list(range(1, 11))
    index_list = list(range(1, 101))
    pairs = list(product(ratio_index_list, index_list))
    for pair in tqdm(pairs):
        process_rotation(pair[0], pair[1], dim)



def postprocess_rotation():
    grid_cost_gw_list_list = []
    skel_grid_cost_gw_list_list = []

    for ratio_index in tqdm(range(1, 11)):
        for index in range(1, 101):
            G = nx.MultiGraph(
                nx.read_graphml(
                    f"data/skeletonize/skeleton_graph/train/{ratio_index}_{index}.graphml"
                )
            )
            if G.number_of_nodes() == 1:
                continue


            grid_cost_gw_list = np.load(
                "data/robustness/rotation/raw_data/grid_cost_gw_{}_{}.npy".format(
                    ratio_index, index
                )
            )

            skel_grid_cost_gw_list = np.load(
                "data/robustness/rotation/raw_data/skel_grid_cost_gw_{}_{}.npy".format(
                    ratio_index, index
                )
            )

            if is_valid_list(grid_cost_gw_list):
                grid_cost_gw_list_list.append(grid_cost_gw_list)
            if is_valid_list(skel_grid_cost_gw_list):
                skel_grid_cost_gw_list_list.append(skel_grid_cost_gw_list)


    grid_cost_gw_list_list = np.array(grid_cost_gw_list_list)
    skel_grid_cost_gw_list_list = np.array(skel_grid_cost_gw_list_list)

    # Initialize the arrays to process
    arrays_to_process = [
        grid_cost_gw_list_list,
        skel_grid_cost_gw_list_list,
    ]

    # Corresponding names for the arrays
    array_names = [
        "grid_cost_gw",
        "skel_grid_cost_gw",
    ]

    # Initialize lists to store the results
    all_group_means = []
    all_group_stds = []
    all_group_sems = []

    # Process each array
    for array in arrays_to_process:
        # Calculate the mean and standard deviation for each group
        group_means = np.mean(array, axis=0)
        group_stds = np.std(array, axis=0)
        group_sems = np.std(array, axis=0, ddof=1) / np.sqrt(np.shape(array)[0])

        # Store the results
        all_group_means.append(group_means)
        all_group_stds.append(group_stds)
        all_group_sems.append(group_sems)

    
    for idx, array_name in enumerate(array_names):
        np.save(
            f"data/robustness/rotation/{array_name}_mean.npy",
            all_group_means[idx],
        )
        np.save(
            f"data/robustness/rotation/{array_name}_std.npy",
            all_group_stds[idx],
        )

        np.save(
            f"data/robustness/rotation/{array_name}_sem.npy",
            all_group_sems[idx],
        )


def generate_seed_numbers(num_seeds):
    # Set an initial seed for reproducibility
    np.random.seed(7)
    
    # Define the range for seed numbers
    max_seed_value = 1000000
    
    # Ensure num_seeds does not exceed the range
    if num_seeds > max_seed_value:
        raise ValueError("num_seeds exceeds the maximum number of unique seeds available.")
    
    # Generate a fixed set of unique seed numbers
    seed_numbers = np.random.choice(range(max_seed_value), size=num_seeds, replace=False)
    
    return seed_numbers


def process_noise(
    ratio_index, index, num_sample_per_env, dim, min_ratio=0.01, max_ratio=0.2
):
    srmb = SRMB(10, 10)
    env = MazeEnvironment(10, 10, auto_reset=False)


    grid_cost_gw_list = np.zeros(num_sample_per_env)
    skel_grid_cost_gw_list = np.zeros(num_sample_per_env)

    env.set_index("{}_{}".format(ratio_index, index))
    env.load_map("data/env/train/")
    srmb.update_map_structure(env.blocks)

    (
        vertex_nodes, deadend_nodes, edge_nodes, G, image_original, skeleton, 
        vertex_corresp, deadend_corresp, edge_corresp, edge_dict, original_G, closest_skeleton_indices
    ) = skeletonize_env(env)
    if G is None:
        return 0

    eigenmap = srmb.return_successormap()

    skel_eigenmap = avg_nodewise(eigenmap, vertex_nodes, deadend_nodes)
    sr_avg_nodewise = avg_nodewise(
        avg_nodewise(srmb.sr, vertex_nodes, deadend_nodes).T,
        vertex_nodes,
        deadend_nodes,
    ).T
    seed_numbers = generate_seed_numbers(100000)

    sample_index = 0
    total_attempt_count = -1
    attempt_count_per_sample = -1

    while sample_index < num_sample_per_env:
        total_attempt_count += 1
        attempt_count_per_sample += 1
        if total_attempt_count >= 100000:
            raise ValueError("Exceeded the maximum number of attempts.")
        env_noised = copy.deepcopy(env)
        np.random.seed(seed_numbers[total_attempt_count])
        num_choice = np.max((int(np.ceil(len(env_noised.nonblocks) * max_ratio)) - (attempt_count_per_sample // 100), 1))
        noise_blocks = np.random.choice(
            env_noised.nonblocks,
            num_choice,
            replace=False,
        )
        env_noised.update_map(blocks=np.append(env_noised.blocks, noise_blocks))
        if not env_noised.is_connected():
            continue
        env_noised.set_index("{}_{}_{}".format(ratio_index, index, sample_index + 1))
        env_noised.visualize(display=False, directory="data/robustness/noise/plot/")
        (
            vertex_nodes_noised, deadend_nodes_noised, edge_nodes_noised, G_noised, image_original_noised, skeleton_noised, 
            vertex_corresp_noised, deadend_corresp_noised, edge_corresp_noised, edge_dict_noised, original_G_noised, closest_skeleton_indices_noised
        ) = skeletonize_env(env_noised)
        if G_noised is None:
            continue
        matcher = nx.algorithms.isomorphism.GraphMatcher(G, G_noised)
        if matcher.is_isomorphic():
            perm = matcher.mapping
        else:
            continue

        blocks = env_noised.return_blocks()
        srmb_noised = SRMB(env_noised.num_row, env_noised.num_column)
        srmb_noised.update_map_structure(blocks=blocks)
        eigenmap_noised = srmb_noised.return_successormap()

        skel_eigenmap_noised = avg_nodewise(
            eigenmap_noised, vertex_nodes_noised, deadend_nodes_noised
        )
        sr_noised_avg_nodewise = avg_nodewise(
            avg_nodewise(srmb_noised.sr, vertex_nodes_noised, deadend_nodes_noised).T,
            vertex_nodes_noised,
            deadend_nodes_noised,
        ).T

        eigenmap_normalized, eigenmap_direct_scale_factor, eigenmap_scaling_norms = center_and_scale_diffusionmap(eigenmap)
        eigenmap_noised_normalized, eigenmap_noised_direct_scale_factor, eigenmap_noised_scaling_norms = center_and_scale_diffusionmap(eigenmap_noised, provided_norms=eigenmap_scaling_norms)
        gw_cost, transport_plan, _, _, _, _ = calc_gromov_wasserstein_dist_alignment(
            eigenmap_normalized, eigenmap_noised_normalized
        )
        grid_cost_gw_list[sample_index] = gw_cost
        # grid_transport_plan_gw_list.append(transport_plan)


        skel_gw_cost, skel_transport_plan, _, _, _, _ = (
            calc_gromov_wasserstein_dist_alignment(
                center_and_scale_diffusionmap(skel_eigenmap, direct_scale_factor=eigenmap_direct_scale_factor)[0], center_and_scale_diffusionmap(skel_eigenmap_noised, direct_scale_factor=eigenmap_noised_direct_scale_factor)[0]
            )
        )
        skel_grid_cost_gw_list[sample_index] = skel_gw_cost
        # skel_grid_transport_plan_gw_list.append(skel_transport_plan)

        sample_index += 1
        attempt_count_per_sample = -1

    np.save(
        f"data/robustness/noise/raw_data/grid_cost_gw_{ratio_index}_{index}.npy",
        grid_cost_gw_list,
    )
    np.save(
        f"data/robustness/noise/raw_data/skel_grid_cost_gw_{ratio_index}_{index}.npy",
        skel_grid_cost_gw_list,
    )


def test_noise(num_sample_per_env=5):
    dim = 10
    ratio_index_list = list(range(1, 11))
    index_list = list(range(1, 101))
    pairs = list(product(ratio_index_list, index_list))
    for pair in tqdm(pairs):
        process_noise(pair[0], pair[1], num_sample_per_env, dim)


def postprocess_noise():
    grid_cost_gw_list_list = []
    skel_grid_cost_gw_list_list = []

    for ratio_index in range(1, 11):
        for index in range(1, 101):
            overall_index = (ratio_index - 1) * 100 + index
            G = nx.MultiGraph(
                nx.read_graphml(
                    f"data/skeletonize/skeleton_graph/train/{ratio_index}_{index}.graphml"
                )
            )
            if G.number_of_nodes() == 1:
                continue
            grid_cost_gw_list = np.load(
                "data/robustness/noise/raw_data/grid_cost_gw_{}_{}.npy".format(
                    ratio_index, index
                )
            )
            skel_grid_cost_gw_list = np.load(
                "data/robustness/noise/raw_data/skel_grid_cost_gw_{}_{}.npy".format(
                    ratio_index, index
                )
            )

            if is_valid_list(grid_cost_gw_list):
                grid_cost_gw_list_list.append(grid_cost_gw_list)
            if is_valid_list(skel_grid_cost_gw_list):
                skel_grid_cost_gw_list_list.append(skel_grid_cost_gw_list)

    grid_cost_gw_list_list = np.array(grid_cost_gw_list_list)
    skel_grid_cost_gw_list_list = np.array(skel_grid_cost_gw_list_list)

    # Initialize the arrays to process
    arrays_to_process = [
        grid_cost_gw_list_list,
        skel_grid_cost_gw_list_list,
    ]

    # Corresponding names for the arrays
    array_names = [
        "grid_cost_gw",
        "skel_grid_cost_gw",
    ]

    # Initialize lists to store the results
    all_group_means = []
    all_group_stds = []
    all_group_sems = []

    # Process each array
    for array in arrays_to_process:
        # Calculate the mean and standard deviation for each group
        group_means = np.mean(array)
        group_stds = np.std(array)
        group_sems = np.std(array, ddof=1) / np.sqrt(len(array.flatten()))

        # Store the results
        all_group_means.append(group_means)
        all_group_stds.append(group_stds)
        all_group_sems.append(group_sems)
    
    
    for idx, array_name in enumerate(array_names):
        np.save(
            f"data/robustness/noise/{array_name}_mean.npy",
            all_group_means[idx],
        )
        np.save(
            f"data/robustness/noise/{array_name}_std.npy",
            all_group_stds[idx],
        )
        np.save(
            f"data/robustness/noise/{array_name}_sem.npy",
            all_group_sems[idx],
        )


def postprocess_all_skel_gw():
    invariance_types = ['shift', 'scale', 'rotation', 'noise']
    base_path = 'data/robustness/{}/raw_data/'
    
    all_skel_gw_data = []
    
    for inv_type in invariance_types:
        type_path = base_path.format(inv_type)
        if not os.path.exists(type_path):
            raise ValueError('The path does not exist.')
        
        for ratio_index in range(1, 11):
            for index in range(1, 101):
                if inv_type == 'scale':
                    file_name = f'skel_grid_cost_normalized_gw_{ratio_index}_{index}.npy'
                else:
                    file_name = f'skel_grid_cost_gw_{ratio_index}_{index}.npy'
                
                file_path = os.path.join(type_path, file_name)
                if os.path.exists(file_path):
                    data = np.load(file_path).flatten()
                    if is_valid_list(data):
                        all_skel_gw_data.extend(data)
    
    all_skel_gw_data = np.array(all_skel_gw_data)
    mean = np.mean(all_skel_gw_data)
    std = np.std(all_skel_gw_data)
    
    return mean, std


def get_sorted_indices(matrix):
    # Get the upper triangular part of the matrix, excluding the diagonal
    upper_tri_indices = np.triu_indices_from(matrix, k=1)

    # Extract the values from the upper triangular part
    upper_tri_values = matrix[upper_tri_indices]

    # Sort the values in descending order and get the sorted indices
    sorted_indices = np.argsort(-upper_tri_values)

    # Map the sorted indices back to the original row and column indices
    sorted_row_indices = upper_tri_indices[0][sorted_indices]
    sorted_col_indices = upper_tri_indices[1][sorted_indices]

    # Combine row and column indices into pairs
    sorted_pairs = list(zip(sorted_row_indices, sorted_col_indices))

    return sorted_pairs


def postprocess_all_skel_gw_nonisomorphic():
    env_dist_mat = np.load('data/isomorphism_distinguish/gh_dist_graph_mat.npy')
    env_dist_mat = (env_dist_mat + env_dist_mat.T) / 2
    skel_grid_dist_mat = np.load("data/isomorphism_distinguish/skel_gw_dist_grid_mat.npy")
    skel_grid_dist_mat = (skel_grid_dist_mat + skel_grid_dist_mat.T) / 2
    grid_dist_mat = np.load("data/isomorphism_distinguish/gw_dist_grid_mat.npy")
    grid_dist_mat = (grid_dist_mat + grid_dist_mat.T) / 2
    for total_index in range(1000):
        ratio_index_1 = total_index // 100 + 1
        index_1 = total_index % 100 + 1
        G = nx.MultiGraph(
            nx.read_graphml(
                f"data/skeletonize/skeleton_graph/train/{ratio_index_1}_{index_1}.graphml"
            )
        )
        if G.number_of_nodes() == 1:
            grid_dist_mat[total_index] = np.nan
            grid_dist_mat[:, total_index] = np.nan
            env_dist_mat[total_index] = np.nan
            env_dist_mat[:, total_index] = np.nan
            skel_grid_dist_mat[total_index] = np.nan
            skel_grid_dist_mat[:, total_index] = np.nan
            
    gh_sorted_pairs = get_sorted_indices(env_dist_mat)
    _, non_isomorphic_pairs = (
        identify_env_pairs_basis10_train_wo_subgraph(
            gh_sorted_pairs, env_dist_mat, grid_dist_mat, skel_grid_dist_mat
        )
    )
    
    skel_grid_non_isomorphic = [
        skel_grid_dist_mat[pair[0], pair[1]] for pair in non_isomorphic_pairs
    ]
    env_non_isomorphic = [env_dist_mat[pair[0], pair[1]] for pair in non_isomorphic_pairs]
    skel_non_isomorphic_gw_values = [
        gw
        for gh, gw in zip(env_non_isomorphic, skel_grid_non_isomorphic)
        if gh >= 0 and gw >= 0
    ]
    skel_non_isomorphic_gw_mean = np.mean(skel_non_isomorphic_gw_values)
    skel_non_isomorphic_gw_std = np.std(skel_non_isomorphic_gw_values)
    return skel_non_isomorphic_gw_mean, skel_non_isomorphic_gw_std



def check_data_availability():
    import os
    
    base_path = 'data/robustness'
    
    if not os.path.exists(base_path):
        return False
    
    # Check each invariance type separately
    invariance_types = ['shift', 'scale', 'rotation', 'noise']
    
    for inv_type in invariance_types:
        type_path = os.path.join(base_path, inv_type)
        if not os.path.exists(type_path):
            return False
        
        # Check if processed data files exist for this type
        required_files = []
        if inv_type == 'scale':
            required_files = [
                'skel_grid_cost_normalized_gw_mean.npy',
                'skel_grid_cost_normalized_gw_std.npy',
            ]
        else:
            required_files = [
                'skel_grid_cost_gw_mean.npy',
                'skel_grid_cost_gw_std.npy',
            ]
        
        for filename in required_files:
            file_path = os.path.join(type_path, filename)
            if not os.path.exists(file_path):
                return False
    
    return True


def main():
    if not check_data_availability():
        test_shift()
        test_scale()
        test_rotation()
        test_noise()
        postprocess_shift()
        postprocess_scale()
        postprocess_rotation()
        postprocess_noise()
