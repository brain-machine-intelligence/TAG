import numpy as np
from modules.env import MazeEnvironment
from modules.base import SRMB
from modules.comparison import ComposDR
from modules.model import TAGZeroShot
from utils.skeletonize import skeletonize_env
from typing import List, Tuple
from tqdm import tqdm


def test_navigation(env, agent, num_trials=1000):
    results = []
    for _ in tqdm(range(num_trials)):
        agent.new_task(np.array([s == env.goal for s in range(env.num_states)]))
        state = env.reset()
        done = False
        step_count = 0
        while not done:
            action = agent.sample_action(state)
            next_state, reward, done, step_count = env.step(action)
            state = next_state
        results.append(step_count)
    return np.array(results)


def set_envs_blocks():
    start_small, goal_small = 0, 77
    blocks_1_small = [12, 21, 22, 23, 32]
    blocks_2_small = [26, 27, 28]
    blocks_3_small = [61, 62, 63, 71, 72, 73, 81, 82, 83]
    blocks_4_small = [66, 67, 68, 76, 86]
    env_list = []
    start_list = []
    goal_list = []
    blocks_1_list = []
    blocks_2_list = []
    blocks_3_list = []
    blocks_4_list = []
    blocks_list = []

    for i in range(6):
        env = MazeEnvironment(20, 20, auto_reset=False)
        env.set_index(f"narrowness_{i}")
        blocks_1 = []
        blocks_2 = []
        blocks_3 = []
        blocks_4 = []
        for b in blocks_1_small:
            r, c = env.state_to_index(b, 10)
            blocks_1.append(env.index_to_state(r + 5 - i, c + 5 - i, 20))
        for b in blocks_2_small:
            r, c = env.state_to_index(b, 10)
            blocks_2.append(env.index_to_state(r + 5 - i, c + 5 + i, 20))
        for b in blocks_3_small:
            r, c = env.state_to_index(b, 10)
            blocks_3.append(env.index_to_state(r + 5 + i, c + 5 - i, 20))
        for b in blocks_4_small:
            r, c = env.state_to_index(b, 10)
            blocks_4.append(env.index_to_state(r + 5 + i, c + 5 + i, 20))
        blocks = []
        blocks.extend(blocks_1)
        blocks.extend(blocks_2)
        blocks.extend(blocks_3)
        blocks.extend(blocks_4)
        start = 0
        goal_nr, goal_nc = env.state_to_index(goal_small, 10)
        goal = env.index_to_state(goal_nr + 5 + i, goal_nc + 5 + i, 20)

        env.insert_block(blocks)
        env.set_start_goal_states(start, goal)

        env_list.append(env)
        start_list.append(start)
        goal_list.append(goal)
        blocks_1_list.append(blocks_1)
        blocks_2_list.append(blocks_2)
        blocks_3_list.append(blocks_3)
        blocks_4_list.append(blocks_4)
        blocks_list.append(blocks)

    return env_list, start_list, goal_list, blocks_1_list, blocks_2_list, blocks_3_list, blocks_4_list, blocks_list


def find_walls(
    transition_matrix: np.ndarray, num_rows: int, num_cols: int, threshold: float = 1e-9
) -> List[Tuple[int, int]]:
    num_states = transition_matrix.shape[0]
    if num_states != num_rows * num_cols:
        raise ValueError("Matrix size does not match num_rows * num_cols.")

    walls: List[Tuple[int, int]] = []
    for r in range(num_rows):
        for c in range(num_cols):
            i = r * num_cols + c

            if c < num_cols - 1:
                j = i + 1
                if (
                    transition_matrix[i, j] < threshold
                    and transition_matrix[j, i] < threshold
                ):
                    walls.append((i, j))  # i < j

            if r < num_rows - 1:
                j = i + num_cols
                if (
                    transition_matrix[i, j] < threshold
                    and transition_matrix[j, i] < threshold
                ):
                    walls.append((i, j))  # i < j
    return walls


def evaluate_params(args):
    (
        kernel,
        g_param,
        alpha1,
        alpha3,
        original_G,
        closest_skeleton_indices,
        env,
        num_trials
    ) = args

    tag = TAGZeroShot(env.num_row, env.num_column, kernel, g_param, alpha1, alpha3, stochastic_action=True)
    tag.update_map(original_G, closest_skeleton_indices, env.blocks, env.walls)

    timesteps_tag = test_navigation(env, tag, num_trials)
    

    return {
        "kernel": kernel,
        "g_param": g_param,
        "alpha_s2n": alpha1,
        "alpha_n2s": alpha3,
        "mean_length": timesteps_tag.mean(),
        "timesteps_tag": timesteps_tag,
        "placefield_tag": tag.placefield,
    }


def run(num_trials=1000,eval_metric='mean_length'):
    env_type=0
    graph_kernels = ['rbf', 'heat']

    graph_param_grid = {
        'rbf':  np.concatenate(([0.0], np.logspace(-2, 1, num=4))),   # [0., 0.01, 0.1, 1., 10.]
        'heat': np.concatenate(([0.0], np.logspace(-1, 2, num=4))),   # [0., 0.1, 1., 10., 100.]
    }


    alpha_state_to_node_grid = np.concatenate(
        ([0.0], np.logspace(-2, 1, num=4))
    )

    alpha_node_to_state_grid = np.concatenate(
        ([0.0], np.logspace(-2, 1, num=4))
    )
    (
        env_list,
        start_list,
        goal_list,
        blocks_1_list,
        blocks_2_list,
        blocks_3_list,
        blocks_4_list,
        blocks_list,
    ) = set_envs_blocks()

    lambda_per_trial = [2, 2, 1, 3, 3, 3]
    for trial in range(len(env_list)):
        env = env_list[trial]
        start = start_list[trial]
        goal = goal_list[trial]
        blocks_1 = blocks_1_list[trial]
        blocks_2 = blocks_2_list[trial]
        blocks_3 = blocks_3_list[trial]
        blocks_4 = blocks_4_list[trial]
        blocks = blocks_list[trial]


        optimal = SRMB(env.num_row, env.num_column, 0.1, stochastic_action=False)
        optimal.update_map_structure(blocks)

        dr = ComposDR(env.num_row, env.num_column, lambda_per_trial[trial], [goal], stochastic_action=True)
        dr.update_map_por([blocks_1, blocks_2, blocks_3, blocks_4])

        (
            vertex_nodes,
            deadend_nodes,
            edge_nodes,
            simplified_G,
            image,
            skeleton,
            vertex_corresp,
            deadend_corresp,
            edge_corresp,
            edge_dict,
            original_G,
            closest_skeleton_indices,
        ) = skeletonize_env(env, with_mapping=True)

        tag = TAGZeroShot(env.num_row, env.num_column, 'heat', 0.1, 10.0, 10.0, stochastic_action=True)
        tag.update_map(original_G, closest_skeleton_indices, env.blocks, env.walls)

        timesteps_optimal = test_navigation(env, optimal, 1)
        timesteps_dr = test_navigation(env, dr, num_trials)
        timesteps_tag = test_navigation(env, tag, num_trials)

        fname_suffix = f"{trial}_{env_type}_{eval_metric}"
        np.save(f'data/policy_decoding/timesteps_dr_{fname_suffix}.npy', timesteps_dr)
        np.save(f'data/policy_decoding/timesteps_tag_{fname_suffix}.npy', timesteps_tag)
        np.save(f'data/policy_decoding/timesteps_optimal_{fname_suffix}.npy', timesteps_optimal)


def check_data_availability():
    import os
    
    base_dir = 'data/policy_decoding'
    
    if not os.path.exists(base_dir):
        return False
    
    for trial in range(6):
        env_type = 0
        eval_metric = 'mean_length'
        
        fname_suffix = f"{trial}_{env_type}_{eval_metric}"
        
        required_files = [
            f'timesteps_dr_{fname_suffix}.npy',
            f'timesteps_tag_{fname_suffix}.npy',
            f'timesteps_optimal_{fname_suffix}.npy'
        ]
        
        for filename in required_files:
            file_path = os.path.join(base_dir, filename)
            if not os.path.exists(file_path):
                return False
    
    return True


def main():
    if not check_data_availability():
        run(1000, 'mean_length')

if __name__ == "__main__":
    main()