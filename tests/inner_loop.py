from modules.model import TAGUpdate
from modules.base import SRDyna, SR, SRTD, SRMB
from modules.comparison import TAGUpdateBiased
from modules.env import MazeEnvironment
import numpy as np
from utils.ot import calc_gromov_wasserstein_dist_alignment
from utils.topology import avg_nodewise
from scipy.stats import spearmanr
from tqdm import tqdm
import os
from utils.skeletonize import skeletonize_env
import multiprocessing as mp
import parmap


def random_walk_policy(rd_prob):
    return np.random.choice(
        4, p=[(1 - rd_prob) / 2, (1 - rd_prob) / 2, rd_prob / 2, rd_prob / 2]
    )


def generate_trajectories(ratio_index, index):
    env = MazeEnvironment(10, 10, auto_reset=False)
    env.set_index("{}_{}".format(ratio_index, index))
    env.load_map("data/env/train/")
    for bias_index in range(5):
        trajectories = []
        index_str = "{}_{}_{}".format(ratio_index, index, bias_index)
        if not os.path.isfile(
            "data/inner_loop/trajectories/{}.npy".format(index_str)
        ):
            rd_prob = 0.5 + 0.1 * bias_index
            assert rd_prob <= 1
            for _ in tqdm(range(1, 101)):
                trajectory = []
                state = env.reset()
                for timestep in range(1, 1002):
                    action = random_walk_policy(rd_prob)
                    next_state, _, _, _, _ = env.step(action)
                    assert next_state in [
                        state,
                        state - 1,
                        state + 1,
                        state - 10,
                        state + 10,
                    ]
                    trajectory.append([state, action, next_state])
                    state = next_state
                trajectories.append(trajectory)
            np.save(
                "data/inner_loop/trajectories/{}.npy".format(index_str),
                trajectories,
            )


def init_agent(agent_index, state):
    if agent_index == 0:  # TAG
        agent = TAGUpdate(10, 10, 0.995)
        name = "tag"
    elif agent_index == 1:  # TAGUpdateBiased
        agent = TAGUpdateBiased(10, 10)
        name = "tag-biased"
    elif agent_index == 2:  # MB
        agent = SRMB(10, 10, 0.995)
        name = "SR-MB"
    elif agent_index == 3:  # Dyna
        agent = SRDyna(10, 10, 0.995)
        name = "SR-Dyna"
    elif agent_index == 4:  # TD
        agent = SRTD(10, 10, 0.995)
        name = "SR-TD"
    else:
        raise ValueError()
    return agent, name


def update_agent(
    agent,
    tag_skel,
    agent_index,
    state,
    action,
    next_state,
    next_action,
    available_actions_per_state,
):
    if state != next_state:
        available_actions_per_state[state][action] = True
    inv_action = [1, 0, 3, 2]
    tag_skel.update_map(state, action, next_state, True)
    if agent_index == 0:  # TAG
        agent.update_map(state, action, next_state, True)
        return agent.sr
    elif agent_index == 1:  # TAGUpdateBiased
        agent.update_map(state, action, next_state)
        return agent.return_map()
    elif agent_index == 2:  # MB
        # if state != next_state:
        agent.update_map(state, action, next_state)
        return agent.sr
    elif agent_index == 3:  # Dyna
        # if state != next_state:
        agent.update(state, action, next_state, next_action, 0)
        return agent.return_map()
    elif agent_index == 4:  # TD
        # if state != next_state:
        agent.update_map(state, next_state, True)
        return agent.sr
    else:
        raise ValueError()


def calculate_metrics(sr, gt, skel_diffusionmap, tag_skel):
    spearmanr_corrs = []
    for p_index in range(100):
        spearmanr_corr, _ = spearmanr(gt.sr[:, p_index], sr[:, p_index].astype(float))

        spearmanr_corrs.append(spearmanr_corr)
    tag_skel_sr = tag_skel.sr.copy()
    tag_skel.sr = sr
    inferred_skel_diffusionmap = tag_skel.return_skeleton_diffusionmap()
    tag_skel.sr = tag_skel_sr
    skel_gw_dist = calc_gromov_wasserstein_dist_alignment(
        skel_diffusionmap, inferred_skel_diffusionmap
    )[0]

    return spearmanr_corrs, skel_gw_dist


def train_inner_loop(ratio_index, index, agent_index, difficulty):
    index_str = "{}_{}_{}".format(ratio_index, index, difficulty)
    if os.path.isfile(
        "data/inner_loop/raw_data/{}_{}_{}_{}.npy".format(
            ratio_index, index, difficulty, agent_index
        )
    ):
        pass

    else:
        trajectory = np.load("data/inner_loop/trajectories/{}.npy".format(index_str))
        env = MazeEnvironment(10, 10, auto_reset=False)
        env.set_index("{}_{}".format(ratio_index, index))
        env.load_map("data/env/train/")
        (
            vertex_nodes,
            deadend_nodes,
            edge_nodes,
            G,
            image,
            skeleton,
            vertex_corresp,
            deadend_corresp,
            edge_corresp,
        ) = skeletonize_env(env)
        gt = SR(10, 10)  # ground truth
        gt.set_index("{}_{}".format(ratio_index, index))
        gt.load_map("data/sr/train/")
        tag_skel = TAGUpdate(10, 10, 0.995)  # skel diffusionmap 구하기 위한 도구
        diffusionmap = gt.return_successormap()
        skel_diffusionmap = avg_nodewise(diffusionmap, vertex_nodes, deadend_nodes)
        state = 0
        agent, name = init_agent(agent_index, state)
        spearmanr_corrs_list = []
        skel_gw_dist_list = []
        available_actions_per_state = np.zeros((100, 4), dtype=bool)
        sr_list = []
        cum_timestep = 0
        for trj_idx in tqdm(range(20)):
            for timestep in tqdm(range(0, 1000)):
                sr = update_agent(
                    agent,
                    tag_skel,
                    agent_index,
                    trajectory[trj_idx][timestep][0],
                    trajectory[trj_idx][timestep][1],
                    trajectory[trj_idx][timestep][2],
                    trajectory[trj_idx][timestep + 1][1],
                    available_actions_per_state,
                )
                if agent_index == 3 and (timestep + 1) % 1000 == 0:
                    agent.offline_update(10000)
                if (timestep + 1) % 10 == 0:
                    spearmanr_corrs, skel_gw_dist = calculate_metrics(
                        sr, gt, skel_diffusionmap, tag_skel
                    )
                    spearmanr_corrs_list.append(np.mean(spearmanr_corrs))
                    skel_gw_dist_list.append(skel_gw_dist)
                cum_timestep += 1
            sr_list.append(sr)
        all_metrics_list = [spearmanr_corrs_list, skel_gw_dist_list]
        agent.set_index("{}_{}_{}_{}".format(ratio_index, index, difficulty, cum_timestep))
        agent.save_map("data/inner_loop/raw_data/agent_save/")
        np.save(
            "data/inner_loop/raw_data/{}_{}".format(index_str, agent_index),
            np.array(all_metrics_list),
        )

def train_inner_loop_mp():
    num_cores = mp.cpu_count()
    
    # Create all parameter combinations
    all_params = []
    for agent_index in range(5):
        for difficulty in range(5):
            for ratio_index in range(1, 11):
                for index in range(1, 101):
                    all_params.append((int(ratio_index), int(index), int(agent_index), int(difficulty)))
    
    # Split parameters for multiprocessing
    splitted_params = np.array_split(all_params, num_cores)
    
    # Define wrapper function for parmap
    def simulate_exploration_wrapper(params_list):
        results = []
        for ratio_index, index, agent_index, difficulty in params_list:
            result = train_inner_loop(ratio_index, index, agent_index, difficulty)
            results.append(result)
        return results
    
    # Execute with multiprocessing
    results = parmap.map(
        simulate_exploration_wrapper,
        splitted_params,
        pm_pbar=True,
        pm_processes=num_cores,
    )


def postprocess_inner_loop(agent_index):
    model_index = agent_index
    all_list = [[] for _ in range(2)]
    all_list_bias_whole = [[[] for _ in range(2)] for _ in range(5)]
    not_found_list = []
    
    for ratio_index in tqdm(range(1, 11)):
        all_list_ratio = [[] for _ in range(2)]
        for bias_index in range(5):
            all_list_ratio_bias = [[] for _ in range(2)]
            all_list_bias = all_list_bias_whole[bias_index]
            
            for index in range(1, 101):
                try:
                    result = np.load(
                        "data/inner_loop/raw_data/{}_{}_{}_{}.npy".format(
                            ratio_index, index, bias_index, model_index
                        )
                    )
                    if len(result[0]) == 20000:
                        result = np.array([[result[j][i] for i in range(20000) if (i + 1) % 10 == 0] for j in range(2)])
                except FileNotFoundError:
                    not_found_list.append([ratio_index, index, bias_index])
                    continue

                for i in range(2):
                    all_list_ratio[i].append(result[i])
                    all_list_ratio_bias[i].append(result[i])
                    all_list_bias[i].append(result[i])
                    all_list[i].append(result[i])

    for i in range(2):
        np.save(
            "data/inner_loop/processed_data/{}_{}_mean.npy".format(
                i, model_index
            ),
            np.mean(all_list[i], axis=0),
        )
        np.save(
            "data/inner_loop/processed_data/{}_{}_std.npy".format(
                i, model_index
            ),
            np.std(all_list[i], axis=0),
        )
        
    for bias_index in range(5):
        for i in range(2):
            np.save(
                "data/inner_loop/processed_data/{}_{}_{}_bias_mean.npy".format(
                    i, model_index, bias_index
                ),
                np.mean(all_list_bias_whole[bias_index][i], axis=0),
            )
            np.save(
                "data/inner_loop/processed_data/{}_{}_{}_bias_std.npy".format(
                    i, model_index, bias_index
                ),
                np.std(all_list_bias_whole[bias_index][i], axis=0),
            )
    
    print(f'{len(not_found_list)}/5000 not found')
    print(f"not found list for {model_index}: ", not_found_list)


def check_inner_loop_data_availability():
    base_dir = "data/inner_loop/processed_data/"
    
    if not os.path.exists(base_dir):
        return False
    
    # Check if all processed data files exist
    for i in range(2):  # 0, 1
        for model_index in range(5):  # 0, 1, 2, 3, 4
            mean_file = f"{i}_{model_index}_mean.npy"
            std_file = f"{i}_{model_index}_std.npy"
            
            if not os.path.exists(os.path.join(base_dir, mean_file)):
                return False
            if not os.path.exists(os.path.join(base_dir, std_file)):
                return False
            
            # Check bias-specific files
            for bias_index in range(5):
                bias_mean_file = f"{i}_{model_index}_{bias_index}_bias_mean.npy"
                bias_std_file = f"{i}_{model_index}_{bias_index}_bias_std.npy"
                
                if not os.path.exists(os.path.join(base_dir, bias_mean_file)):
                    return False
                if not os.path.exists(os.path.join(base_dir, bias_std_file)):
                    return False
    
    return True


def main():
    if not check_inner_loop_data_availability():
        generate_trajectories()
        train_inner_loop_mp()
        postprocess_inner_loop()


if __name__ == "__main__":
    main()