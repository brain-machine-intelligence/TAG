from modules.model import TAGTransfer, TAGNetwork
from modules.comparison import TAGTransferUCB, TAGTransferEGreedy
from modules.env import MazeEnvironment
from tqdm import tqdm
from utils.ot import calc_gromov_wasserstein_dist_alignment
import networkx as nx
import numpy as np
import traceback
from networkx.algorithms import isomorphism
from sklearn.metrics import auc
from scipy.stats import spearmanr


def simulate_exploration(ratio_index, index, expl_strategy_index, debug=False):
    corr_list = []
    gw_list = []

    if expl_strategy_index == 0:  # TAG with topological exploration
        tag = TAGTransfer(10, 10, 0.995)
        topology_graph = nx.MultiGraph(
            nx.read_graphml(
                f"data/skeletonize/skeleton_graph/train/{ratio_index}_{index}.graphml"
            )
        )
        tag.set_topological_prior(topology_graph)
        map_saved = False
    elif expl_strategy_index == 1:  # TAG with UCB
        tag = TAGTransfer(10, 10, 0.995)
    elif expl_strategy_index == 4:  # TAG with random exploration
        tag = TAGTransferEGreedy(10, 10, 0.995, 1)
    else:
        raise ValueError("Invalid exploration strategy index")
    tag.set_index(f"{ratio_index}_{index}")

    tag_target = TAGNetwork(10, 10, 0.995)
    tag_target.set_index(f"{ratio_index}_{index}")
    tag_target.load_map("data/sr/train/")
    target_diffusionmap = tag_target.return_skeleton_diffusionmap()
    if target_diffusionmap.shape[0] < 3:
        return [-1 for _ in range(2000)]
    env = MazeEnvironment(10, 10, auto_reset=False)
    env.set_index(f"{ratio_index}_{index}")
    env.load_map("data/env/train/")
    state = env.reset()

    if expl_strategy_index == 0:
        isomorphic_count = 0
        first_isomorphic_sr = None
        first_isomorphic_timestep = -1
        first_experienced_transitions = None
    else:
        try:
            first_isomorphic_timestep = np.load(
                f"data/exploration/raw_data/0_{ratio_index}_{index}_timestep.npy"
            )[0]
        except:
            first_isomorphic_timestep = -1

    for timestep in tqdm(range(0, 1000)):
        action = tag.sample_action_exploration(state)
        next_state, _, _, _ = env.step(action)
        tag.update_map(state, action, next_state, bidirection=True)
        state = next_state
        inferred_diffusionmap = tag.return_skeleton_diffusionmap()
        if expl_strategy_index == 0:
            if not map_saved:
                if tag.skeleton_graph.number_of_nodes() >= 3:
                    GM = isomorphism.MultiGraphMatcher(
                        tag.topological_prior, tag.skeleton_graph
                    )
                    if GM.is_isomorphic():
                        isomorphic_count += 1
                        if isomorphic_count == 1:
                            first_isomorphic_sr = tag.sr
                            first_isomorphic_timestep = timestep
                            first_experienced_transitions = (
                                tag.experienced_transitions + tag.env_border_transitions
                            )
                    else:
                        isomorphic_count = 0
                        first_isomorphic_sr = None
                        first_isomorphic_timestep = -1
                        first_experienced_transitions = None
                    if isomorphic_count == 50:
                        assert first_isomorphic_sr is not None
                        np.save(
                            f"data/exploration/raw_data/0_{ratio_index}_{index}_sr.npy",
                            first_isomorphic_sr,
                        )
                        np.save(
                            f"data/exploration/raw_data/0_{ratio_index}_{index}_timestep.npy",
                            [first_isomorphic_timestep],
                        )
                        np.save(
                            f"data/exploration/raw_data/0_{ratio_index}_{index}_experienced_transitions.npy",
                            first_experienced_transitions,
                        )
        else:
            if timestep == first_isomorphic_timestep:
                np.save(
                    f"data/exploration/raw_data/{expl_strategy_index}_{ratio_index}_{index}_sr.npy",
                    tag.sr,
                )
                np.save(
                    f"data/exploration/raw_data/{expl_strategy_index}_{ratio_index}_{index}_experienced_transitions.npy",
                    tag.experienced_transitions + tag.env_border_transitions,
                )

        if inferred_diffusionmap.shape[0] == 0:
            gw_list.append(-1)
        else:
            gw_list.append(
                calc_gromov_wasserstein_dist_alignment(
                    target_diffusionmap, inferred_diffusionmap
                )[0]
            )
        spearmanr_corrs = []
        for p_index in range(100):
            spearmanr_corr, _ = spearmanr(tag_target.sr[:, p_index], tag.sr[:, p_index].astype(float))

            spearmanr_corrs.append(spearmanr_corr)
        corr_list.append(np.mean(spearmanr_corrs))
        if debug:
            if (timestep + 1) % 10 == 0:
                tag.set_index(f"{ratio_index}_{index}_{expl_strategy_index}_{timestep}")
                tag.visualize_map()
                env.set_index(f"{ratio_index}_{index}_{expl_strategy_index}_{timestep}")
                env.visualize(
                    trajectory=True, directory="./display/", no_startgoal=True
                )
    return gw_list, corr_list


def process_exploration(start_index, end_index, expl_strategy_index):
    for overall_index in tqdm(range(start_index, end_index)):
        ratio_index = (overall_index - 1) // 100 + 1
        index = (overall_index - 1) % 100 + 1
        try:
            gw_list, corr_list = simulate_exploration(ratio_index, index, expl_strategy_index)
            np.save(
                f"data/exploration/raw_data/{expl_strategy_index}_{ratio_index}_{index}.npy",
                gw_list,
            )
            np.save(
                f"data/exploration/raw_data/{expl_strategy_index}_{ratio_index}_{index}_corr.npy",
                corr_list,
            )
        except Exception as e:
            np.save(
                f"data/exploration/raw_data/failed_{expl_strategy_index}_{ratio_index}_{index}.npy",
                [],
            )
            error_details = traceback.format_exc()
            with open(
                f"data/exploration/raw_data/failed_{expl_strategy_index}_{ratio_index}_{index}.txt",
                "w",
            ) as error_file:
                error_file.write(error_details)


def postprocess_exploration(expl_strategy_index):
    gw_lists = []
    for ratio_index in tqdm(range(1, 11)):
        for index in range(1, 101):
            gw_list = np.load(
                f"data/exploration/raw_data/{expl_strategy_index}_{ratio_index}_{index}.npy"
            )[:1000]
            if gw_list[0] > 0:
                gw_lists.append(gw_list)
    gw_lists = np.array(gw_lists)
    auc_values = [auc(range(len(gw_list)), gw_list) for gw_list in gw_lists]

    np.save(f"data/exploration/{expl_strategy_index}_auc.npy", auc_values)

    mean_auc = np.mean(auc_values)
    std_auc = np.std(auc_values)
    np.save(
        f"data/exploration/{expl_strategy_index}_auc_mean.npy",
        mean_auc,
    )
    np.save(
        f"data/exploration/{expl_strategy_index}_auc_std.npy",
        std_auc,
    )

    means = []
    stds = []

    for i in range(gw_lists.shape[1]):
        valid_values = gw_lists[:, i][gw_lists[:, i] >= 0]
        if len(valid_values) > 0:
            means.append(np.mean(valid_values))
            stds.append(np.std(valid_values))
        else:
            means.append(np.nan)  # or some other placeholder for no valid values
            stds.append(np.nan)  # or some other placeholder for no valid values

    np.save(
        f"data/exploration/{expl_strategy_index}_means.npy",
        means,
    )
    np.save(
        f"data/exploration/{expl_strategy_index}_stds.npy",
        stds,
    )


def check_data_availability():
    import os
    
    base_dir = "data/exploration"
    
    if not os.path.exists(base_dir):
        return False
    
    # Check for all exploration strategies (0, 1, 4)
    for exp_strategy in [0, 1, 4]:
        required_files = [
            f'{exp_strategy}_auc.npy',
            f'{exp_strategy}_auc_mean.npy',
            f'{exp_strategy}_auc_std.npy',
            f'{exp_strategy}_means.npy',
            f'{exp_strategy}_stds.npy'
        ]
        
        for filename in required_files:
            file_path = os.path.join(base_dir, filename)
            if not os.path.exists(file_path):
                return False
    
    return True


def main():
    if not check_data_availability():
        for exp_strategy in [0, 1, 4]:
            process_exploration(1, 1001, exp_strategy)
        postprocess_exploration(exp_strategy)
