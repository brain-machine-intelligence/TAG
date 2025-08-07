import numpy as np
import matplotlib.pyplot as plt
from modules.model import TAGDecisionMaking
from modules.base import SRMB, DR
from modules.comparison import DP
from modules.env import MazeEnvironmentMultisubgoal
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pickle
import copy


# Function to process a single (ratio_index, index, seed) combination
def process_combination(args):
    if len(args) == 4:
        ratio_index, index, seed, gamma = args
        process_sfql = True
        debug = False
    elif len(args) == 5:
        ratio_index, index, seed, gamma, process_sfql = args  # Ensure all variables are passed in
        debug = False
    else:
        ratio_index, index, seed, gamma, debug, process_sfql = args
    index_str = f"{ratio_index}_{index}"

    # Initialize environment and agent INSIDE the function
    env = MazeEnvironmentMultisubgoal(10, 10, 1, 4, seed=seed)
    env.set_index(index_str)
    env.load_map("data/env/train/")
    env.set_goal_reward(10)
    env.set_feature_reward([3])

    sr = SRMB(10, 10, gamma=0.1)
    sr.set_index(index_str)
    sr.load_map_structure("data/env/train/")
    sr.set_w(env.return_state_reward())

    # Initialize agent correctly within the function
    tag = TAGDecisionMaking(10, 10, gamma, seed=seed)
    tag.set_index(index_str)
    tag.load_map("data/sr/train/")
    tag.set_start(env.start)
    tag.set_w(env.return_state_reward(), env.features_states.flatten())



    dr = DR(10, 10, terminal_states=[env.goal])
    dr.set_index(index_str)
    dr.load_map_structure("data/env/train/")
    dr.update_map_reward(env.return_state_reward())

    # Set up DP for optimal path calculation
    dp = DP(10, 10, gamma=gamma)
    dp.set_index(index_str)
    dp.load_structure("data/env/train/")
    dp.set_start_goal_states(env.start, env.goal)

    sub_temp = []
    for feat in range(len(env.features_states)):
        featured_states = env.features_states[feat].flatten().tolist()
        sub_temp += featured_states
    dp.set_subgoals(sub_temp)
    dp.set_w(env.return_state_reward())

    # Get optimal path and its metrics
    optimal_path = dp.get_low_level_answer()
    optimal_path.insert(0, env.start)

    optimal_reward_per_timestep = []

    optimal_discounted_return_list = []
    for t, state in enumerate(optimal_path):
        optimal_discounted_return_list.append(
            (gamma**t) * env.return_state_reward()[state]
        )
        if env.return_state_reward()[state] > 0:
            optimal_reward_per_timestep.append((env.return_state_reward()[state], t))
    optimal_discounted_return = sum(optimal_discounted_return_list)

    # Simulate agent's trajectory
    agent_discounted_return_list = []
    agent_reward_per_timestep_list = []
    agent_list = [tag, sr, dr]
    for agent_index in range(len(agent_list)):
        agent = agent_list[agent_index]
        state = env.reset()
        done = False
        discounted_return_list_per_agent = []
        reward_per_timestep_per_agent = []
        state_rewards = copy.deepcopy(env.return_state_reward())
        while not done:
            action = agent.sample_action(state)  # Ensure agent is initialized properly
            next_state, reward, done, t = env.step(action)
            state = next_state
            discounted_return_list_per_agent.append((gamma**t) * reward)
            if reward > 0:
                reward_per_timestep_per_agent.append((reward, t))
                if agent_index != 0:
                    state_rewards[state] = 0
                    if agent_index == 1: # sr
                        agent.set_w(state_rewards)
                    elif agent_index == 2: # dr
                        agent.update_map_reward(state_rewards)
                    else:
                        raise NotImplementedError()
        agent_discounted_return = sum(discounted_return_list_per_agent)
        agent_discounted_return_list.append(agent_discounted_return)
        agent_reward_per_timestep_list.append(reward_per_timestep_per_agent)
    if process_sfql:
        sfql_discounted_return_list, sfql_reward_per_timestep_list_list = train_sfql(env, 1000)
    else:
        with open(
            f"data/multi_subgoals/raw_data/results_{index_str}_{seed}.pkl", "rb"
        ) as f:
            results = pickle.load(f)
            _, _, sfql_discounted_return_list = results
        with open(
            f"data/multi_subgoals/raw_data/reward_per_timestep_{index_str}_{seed}.pkl", "rb"
        ) as f:
            results = pickle.load(f)
            _, _, sfql_reward_per_timestep_list_list = results
    results = [
        optimal_discounted_return,
        agent_discounted_return_list,  # per agent
        sfql_discounted_return_list,  # per episode
    ]
    with open(
        f"data/multi_subgoals/raw_data/results_{index_str}_{seed}.pkl", "wb"
    ) as f:
        pickle.dump(results, f)
    with open(
        f"data/multi_subgoals/raw_data/reward_per_timestep_{index_str}_{seed}.pkl", "wb"
    ) as f:
        pickle.dump(
            [
                optimal_reward_per_timestep,
                agent_reward_per_timestep_list,
                sfql_reward_per_timestep_list_list,
            ], f,
        )
    print(f"Processed {index_str}_{seed}")

    if debug:
        env.visualize(
            display=False,
            directory="display/result",
            trajectory=True,
            include_agent=False,
        )
        env.trajectory = optimal_path
        env.visualize(
            display=False,
            directory="display/optimal",
            trajectory=True,
            include_agent=False,
        )


def train_sfql(env, num_episode):
    from sf.agents.sfql import SFQL
    from sf.features.tabular import TabularSF
    from sf.tasks.gridworld import Shapes
    from sf.utils.config import parse_config_file

    # general training params
    config_params = parse_config_file("gridworld.cfg")
    agent_params = config_params["AGENT"]
    sfql_params = config_params["SFQL"]
    features_str = ["1", "2", "3"]
    if len(features_str) < env.num_features:
        raise ValueError("Not enough features in the environment")

    maze_shape = []
    for row in range(env.num_row):
        maze_shape.append([])
        for col in range(env.num_column):
            s = env.index_to_state(row, col)
            if s == env.goal:
                maze_shape[row].append("G")
            elif s == env.start:
                maze_shape[row].append("_")
            elif s in env.blocks:
                maze_shape[row].append("X")
            elif s in env.features_states.flatten():
                for feature_index in range(len(env.features_states)):
                    if s in env.features_states[feature_index]:
                        maze_shape[row].append(features_str[feature_index])
                        break
            else:
                maze_shape[row].append(" ")

    # tasks
    def generate_task():
        rewards = dict(zip(["1"], [3.0]))
        return Shapes(
            maze=np.array(maze_shape), shape_rewards=rewards, goal_reward=10.0
        )

    # agents
    sfql = SFQL(TabularSF(**sfql_params), **agent_params)
    # train each agent on a set of tasks
    
    sfql.reset()
    task = generate_task()
    sfql.train_on_task(task, num_episode)
    return sfql.total_discounted_return_list, sfql.reward_per_timestep_list_list


def simulate_navigation(mp=True):
    gamma = 0.995

    # Create all (ratio_index, index, seed) combinations
    all_combinations = [
        (ratio_index, index, seed, gamma, False)
        for ratio_index in range(1, 11)
        for index in range(1, 101)
        for seed in range(5)
    ]
    if not mp:
        for comb in tqdm(all_combinations):
            process_combination(comb)
    else:
        # Use a multiprocessing pool to parallelize the execution
        with Pool(cpu_count()) as pool:
            list(
                tqdm(
                    pool.imap(process_combination, all_combinations),
                    total=len(all_combinations),
                )
            )

def check_data_availability():
    import os
    
    base_dir = "data/multi_subgoals/raw_data"
    
    if not os.path.exists(base_dir):
        return False
    
    # Check for all combinations (ratio_index: 1-10, index: 1-100, seed: 0-4)
    for ratio_index in range(1, 11):
        for index in range(1, 101):
            for seed in range(5):
                index_str = f"{ratio_index}_{index}"
                required_files = [
                    f'results_{index_str}_{seed}.pkl',
                    f'reward_per_timestep_{index_str}_{seed}.pkl'
                ]
                
                for filename in required_files:
                    file_path = os.path.join(base_dir, filename)
                    if not os.path.exists(file_path):
                        return False
    
    return True


def main():
    if not check_data_availability():
        simulate_navigation(mp=True)
