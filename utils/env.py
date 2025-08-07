from itertools import permutations
from collections import deque
from tqdm import tqdm
import numpy as np


def bfs(visited, start_index, goal_index, adjacency_matrix, nonblocks):
    if start_index == goal_index:
        return [], 0
    else:
        queue = deque([[start_index]])
        shortest_paths = []
        shortest_path_length = 10000
        while queue:
            path = queue.popleft()
            current_index = path[-1]
            if not visited[current_index]:
                visited[current_index] = True
                for child in np.where(adjacency_matrix[current_index] == 1)[0]:
                    new_path = list(path)
                    new_path.append(child)
                    if len(new_path) - 1 > shortest_path_length:
                        continue
                    if child == goal_index:
                        if shortest_path_length >= len(new_path) - 1:
                            shortest_path = []
                            for node in new_path:
                                shortest_path.append(nonblocks[node])
                            if shortest_path_length > len(new_path) - 1:
                                shortest_path_length = len(new_path) - 1
                                shortest_paths = [shortest_path]
                            else:
                                shortest_paths.append(shortest_path)
                    else:
                        queue.append(new_path)
        return shortest_paths, shortest_path_length


def traverse_env(env):
    adjacency_matrix = env.return_adjacency_matrix()
    timesteps_list = []
    gt_results = []

    for start_index, goal_index in tqdm(permutations(range(len(env.nonblocks)), 2)):
        visited = [False] * np.shape(adjacency_matrix)[0]
        timesteps = np.ones(env.num_states) * (-10)
        timesteps[env.nonblocks] = -1
        shortest_paths, shortest_timestep = bfs(
            visited, start_index, goal_index, adjacency_matrix, env.nonblocks
        )
        for path in shortest_paths:
            timestep = 0
            for node in path[::-1]:
                assert timesteps[node] != -10
                if timesteps[node] == -1:
                    timesteps[node] = timestep
                else:
                    assert timesteps[node] == timestep
                timestep += 1
        timesteps_list.append(timesteps)
        start = env.nonblocks[start_index]
        goal = env.nonblocks[goal_index]
        gt_results.append((start, goal, shortest_timestep))
    return gt_results, timesteps_list
