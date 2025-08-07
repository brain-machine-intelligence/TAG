from modules.model import TAGUpdate, TAGTransfer
from modules.base import SRMB, DR
import numpy as np
import os
from itertools import combinations, permutations
from collections import deque
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy
import random
import scipy.linalg


def extract_subP(P0: np.ndarray, blocks: list[int]) -> np.ndarray:
    subP = P0[np.ix_(blocks, blocks)].copy()
    row_sums = subP.sum(axis=1)
    for i, s in enumerate(row_sums):
        if s > 0:
            subP[i, :] /= s
        else:
            subP[i, i] = 1.

    return subP


class ComposDR(DR):
    def __init__(self, num_row, num_column, lamda=10, terminal_states=[], stochastic_action=False):
        super(ComposDR, self).__init__(num_row, num_column, lamda=lamda, terminal_states=terminal_states, uniform_transition_matrix=True, stochastic_action=stochastic_action)
        self.update_map_structure()
        self.empty_transition_matrix = deepcopy(self.transition_matrix)
        self.empty_dr = deepcopy(self.sr) * self.gamma

    def update_map_por(self, blocks_list=[]):
        C_list = []
        R_list = []
        A_list = []
        for blocks in blocks_list:
            dr_sub = DR(self.num_row, self.num_column, self.lamda, terminal_states=self.terminal_states, uniform_transition_matrix=True)
            dr_sub.update_map_structure(blocks)
            dr_sub.transition_matrix[np.ix_(blocks, blocks)] = extract_subP(self.empty_transition_matrix, blocks)
            affected_states = np.where(np.sum(np.abs(self.empty_transition_matrix - dr_sub.transition_matrix), axis=1) > 0)[0]
            
            C = np.eye(self.num_states)[:, affected_states]
            C_list.append(C)
            R = (self.empty_transition_matrix - dr_sub.transition_matrix)[affected_states, :]
            R_list.append(R)
            Z = - R @ self.empty_dr @ C
            A = np.linalg.inv(np.eye(Z.shape[0]) - Z)
            A_list.append(A)
        
        C = np.concatenate(C_list, axis=1)
        A = scipy.linalg.block_diag(*A_list)
        R = np.concatenate(R_list, axis=0)

        self.dr = self.empty_dr - self.empty_dr @ C @ A @ R @ self.empty_dr

    def calculate_placefield(self):
        self.placefield = self.dr @ self.w
        if (self.placefield < 0).any():
            self.placefield[self.placefield < 0] = np.min(self.placefield[self.placefield > 0])

    def calculate_next_state_prob(self, state):
        prob = [0, 0, 0, 0]
        row, column = self.state_to_index(state)
        if 0 <= column - 1 < self.num_column:
            prob[0] = self.placefield[self.index_to_state(row, column - 1)]
        if 0 <= column + 1 < self.num_column:
            prob[1] = self.placefield[self.index_to_state(row, column + 1)]
        if 0 <= row - 1 < self.num_row:
            prob[2] = self.placefield[self.index_to_state(row - 1, column)]
        if 0 <= row + 1 < self.num_row:
            prob[3] = self.placefield[self.index_to_state(row + 1, column)]
        return prob

    def choose_action(self, prob):
        if np.sum(prob) == 0:
            if self.stochastic_action:
                return np.random.choice(4)
            else:
                raise ValueError("Invalid probability")
        else:
            if self.stochastic_action:
                prob = prob / np.sum(prob)
                return np.random.choice(4, p=prob)
            else:
                return np.argmax(prob)

    def sample_action(self, state):
        prob = self.calculate_next_state_prob(state)
        return self.choose_action(prob)
            

class DP(SRMB):
    """
    Credit: Nayeong Jeong (2024)
    """

    def __init__(self, num_row, num_column, gamma):
        super(DP, self).__init__(num_row, num_column, gamma)
        self.num_row = num_row
        self.num_column = num_column
        self.num_states = num_row * num_column
        self.num_actions = 4
        self.index = None
        self.w = np.zeros(self.num_states)
        self.agent_name = "sr"
        self.blocks = None
        self.walls = None
        self.start = 0
        self.maingoal = self.num_states - 1
        self.subgoals = []
        self.puddle_states = []
        self.num_sub = len(self.subgoals)
        self.node_indices = [self.start] + self.subgoals + [self.maingoal]
        self.num_node = len(self.node_indices)
        self.one_step_val = np.zeros((self.num_node, self.num_node))
        self.one_step_dist = np.zeros((self.num_node, self.num_node))
        self.one_step_path = dict()
        self.gamma = gamma
        self.optindex = None
        self.trj_val = dict()  # (0, 1, 2, 3, 4): value
        self.trj = dict()  # 0: (0, 1, 2, 3, 4)
        self.random_sr = None
        self.option_maps = dict()  # (i, t): option_value

    def set_index(self, index):
        self.index = str(index)

    def set_w(self, weight):
        self.w = np.array(weight)

    def set_start_goal_states(self, start, goal):
        self.start = start  # env.start
        self.maingoal = goal  # env.goal

    def set_subgoals(self, subgoals):
        self.subgoals = subgoals
        self.num_sub = len(self.subgoals)
        self.node_indices = [self.start] + self.subgoals + [self.maingoal]
        self.num_node = len(self.node_indices)
        self.one_step_val = np.zeros((self.num_node, self.num_node))
        self.one_step_dist = np.zeros((self.num_node, self.num_node))

    def set_puddles(self, puddles):
        self.puddle_states = puddles

    def generate_trajectory_candidates(self):
        if self.num_node < 2:
            raise ValueError("Number of nodes is less than 2")
        result = []
        elements = list(range(1, self.num_node - 1))
        for r in range(len(elements) + 1):
            for combination in combinations(elements, r):
                for perm in permutations(combination):
                    result.append((0,) + perm + (self.num_node - 1,))
        return result

    def set_trj_dict(self):
        candidates = self.generate_trajectory_candidates()
        for i, can in enumerate(candidates):
            self.trj[i] = can

    def load_structure(self, directory):
        if self.index is None:
            raise ValueError("Index not set")
        if os.path.exists(directory + "blocks_" + self.index + ".npy"):
            self.blocks = np.load(directory + "blocks_" + self.index + ".npy")
        if os.path.exists(directory + "walls_" + self.index + ".npy"):
            self.walls = np.load(directory + "walls_" + self.index + ".npy")

    def calculate_option_distance_bfs(self, init, term):
        def bfs(allow_puddles):
            grid = [[0] * self.num_column for _ in range(self.num_row)]
            for block in self.blocks:
                row, column = self.state_to_index(block)
                grid[row][column] = -1
            if term != self.maingoal:
                row, column = self.state_to_index(self.maingoal)
                grid[row][column] = -1
            if not allow_puddles:
                for puddle in self.puddle_states:
                    row, column = self.state_to_index(puddle)
                    grid[row][column] = -1
            queue = deque()
            sx, sy = self.state_to_index(init)
            gx, gy = self.state_to_index(term)
            queue.append((sx, sy, [self.index_to_state(sx, sy)]))
            dx = [0, 0, -1, 1]
            dy = [-1, 1, 0, 0]
            while queue:
                x, y, path = queue.popleft()
                for i in range(4):
                    nx = x + dx[i]
                    ny = y + dy[i]
                    if nx < 0 or nx >= self.num_row or ny < 0 or ny >= self.num_column:
                        continue
                    if grid[nx][ny] < 0:
                        continue
                    if grid[nx][ny] == 0:
                        grid[nx][ny] = grid[x][y] + 1
                        if nx == gx and ny == gy:
                            return len(path), path + [self.index_to_state(nx, ny)]
                        else:
                            queue.append((nx, ny, path + [self.index_to_state(nx, ny)]))
            return None

        result = bfs(allow_puddles=False)
        if result:
            return result

        result = bfs(allow_puddles=True)
        if result:
            return result

        return 1e9, []

    def visualize_grid(self, grid, init):
        grid[self.state_to_index(init)[0]][self.state_to_index(init)[1]] = 0.5
        grid_array = np.array(grid)
        plt.imshow(grid_array, cmap="viridis", interpolation="none")
        plt.colorbar()
        plt.title("Grid Visualization")
        plt.show()

    def calculate_onestepval(self):
        adj_matrix = np.ones((self.num_node, self.num_node))
        adj_matrix[:, 0] = 0
        adj_matrix[-1, 1:] = 0
        np.fill_diagonal(adj_matrix[1:-1, 1:-1], 0)
        for r, row in enumerate(adj_matrix):
            for c, _ in enumerate(row):
                if (
                    adj_matrix[r][c] == 1
                ):  # 자기 자신에게로 가거나/ S로/ G에서 가는 경우 제외하고 계산
                    dist, path = self.calculate_option_distance_bfs(
                        self.node_indices[r], self.node_indices[c]
                    )
                    self.one_step_dist[r][c] = dist
                    self.one_step_path[(r, c)] = path
                    if (
                        self.one_step_path[(r, c)] != []
                    ):  # 그 중에서 갈 수 있는 경우에만 Value 계산
                        self.one_step_val[r][c] = (
                            self.gamma ** self.one_step_dist[r][c]
                        ) * self.w[self.node_indices[c]]

    def calculate_trj_val(self):
        self.calculate_onestepval()
        for trj in self.trj.values():
            val = 0
            for i in range(1, len(trj)):
                if i == 1:
                    val += self.one_step_val[trj[i - 1]][trj[i]]
                else:
                    accum_dist = 0
                    for j in range(i - 1):
                        accum_dist += self.one_step_dist[trj[j]][trj[j + 1]]
                    val += (self.gamma**accum_dist) * self.one_step_val[trj[i - 1]][
                        trj[i]
                    ]
            self.trj_val[trj] = round(val, 3)

    def get_low_level_answer(self):
        self.set_trj_dict()
        self.calculate_trj_val()
        max_key = max(self.trj_val, key=self.trj_val.get)
        path_answer = []
        for i in range(1, len(max_key)):
            path_answer += self.one_step_path[(max_key[i - 1], max_key[i])][1:]
        return path_answer


class TAGTransferUCB(TAGTransfer):
    def __init__(self, num_row, num_column, gamma):
        super(TAGTransferUCB, self).__init__(num_row, num_column, gamma)
        self.transition_count = np.zeros(
            (self.num_states, 4)
        )  # Count visits for each state-action pair
        self.total_counts = np.zeros(self.num_states)  # Total counts for each state

    def sample_action_exploration(self, state):
        ucb_values = np.zeros(4)  # Assuming 4 possible actions (up, down, left, right)
        total_state_visits = self.total_counts[
            state
        ]  # Use pre-defined total count for the state

        for action in range(4):  # Loop over all possible actions
            if self.transition_count[state][action] > 0:
                # Exploration term based on visitation counts
                exploration_term = np.sqrt(
                    2
                    * np.log(total_state_visits)
                    / self.transition_count[state][action]
                )
                ucb_values[action] = exploration_term
            else:
                # Encourage exploration of unvisited actions
                ucb_values[action] = float("inf")

        # Select the action with the highest UCB value
        selected_action = np.argmax(ucb_values)

        return selected_action

    def update_map(self, current_state, action, next_state, bidirection=True):
        super().update_map(current_state, action, next_state, bidirection)
        # Update transition count and total count
        self.transition_count[current_state][action] += 1
        self.total_counts[current_state] += 1
        if current_state != next_state:
            self.transition_count[next_state][self.reverse_action[action]] += 1
            self.total_counts[next_state] += 1


class TAGTransferMBIE(TAGTransfer):
    def __init__(self, num_row, num_column, gamma):
        super(TAGTransferMBIE, self).__init__(num_row, num_column, gamma)
        self.transition_count = np.zeros(
            (self.num_states, 4)
        )  # Count visits for each state-action pair
        self.total_counts = np.zeros(self.num_states)  # Total counts for each state

    def sample_action_exploration(self, state):
        mbie_values = np.zeros(4)  # Assuming 4 possible actions (up, down, left, right)
        total_state_visits = self.total_counts[
            state
        ]  # Use pre-defined total count for the state

        for action in range(4):  # Loop over all possible actions
            if self.transition_count[state][action] > 0:
                # Exploration term based on visitation counts
                exploration_term = np.sqrt(
                    2
                    * np.log(total_state_visits)
                    / self.transition_count[state][action]
                )
                mbie_values[action] = exploration_term
            else:
                # Encourage exploration of unvisited actions
                mbie_values[action] = float("inf")

        # Select the action with the highest UCB value
        selected_action = np.argmax(mbie_values)

        return selected_action

    def update_map(self, current_state, action, next_state, bidirection=True):
        super().update_map(current_state, action, next_state, bidirection)
        # Update transition count and total count
        self.transition_count[current_state][action] += 1
        self.total_counts[current_state] += 1
        if current_state != next_state:
            self.transition_count[next_state][self.reverse_action[action]] += 1
            self.total_counts[next_state] += 1


class TAGTransferMBIEEB(TAGTransfer):
    def __init__(self, num_row, num_column, gamma, beta, bidirectional_count):
        super(TAGTransferMBIEEB, self).__init__(num_row, num_column, gamma)
        self.Q = np.ones(
            (self.num_states, 4)
        ) * 1 / (1 - self.gamma)
        self.beta = beta
        # self.timestep = 0 # for printing progress
        self.bidirectional_count = bidirectional_count
        self.SAS_count = np.zeros(
            (self.num_states, 4, self.num_states)
        )  # Count visits for each SAS'
        self.transition_count = np.zeros(
            (self.num_states, 4)
        )  # Count visits for each state-action pair
        self.total_counts = np.zeros(self.num_states)  # Total counts for each state
    

    def sample_action_exploration(self, state): 
        return np.argmax(self.Q[state])

    def update_Q(self):
        # agent 의 실제 경험만을 사용
        for state in range(self.num_states):
            for action in range(4):
                if self.transition_count[state][action] == 0:
                    self.Q[state][action] = 1 / (1 - self.gamma)
                    # n(s,a)=0이면 Q(s,a)는 1/(1-γ) (optimal initailization 인 채 유지)
                else:
                    transition_term = 0.0
                    for next_state in range(self.num_states):
                        transition_term += (
                            self.SAS_count[state][action][next_state] 
                            /self.transition_count[state][action]
                        )* max(self.Q[next_state])
                    # SAS transition 경험이 없으면 transition term은 0
                    exploration_term = np.sqrt(
                        1 / self.transition_count[state][action]
                    ) * self.beta
                    # n(s,a)가 nonzero인 이상 exploration term 은 계속 누적
                    self.Q[state][action] = self.gamma * transition_term + exploration_term

    def update_map(self, current_state, action, next_state, bidirection=True):
        super().update_map(current_state, action, next_state, bidirection)
        # Update SAS' count, transition count and total count
        self.update_Q()
        """
        if self.timestep % 400 == 0:
            print(self.Q)
        """
        self.SAS_count[current_state][action][next_state] += 1
        self.transition_count[current_state][action] += 1
        self.total_counts[current_state] += 1
        # self.timestep += 1
        
        if self.bidirectional_count and current_state != next_state: 
            self.SAS_count[next_state][self.reverse_action[action]][current_state] += 1
            self.transition_count[next_state][self.reverse_action[action]] += 1
            self.total_counts[next_state] += 1
        
        
class TAGTransferEGreedy(TAGTransfer):
    def __init__(self, num_row, num_column, gamma, epsilon):
        super(TAGTransferEGreedy, self).__init__(num_row, num_column, gamma)
        self.transition_count = np.zeros(
            (self.num_states, 4)
        )  # Count visits for each state-action pair
        self.total_counts = np.zeros(self.num_states)  # Total counts for each state
        self.epsilon = epsilon

    def sample_action_exploration(self, state):
        if np.random.rand() < self.epsilon:
            # With probability epsilon, take a random action
            selected_action = np.random.randint(4)  # Assuming 4 possible actions
        else:
            # With probability 1 - epsilon, select the least-sampled action
            selected_action = np.argmin(self.transition_count[state])

        return selected_action

    def update_map(self, current_state, action, next_state, bidirection=True):
        super().update_map(current_state, action, next_state, bidirection)
        # Update transition count and total count
        self.transition_count[current_state][action] += 1
        self.total_counts[current_state] += 1
        if current_state != next_state:
            self.transition_count[next_state][self.reverse_action[action]] += 1
            self.total_counts[next_state] += 1


class TAGUpdateBiased(TAGUpdate):
    def __init__(self, num_row, num_column):
        super(TAGUpdateBiased, self).__init__(num_row, num_column, 0.995)
        self.transition_count = np.zeros((self.num_states, self.num_states))
        self.exploration_mask = np.ones(self.num_states, dtype=bool)

    def update_map(self, current_state, action, next_state, bidirection=True):
        """
        SPE를 count-based로 교체
        """
        if self.sr[current_state][current_state] == 1:
            self.sr[current_state][current_state] = 1 / (1 - self.gamma)
        if not self.exploration_mask[current_state]:
            self.exploration_mask[current_state] = True
        if current_state != next_state:
            if not self.exploration_mask[next_state]:
                self.exploration_mask[next_state] = True
            current_legal_states_hot = (
                self.structure_head_border_stack()[current_state] >= self.t_threshold
            )  # step function
            if bidirection:
                next_legal_states_hot = (
                    self.structure_head_border_stack()[next_state] >= self.t_threshold
                )  # step function
                if self.sr[next_state][next_state] == 1:
                    self.sr[next_state][next_state] = 1 / (1 - self.gamma)

            if 0 < np.sum(current_legal_states_hot) <= 4:
                current_original_rw_prob = self.transition_count[
                    current_state
                ] / np.sum(self.transition_count[current_state])
                self.transition_count[current_state][next_state] += 1
                current_new_rw_prob = self.transition_count[current_state] / np.sum(
                    self.transition_count[current_state]
                )
                current_prob_diff = current_original_rw_prob - current_new_rw_prob
                self.woodbury_update(current_state, current_prob_diff)
                if bidirection:
                    if np.sum(next_legal_states_hot) == 0:
                        next_prob_diff = np.zeros(self.num_states)
                        next_prob_diff[current_state] = -1
                        next_prob_diff[next_state] = 1
                        self.transition_count[next_state][current_state] += 1
                    else:
                        next_original_rw_prob = self.transition_count[
                            next_state
                        ] / np.sum(self.transition_count[next_state])
                        if next_legal_states_hot[next_state]:
                            next_legal_states_hot[next_state] = False
                        next_legal_states_hot[current_state] = True
                        self.transition_count[next_state][current_state] += 1
                        next_new_rw_prob = self.transition_count[next_state] / np.sum(
                            self.transition_count[next_state]
                        )
                        next_prob_diff = next_original_rw_prob - next_new_rw_prob
                    self.woodbury_update(next_state, next_prob_diff)
            elif np.sum(current_legal_states_hot) == 0:
                current_prob_diff = np.zeros(self.num_states)
                current_prob_diff[current_state] = 1
                current_prob_diff[next_state] = -1
                self.transition_count[current_state][next_state] += 1
                self.woodbury_update(current_state, current_prob_diff)
                if bidirection:
                    next_prob_diff = -current_prob_diff
                    self.transition_count[next_state][current_state] += 1
                    self.woodbury_update(next_state, next_prob_diff)
            else:
                raise ValueError()

            left_singvecs, right_singvecs, singvals = self.decompose_map_svd()
            self.G = left_singvecs.T
            self.G_inv = right_singvecs
            self.lambdas = singvals

        else:
            current_row, current_column = self.state_to_index(current_state)
            if (0 <= current_row + self.one_step_deltas[action][0] < self.num_row) and (
                0 <= current_column + self.one_step_deltas[action][1] < self.num_column
            ):
                block_state = self.index_to_state(
                    current_row + self.one_step_deltas[action][0],
                    current_column + self.one_step_deltas[action][1],
                )
                self.exploration_mask[block_state] = True

    def save_map(self, directory):
        np.save(directory + "rgct_biased_" + self.index + ".npy", self.sr)
        np.save(
            directory + "exploration_mask_biased_" + self.index + ".npy",
            self.exploration_mask,
        )

    def load_map(self, directory):
        self.sr = np.load(directory + "rgct_biased_" + self.index + ".npy")
        self.update_grid()
        if os.path.exists(directory + "exploration_mask_biased_" + self.index + ".npy"):
            self.exploration_mask = np.load(
                directory + "exploration_mask_biased_" + self.index + ".npy"
            )
        else:
            self.exploration_mask = np.ones(self.num_states, dtype=bool)
