import numpy as np
import scipy.linalg as LA
import utils.plot
import matplotlib.pyplot as plt
import os
import pickle
EPSILON = 1e-9


class SR:
    def __init__(self, num_row, num_column, stochastic_action=False):
        self.num_row = num_row
        self.num_column = num_column
        self.num_states = num_row * num_column
        self.sr = np.identity(self.num_states)
        self.index = None
        self.w = np.zeros(self.num_states)
        self.agent_name = "sr"
        self.stochastic_action = stochastic_action
        self.placefield = None

    def reset_map(self):
        self.sr = np.identity(self.num_states)

    def set_map(self, new_sr, row_indices=None, column_indices=None):
        if row_indices is None:
            row_indices = np.arange(np.shape(new_sr)[0])
        if column_indices is None:
            column_indices = np.arange(np.shape(new_sr)[1])
        for row_index in range(len(row_indices)):
            for column_index in range(len(column_indices)):
                self.sr[row_indices[row_index], column_indices[column_index]] = new_sr[
                    row_index, column_index
                ]

    def new_task(self, w):
        self.w = w
        self.calculate_placefield()

    def return_map(self):
        return self.sr

    def decompose_map_svd(self):
        left_singularvecs, singularvals, right_singularvecs = np.linalg.svd(self.sr)
        return left_singularvecs.T, right_singularvecs, singularvals

    def decompose_map_eigen(self):
        eigenvals, eigenvecs = LA.eig(self.sr)
        idx_desc = np.argsort(eigenvals.real)[::-1]
        eigenvals_sorted = eigenvals[idx_desc]
        eigenvecs_sorted = eigenvecs[:, idx_desc]
        eigenvecs_sorted = eigenvecs_sorted.T
        return eigenvecs_sorted.real, eigenvals_sorted.real

    def decompose_map_nonnegpca(self):
        raise NotImplementedError()

    def return_successormap(self, eigenmap_dim=10):
        eigenvals, eigenvecs = LA.eig(self.sr)
        idx_desc = np.argsort(eigenvals.real)[::-1]
        eigenvals = eigenvals[idx_desc]
        eigenvecs = eigenvecs[:, idx_desc]
        if eigenmap_dim > self.num_states - 1:
            eigenmap_dim = self.num_states - 1
        # Find the indices where eigenvalues are positive and less than eigenmap_dim
        positive_indices = np.where(eigenvals[1:eigenmap_dim + 1].real > 0)[0] + 1

        # Slice the eigenvecs and eigenvals using the positive indices
        selected_eigenvecs = eigenvecs[:, positive_indices].real
        selected_eigenvals_sqrt = np.sqrt(eigenvals[positive_indices].real)

        # Perform the matrix multiplication
        return np.matmul(selected_eigenvecs, np.diag(selected_eigenvals_sqrt))
    
    def save_map(self, directory):
        np.save(
            directory + "{}_".format(self.agent_name) + self.index + ".npy", self.sr
        )

    def load_map(self, directory):
        if self.index is None:
            raise ValueError("Index not set")
        self.sr = np.load(
            directory + "{}_".format(self.agent_name) + self.index + ".npy"
        )

    def set_index(self, index):
        self.index = str(index)

    def save_basis(self, directory):
        left_singvecs, right_singvecs, singvals = self.decompose_map_svd()
        np.save(directory + "left_singularvecs_" + self.index + ".npy", left_singvecs)
        np.save(directory + "right_singularvecs_" + self.index + ".npy", right_singvecs)
        np.save(directory + "singularvals_" + self.index + ".npy", singvals)

    def index_to_state(self, row, column, num_column=None):
        if num_column is None:
            num_column = self.num_column
        return row * num_column + column

    def state_to_index(self, state, num_column=None):
        if num_column is None:
            num_column = self.num_column
        return np.divmod(state, num_column)

    def visualize_map(self, display=False):
        plt.figure(1, figsize=(self.num_column * 3, self.num_row * 3))
        plt.subplots_adjust(wspace=0.1, hspace=0.0001)
        for k in range(self.num_states):
            ax = plt.subplot(self.num_row, self.num_column, k + 1)
            ax.imshow(self.sr.T[k].reshape(self.num_row, self.num_column))
        if display:
            plt.show()
        else:
            plt.savefig("./display/{}_{}.png".format(self.agent_name, self.index))
            plt.close()

    def rotate_map(self, rotation_factor):
        for _ in range(rotation_factor):  # 한번에 시계반대방향으로 90도씩
            new_sr = np.zeros((self.num_states, self.num_states))
            for row_state in range(self.num_states):
                row_row, row_column = self.state_to_index(row_state)
                for column_state in range(self.num_states):
                    column_row, column_column = self.state_to_index(column_state)
                    new_sr[
                        utils.plot.index_to_state(
                            self.num_column - 1 - row_column, row_row, self.num_row
                        )
                    ][
                        utils.plot.index_to_state(
                            self.num_column - 1 - column_column,
                            column_row,
                            self.num_row,
                        )
                    ] = self.sr[
                        row_state
                    ][
                        column_state
                    ]
            self.sr = new_sr
            column = self.num_column
            self.num_column = self.num_row
            self.num_row = column

    def calculate_placefield(self):
        self.placefield = np.matmul(self.sr, self.w)

    def set_w(self, w):
        self.w = w

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
            raise ValueError()
            # return np.random.choice(4)
        else:
            """index = []
            for a in range(4):
                if np.max(prob) == prob[a]:
                    index.append(a)
            if len(index):
                return np.random.choice(index)
            else:
                return np.random.choice(4)"""
            if self.stochastic_action:
                prob = prob / np.sum(prob)
                return np.random.choice(4, p=prob)
            else:
                return np.argmax(prob)

    def sample_action(self, state):
        prob = self.calculate_next_state_prob(state)
        return self.choose_action(prob)


class SRTD(SR):
    def __init__(self, num_row, num_column, gamma=0.995):
        super(SRTD, self).__init__(num_row, num_column)
        self.e_trace = np.zeros(self.num_states)
        self.gamma = gamma
        self.lmda = 0.9
        self.alpha_sr = 0.3
        self.visit_list = np.zeros(self.num_states)
        self.agent_name = "srtd"

    def update_map(self, current_state, next_state, use_e=False):
        if current_state != next_state:
            self.visit_list[current_state] = 1
            self.visit_list[next_state] = 1
            td_error = (
                np.array([x == current_state for x in range(self.num_states)])
                + self.gamma * self.sr[next_state]
                - self.sr[current_state]
            )
            if use_e:
                self.e_trace *= self.lmda * self.gamma
                self.e_trace[current_state] += 1
                update_amount = np.matmul(
                    self.e_trace.reshape(-1, 1), td_error.reshape(1, -1)
                )
                self.sr += self.alpha_sr * update_amount
            else:
                self.sr[current_state] += self.alpha_sr * td_error

    def update(self, state, action, next_state, reward):
        self.update_map(state, next_state, use_e=True)
        self.w[next_state] = reward

    def save_map(self, directory):
        np.save(directory + "srtd_" + self.index + ".npy", self.sr)
        np.save(directory + "e_trace_" + self.index + ".npy", self.e_trace)

    def load_map(self, directory):
        if self.index is None:
            raise ValueError("Index not set")
        self.sr = np.load(directory + "srtd_" + self.index + ".npy")
        self.e_trace = np.load(directory + "e_trace_" + self.index + ".npy")


class SRMB(SR):
    def __init__(self, num_row, num_column, gamma=0.995, stochastic_action=False):
        super(SRMB, self).__init__(num_row, num_column, stochastic_action)
        self.num_actions = 4
        self.transition_matrix = np.zeros((self.num_states, self.num_states))
        self.policy = np.ones((self.num_states, self.num_actions)) / self.num_actions
        self.available_actions = np.zeros(
            (self.num_states, self.num_actions), dtype=bool
        )
        self.gamma = gamma
        self.alpha_pi = 0.1
        self.agent_name = "srmb"

    def set_transition_matrix(self, weight_matrix):
        for row_index in range(weight_matrix.shape[0]):
            self.transition_matrix[row_index] = np.exp(
                weight_matrix[row_index]
            ) / np.sum(np.exp(weight_matrix[row_index]))

    def load_structure(self, directory):
        if self.index is None:
            raise ValueError("Index not set")
        if os.path.exists(directory + "blocks_" + self.index + ".npy"):
            blocks = np.load(directory + "blocks_" + self.index + ".npy")
        else:
            blocks = None
        if os.path.exists(directory + "walls_" + self.index + ".npy"):
            walls = np.load(directory + "walls_" + self.index + ".npy")
        else:
            walls = None
        return blocks, walls

    def load_map_structure(self, directory, rd_prob=0.5):
        blocks, walls = self.load_structure(directory)
        if blocks is None and walls is None:
            return False
        else:
            self.update_map_structure(blocks, walls, rd_prob)
            return True

    def update_t(self, state, action, next_state):
        if state != next_state:
            self.available_actions[state][action] = True
            self.available_actions[next_state][action + (action % 2) * (-2) + 1] = True
            self.policy[state] = (
                self.alpha_pi * np.array([x == action for x in range(self.num_actions)])
                + (1 - self.alpha_pi) * self.policy[state]
            )
            self.transition_matrix = np.zeros((self.num_states, self.num_states))
            for s in range(self.num_states):
                sum_pi = 0
                transition_vector = np.zeros(self.num_states)
                row, column = self.state_to_index(s)
                for a in range(self.num_actions):
                    if self.available_actions[s][a]:
                        sum_pi += self.policy[s][a]
                        if a == 0:
                            if column > 0:
                                transition_vector[
                                    self.index_to_state(row, column - 1)
                                ] = self.policy[s][a]
                        elif a == 1:
                            if column < self.num_column - 1:
                                transition_vector[
                                    self.index_to_state(row, column + 1)
                                ] = self.policy[s][a]
                        elif a == 2:
                            if row > 0:
                                transition_vector[
                                    self.index_to_state(row - 1, column)
                                ] = self.policy[s][a]
                        else:
                            if row < self.num_row - 1:
                                transition_vector[
                                    self.index_to_state(row + 1, column)
                                ] = self.policy[s][a]
                if sum_pi:
                    transition_vector /= sum_pi
                self.transition_matrix[s] = transition_vector

    def compute_sr(self):
        self.sr = np.linalg.inv(
            np.identity(self.num_states) - self.gamma * self.transition_matrix
        )

    def update_map(self, state, action, next_state):
        self.update_t(state, action, next_state)
        self.compute_sr()

    def update(self, state, action, next_state, reward):
        self.update_map(state, action, next_state)
        self.w[next_state] = reward

    def calculate_t(self, blocks=None, walls=None, rd_prob=0.5):
        self.transition_matrix = np.zeros((self.num_states, self.num_states))
        if blocks is None:
            blocks = []
        if walls is None:
            walls = []
        for i in range(self.num_row):
            for j in range(self.num_column):
                state = self.index_to_state(i, j)
                if state not in blocks:
                    available_action = [True, True, True, True]  # left, right, up, down
                    state_list = [
                        self.index_to_state(i, j - 1),
                        self.index_to_state(i, j + 1),
                        self.index_to_state(i - 1, j),
                        self.index_to_state(i + 1, j),
                    ]  # state indices of left, right, up, down respectively
                    if (
                        j == 0
                        or self.index_to_state(i, j - 1) in blocks
                        or (self.index_to_state(i, j - 1), self.index_to_state(i, j))
                        in walls
                    ):
                        available_action[0] = False
                    if (
                        j == self.num_column - 1
                        or self.index_to_state(i, j + 1) in blocks
                        or (self.index_to_state(i, j), self.index_to_state(i, j + 1))
                        in walls
                    ):
                        available_action[1] = False
                    if (
                        i == 0
                        or self.index_to_state(i - 1, j) in blocks
                        or (self.index_to_state(i - 1, j), self.index_to_state(i, j))
                        in walls
                    ):
                        available_action[2] = False
                    if (
                        i == self.num_row - 1
                        or self.index_to_state(i + 1, j) in blocks
                        or (self.index_to_state(i, j), self.index_to_state(i + 1, j))
                        in walls
                    ):
                        available_action[3] = False
                    self.available_actions[state] = available_action
                    lu_prob = 1 - rd_prob
                    probs = [lu_prob / 2, rd_prob / 2, lu_prob / 2, rd_prob / 2]
                    av_actions = np.array(available_action)
                    unav_actions = 1 - av_actions
                    unav_prob = np.sum(probs * unav_actions)
                    av_probs = probs * av_actions
                    if np.sum(av_probs) > EPSILON:
                        av_props = av_probs / np.sum(av_probs)
                    av_probs += unav_prob * av_props
                    for k in range(4):
                        if available_action[k]:
                            self.transition_matrix[state][state_list[k]] = float(
                                av_probs[k]
                            )

    def decompose_t_svd(self):
        left_singularvecs, singularvals, right_singularvecs = np.linalg.svd(
            self.transition_matrix
        )
        return left_singularvecs.T, right_singularvecs, singularvals

    def decompose_t_eigen(self):
        eigenvals, eigenvecs = LA.eig(self.transition_matrix)
        eigenvecs = eigenvecs.T
        return eigenvecs.real, eigenvals.real

    def update_map_structure(self, blocks=None, walls=None, rd_prob=0.5):
        self.calculate_t(blocks, walls, rd_prob)
        self.compute_sr()

    def save_map(self, directory):
        np.save(directory + "srmb_" + self.index + ".npy", self.sr)
        np.save(directory + "transition_matrix_" + self.index, self.transition_matrix)
        np.save(directory + "policy_" + self.index + ".npy", self.policy)
        np.save(
            directory + "available_actions_" + self.index + ".npy",
            self.available_actions,
        )

    def load_map(self, directory):
        if self.index is None:
            raise ValueError("Index not set")
        self.sr = np.load(directory + "srmb_" + self.index + ".npy")
        self.transition_matrix = np.load(
            directory + "transition_matrix_" + self.index + ".npy"
        )
        self.policy = np.load(directory + "policy_" + self.index + ".npy")
        self.available_actions = np.load(
            directory + "available_actions_" + self.index + ".npy"
        )


class DR(SRMB):
    def __init__(self, num_row, num_column, lamda=1, terminal_states=[], uniform_transition_matrix=True, stochastic_action=False):
        super(DR, self).__init__(num_row, num_column, stochastic_action=stochastic_action)
        self.lamda = lamda
        self.gamma = 1 / np.exp(1 / self.lamda)
        self.reward_temp_param = 1
        self.terminal_states = np.sort(np.array(terminal_states))
        self.nonterminal_states = np.sort(
            np.setdiff1d(np.arange(self.num_states), terminal_states)
        )
        self.num_nonterminal_states = len(self.nonterminal_states)
        self.num_terminal_states = len(self.terminal_states)
        self.uniform_transition_matrix = uniform_transition_matrix
        if self.num_terminal_states and not uniform_transition_matrix:
            self.nonterminal_transition_matrix = np.zeros(
                (self.num_nonterminal_states, self.num_nonterminal_states)
            )
            self.terminal_transition_matrix = np.zeros(
                (self.num_nonterminal_states, self.num_terminal_states)
            )   
            self.nonterminal_rewards = -1 * np.ones(self.num_nonterminal_states)
            self.terminal_rewards = np.zeros(self.num_terminal_states)
        else:
            self.nonterminal_transition_matrix = self.transition_matrix
            self.terminal_transition_matrix = None
            self.nonterminal_rewards = -1 * np.ones(self.num_states)
            self.terminal_rewards = None

        self.alpha_pi = 0
        self.agent_name = "dr"
        self.dr = None

    def compute_dr(self):
        self.dr = np.linalg.inv(
            np.diag(np.exp(-self.nonterminal_rewards / self.lamda))
            - self.nonterminal_transition_matrix
        )
        self.compute_sr()

    def segregate_transition_matrix(self):
        if self.num_terminal_states and not self.uniform_transition_matrix:
            for nts_idx in range(len(self.nonterminal_states)):
                for ts_idx in range(len(self.terminal_states)):
                    if self.transition_matrix[self.nonterminal_states[nts_idx]][
                        self.terminal_states[ts_idx]
                    ] > 0:
                        self.terminal_transition_matrix[nts_idx][ts_idx] = 1
                        self.transition_matrix[self.nonterminal_states[nts_idx]][
                            self.terminal_states[ts_idx]
                        ] = 0
            for ts_idx in range(len(self.terminal_states)):
                if np.sum(self.transition_matrix[self.terminal_states[ts_idx]]) > 0:
                    self.transition_matrix[self.terminal_states[ts_idx]] /= np.sum(
                        self.transition_matrix[self.terminal_states[ts_idx]]
                    )
            for nts_idx_1 in range(len(self.nonterminal_states)):
                if np.sum(self.transition_matrix[self.nonterminal_states[nts_idx_1]]) > 0:
                    self.transition_matrix[self.nonterminal_states[nts_idx_1]] /= np.sum(
                        self.transition_matrix[self.nonterminal_states[nts_idx_1]]
                    )
            for nts_idx_1 in range(len(self.nonterminal_states)):
                for nts_idx_2 in range(len(self.nonterminal_states)):
                    self.nonterminal_transition_matrix[nts_idx_1][nts_idx_2] = (
                        self.transition_matrix[self.nonterminal_states[nts_idx_1]][
                            self.nonterminal_states[nts_idx_2]
                        ]
                    )
        else:
            self.nonterminal_transition_matrix = self.transition_matrix


    def update_map_structure(self, blocks=None, walls=None, rd_prob=0.5):
        super(DR, self).calculate_t(blocks, walls, rd_prob)
        self.segregate_transition_matrix()
        self.compute_dr()

    def update_map_reward(self, rewards):
        transformed_rewards = - np.exp(-rewards / self.reward_temp_param)  # DR reward 처리 변경
        if self.num_terminal_states:
            for nts_idx in range(len(self.nonterminal_states)):
                self.nonterminal_rewards[nts_idx] = transformed_rewards[
                    self.nonterminal_states[nts_idx]
                ]
            for ts_idx in range(len(self.terminal_states)):
                self.terminal_rewards[ts_idx] = transformed_rewards[self.terminal_states[ts_idx]]
        else:
            self.nonterminal_rewards = np.array(transformed_rewards)
        self.compute_dr()

    def update_map(self, state, action, next_state):
        raise NotImplementedError("Cannot use update_map without reward in DR")

    def load_map_structure(self, directory, rd_prob=0.5):
        blocks, walls = super(DR, self).load_structure(directory)
        if blocks is None and walls is None:
            return False
        else:
            self.update_map_structure(blocks, walls, rd_prob)
            return True

    def update_map(self, state, action, next_state, reward):
        super(DR, self).update_t(state, action, next_state)
        self.segregate_transition_matrix()
        if next_state in self.nonterminal_states:
            self.nonterminal_rewards[
                np.argwhere(self.nonterminal_states == next_state)[0][0]
            ] = (reward - 1)
        else:
            self.terminal_rewards[
                np.argwhere(self.terminal_states == next_state)[0][0]
            ] = reward
        self.compute_dr()

    def save_map(self, directory):
        np.save(directory + "dr_" + self.index + ".npy", self.sr)
        np.save(
            directory + "nonterminal_transition_matrix_" + self.index + ".npy",
            self.nonterminal_transition_matrix,
        )
        np.save(
            directory + "terminal_transition_matrix_" + self.index + ".npy",
            self.terminal_transition_matrix,
        )
        np.save(
            directory + "nonterminal_rewards_" + self.index, self.nonterminal_rewards
        )
        np.save(directory + "terminal_rewards_" + self.index, self.terminal_rewards)
        np.save(directory + "nonterminal_states_" + self.index, self.nonterminal_states)
        np.save(directory + "terminal_states_" + self.index, self.terminal_states)

    def load_map(self, directory):
        if self.index is None:
            raise ValueError("Index not set")
        self.dr = np.load(directory + "dr_" + self.index + ".npy")

    def visualize_map(self, display=False):  # placefield 계산 방식에 맞추어 업데이트
        fig_size_num = 6
        if self.num_nonterminal_states == 0:
            display_sr = np.log(self.dr + EPSILON)
            plt.figure(
                1, figsize=(self.num_column * fig_size_num, self.num_row * fig_size_num)
            )
            plt.subplots_adjust(wspace=0.1, hspace=0.0001)
            for k in range(self.num_states):
                ax = plt.subplot(self.num_row, self.num_column, k + 1)
                ax.imshow(display_sr.T[k].reshape(self.num_row, self.num_column))
        else:
            terminal_sr = self.dr @ self.terminal_transition_matrix
            display_sr = np.log(terminal_sr + EPSILON)
            plt.figure(
                1, figsize=(self.num_terminal_states * fig_size_num, 1 * fig_size_num)
            )
            plt.subplots_adjust(wspace=0.1, hspace=0.0001)
            for k in range(self.num_terminal_states):
                display_placefield = display_sr.T[k]
                display_placefield_whole = []
                for j in range(self.num_states):
                    if j in self.nonterminal_states:
                        display_placefield_whole.append(display_placefield[j])
                    else:
                        if j == self.terminal_states[k]:
                            display_placefield_whole.append(np.max(display_placefield) + EPSILON)
                        else:
                            display_placefield_whole.append(0)
                ax = plt.subplot(self.num_row, self.num_column, k + 1)
                ax.imshow(np.array(display_placefield_whole).reshape(self.num_row, self.num_column))
        if display:
            plt.show()
        else:
            plt.savefig("./display/{}_{}.png".format(self.agent_name, self.index))
            plt.close()

    def calculate_next_state_prob(self, placefield, state):
        if self.num_nonterminal_states:
            prob = [0, 0, 0, 0]
            if state in self.nonterminal_states:
                row, column = self.state_to_index(state)
                if 0 <= column - 1 < self.num_column:
                    next_state = self.index_to_state(row, column - 1)
                    if next_state in self.nonterminal_states:
                        prob[0] = placefield[next_state]
                    elif next_state in self.terminal_states:
                        prob[0] = np.max(placefield) + EPSILON
                    else:
                        raise ValueError("Next state not found")
                if 0 <= column + 1 < self.num_column:
                    next_state = self.index_to_state(row, column + 1)
                    if next_state in self.nonterminal_states:
                        prob[1] = placefield[next_state]
                    elif next_state in self.terminal_states:
                        prob[1] = np.max(placefield) + EPSILON
                    else:
                        raise ValueError("Next state not found")
                if 0 <= row - 1 < self.num_row:
                    next_state = self.index_to_state(row - 1, column)
                    if next_state in self.nonterminal_states:
                        prob[2] = placefield[next_state]
                    elif next_state in self.terminal_states:
                        prob[2] = np.max(placefield) + EPSILON
                    else:
                        raise ValueError("Next state not found")
                if 0 <= row + 1 < self.num_row:
                    next_state = self.index_to_state(row + 1, column)
                    if next_state in self.nonterminal_states:
                        prob[3] = placefield[next_state]
                    elif next_state in self.terminal_states:
                        prob[3] = np.max(placefield) + EPSILON
                    else:
                        raise ValueError("Next state not found")
        else:
            prob = (DR, self).calculate_next_state_prob(placefield, state)
        return prob

    def sample_action(self, state):  # sample_action 함수 수정 - terminal_state 있을 때와 없을 때 모두 반영
        if self.num_terminal_states and not self.uniform_transition_matrix:
            placefield = (
                self.dr
                @ self.terminal_transition_matrix
                @ np.exp(self.terminal_rewards / self.lamda).reshape(-1, 1)
            ).flatten()
        else:
            placefield = np.matmul(self.dr, self.nonterminal_rewards)
        prob = self.calculate_next_state_prob(placefield, state)
        return self.choose_action(prob)


class SRDyna(SRTD):
    """
    Modified version of SR-Dyna for deterministic state-transition dynamics
    """

    def __init__(self, num_row, num_column, gamma=0.995):
        super(SRDyna, self).__init__(num_row, num_column, gamma)
        self.num_row = num_row
        self.num_column = num_column
        self.num_states = num_row * num_column
        self.num_actions = 4
        self.gamma = gamma
        self.alpha_w = 0.3
        self.num_samples = 10000
        self.sa_samples = []
        self.saprime_samples = [[] for _ in range(self.num_states * self.num_actions)]
        self.reward_samples = [0 for _ in range(self.num_states * self.num_actions)]
        self.epsilon = 0.1
        self.pdf = self.exponential(np.arange(100) + 1, 1 / 5)
        self.pdf /= np.sum(self.pdf)
        self.cdf = np.cumsum(self.pdf)
        self.index = None

    def sa_to_index(self, state, action):
        return state * self.num_actions + action

    def index_to_sa(self, index):
        return index // self.num_actions, index % self.num_actions

    def return_qvalue(self, state, action):
        raise NotImplementedError()

    def update_map(self, state, action, next_state, next_action):
        super(SRDyna, self).update_map(state, next_state)

    def update_w(self, state, action, next_state, next_action, reward):
        raise NotImplementedError()

    def update(self, state, action, next_state, next_action, reward):
        if state != next_state:
            self.update_map(state, action, next_state, next_action)
            if self.sa_to_index(state, action) not in self.sa_samples:
                self.sa_samples.append(self.sa_to_index(state, action))
            self.reward_samples[self.sa_to_index(state, action)] = reward
            if (
                self.sa_to_index(next_state, next_action)
                not in self.saprime_samples[self.sa_to_index(state, action)]
            ):
                self.saprime_samples[self.sa_to_index(state, action)].append(
                    self.sa_to_index(next_state, next_action)
                )
        if len(self.sa_samples):
            self.offline_update(10)

    def exponential(self, x_list, mu):
        return np.array([(1 / mu) * np.exp(-x / mu) for x in x_list])

    def on_policy(self, state):
        raise NotImplementedError()
    
    def offline_update(self, num_samples):
        """
        현재로선 random policy만 반영하도록 되어있음
        random walk에 대한 학습을 더 잘하도록 uniform random sample
        """
        for k in range(num_samples):
            sa_index = np.random.choice(self.sa_samples)
            saprime_index = np.random.choice(self.saprime_samples[sa_index])
            state, action = self.index_to_sa(sa_index)
            next_state, next_action = self.index_to_sa(saprime_index)
            self.update_map(state, action, next_state, next_action)

    def return_map(self):
        """
        선택 가능한 action에 대해서만 나누도록 수정
        """
        return self.sr

    def save_map(self, directory):
        np.save(directory + "srdyna_" + self.index + ".npy", self.sr)
        np.save(directory + "w_" + self.index + ".npy", self.w)
        with open(directory + "sa_samples_" + self.index + ".list", "wb") as f:
            pickle.dump(self.sa_samples, f)
        with open(directory + "saprime_samples_" + self.index + ".list", "wb") as f:
            pickle.dump(self.saprime_samples, f)
        with open(directory + "reward_samples_" + self.index + ".list", "wb") as f:
            pickle.dump(self.reward_samples, f)

    def load_map(self, directory):
        if self.index is None:
            raise ValueError("Index not set")
        self.sr = np.load(directory + "srdyna_" + self.index + ".npy")
        self.w = np.load(directory + "w_" + self.index + ".npy")
        self.sa_samples = pickle.load(
            open(directory + "sa_samples_" + self.index + ".list", "rb")
        )
        self.saprime_samples = pickle.load(
            open(directory + "saprime_samples_" + self.index + ".list", "rb")
        )
        self.reward_samples = pickle.load(
            open(directory + "reward_samples_" + self.index + ".list", "rb")
        )

    def set_index(self, index):
        self.index = str(index)


class SRDynaOriginal:
    """
    Russek, Momennejad, Botvinick, Gershman, Daw, PLOS Comp Biol, 2017
    """

    def __init__(self, num_row, num_column, gamma=0.995):
        self.num_row = num_row
        self.num_column = num_column
        self.num_states = num_row * num_column
        self.num_actions = 4
        self.sr = np.identity(self.num_states * self.num_actions)
        self.w = np.zeros(self.num_states * self.num_actions)
        self.gamma = gamma
        self.alpha_sr = 0.3
        self.alpha_w = 0.3
        self.num_samples = 10000
        self.sa_samples = []
        self.saprime_samples = [[] for _ in range(self.num_states * self.num_actions)]
        self.reward_samples = [0 for _ in range(self.num_states * self.num_actions)]
        self.epsilon = 0.1
        self.pdf = self.exponential(np.arange(100) + 1, 1 / 5)
        self.pdf /= np.sum(self.pdf)
        self.cdf = np.cumsum(self.pdf)
        self.index = None

    def sa_to_index(self, state, action):
        return state * self.num_actions + action

    def index_to_sa(self, index):
        return index // self.num_actions, index % self.num_actions

    def return_qvalue(self, state, action):
        return np.dot(self.sr[self.sa_to_index(state, action)], self.w)

    def update_map(self, state, action, next_state, next_action):
        td_error = (
            np.array(
                [
                    x == self.sa_to_index(state, action)
                    for x in range(self.num_states * self.num_actions)
                ]
            )
            + self.gamma * self.sr[self.sa_to_index(next_state, next_action)]
            - self.sr[self.sa_to_index(state, action)]
        )
        self.sr[self.sa_to_index(state, action)] += self.alpha_sr * td_error

    def update_w(self, state, action, next_state, next_action, reward):
        delta = (
            reward
            + self.gamma * self.return_qvalue(next_state, next_action)
            - self.return_qvalue(state, action)
        )
        self.w += self.alpha_w * delta * self.sr[self.sa_to_index(state, action)]

    def update(self, state, action, next_state, next_action, reward):
        if state != next_state:
            self.update_w(state, action, next_state, next_action, reward)
            self.update_map(state, action, next_state, next_action)
            if self.sa_to_index(state, action) not in self.sa_samples:
                self.sa_samples.append(self.sa_to_index(state, action))
            self.reward_samples[self.sa_to_index(state, action)] = reward
            # if len(self.sa_samples) > self.num_samples:
            #    _ = self.sa_samples.pop(0)
            if (
                self.sa_to_index(next_state, next_action)
                not in self.saprime_samples[self.sa_to_index(state, action)]
            ):
                self.saprime_samples[self.sa_to_index(state, action)].append(
                    self.sa_to_index(next_state, next_action)
                )
        # if len(self.saprime_samples[self.sa_to_index(state, action)]) > self.num_samples:
        #    _ = self.saprime_samples[self.sa_to_index(state, action)].pop(0)
        if len(self.sa_samples):
            self.offline_update(10)

    def exponential(self, x_list, mu):
        return np.array([(1 / mu) * np.exp(-x / mu) for x in x_list])

    def on_policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(4)
        else:
            q_values = np.zeros(4)
            for a in range(4):
                q_values[a] = self.return_qvalue(state, a)
            return np.argmax(q_values)

    def offline_update(self, num_samples):
        """
        현재로선 random policy만 반영하도록 되어있음
        random walk에 대한 학습을 더 잘하도록 uniform random sample
        """
        for k in range(num_samples):
            # sa_index = np.random.choice(np.unique(self.sa_samples))
            sa_index = np.random.choice(self.sa_samples)
            saprime_index = np.random.choice(self.saprime_samples[sa_index])
            # saprime_sample_index = np.min(np.where(np.random.random() < self.cdf)[0])
            # if saprime_sample_index >= len(self.saprime_samples[sa_index]):
            #    saprime_sample_index = len(self.saprime_samples[sa_index]) - 1
            # saprime_index = self.saprime_samples[sa_index][
            #    len(self.saprime_samples[sa_index]) - saprime_sample_index - 1
            # ]
            state, action = self.index_to_sa(sa_index)
            next_state, next_action = self.index_to_sa(saprime_index)
            # next_action = self.on_policy(next_state)
            self.update_map(state, action, next_state, next_action)

    def return_map(self):
        """
        선택 가능한 action에 대해서만 나누도록 수정
        """
        sr_temp = np.zeros((self.num_states * self.num_actions, self.num_states))
        for state in range(self.num_states):
            for action in range(self.num_actions):
                index = self.sa_to_index(state, action)
                for next_state in range(self.num_states):
                    sum_per_state = 0
                    action_count = 0
                    for next_action in range(self.num_actions):
                        next_index = self.sa_to_index(next_state, next_action)
                        if self.sr[index, next_index] > 0:
                            sum_per_state += self.sr[index, next_index]
                            action_count += 1
                    if action_count:
                        sr_temp[index, next_state] = sum_per_state / action_count
        sr = np.zeros((self.num_states, self.num_states))
        for state in range(self.num_states):
            for next_state in range(self.num_states):
                sum_per_state = 0
                action_count = 0
                for action in range(self.num_actions):
                    index = self.sa_to_index(state, action)
                    if sr_temp[index, next_state] > 0:
                        sum_per_state += sr_temp[index, next_state]
                        action_count = +1
                if action_count:
                    sr[state, next_state] = sum_per_state / action_count
        return sr

    def save_map(self, directory):
        np.save(directory + "srdyna_" + self.index + ".npy", self.sr)
        np.save(directory + "w_" + self.index + ".npy", self.w)
        with open(directory + "sa_samples_" + self.index + ".list", "wb") as f:
            pickle.dump(self.sa_samples, f)
        with open(directory + "saprime_samples_" + self.index + ".list", "wb") as f:
            pickle.dump(self.saprime_samples, f)
        with open(directory + "reward_samples_" + self.index + ".list", "wb") as f:
            pickle.dump(self.reward_samples, f)

    def load_map(self, directory):
        if self.index is None:
            raise ValueError("Index not set")
        self.sr = np.load(directory + "srdyna_" + self.index + ".npy")
        self.w = np.load(directory + "w_" + self.index + ".npy")
        self.sa_samples = pickle.load(
            open(directory + "sa_samples_" + self.index + ".list", "rb")
        )
        self.saprime_samples = pickle.load(
            open(directory + "saprime_samples_" + self.index + ".list", "rb")
        )
        self.reward_samples = pickle.load(
            open(directory + "reward_samples_" + self.index + ".list", "rb")
        )

    def set_index(self, index):
        self.index = str(index)
