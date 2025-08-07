import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
import copy
from modules.base import SR, SRMB
import os
import networkx as nx
from utils.ot import compute_gw_distance_graph
from utils.topology import avg_nodewise
from utils.skeletonize import (
    preprocess_image,
    skeletonize,
    merge_3x3_to_1x1,
    map_pixels_to_skeleton_optimized,
    create_graph_from_boolean_matrix,
    simplify_graph_traverse,
    segment_nodes,
)
from modules.outer_loop import TransformerClassifier, transform_corner
import torch
from itertools import combinations
from networkx.algorithms import isomorphism
import math
from itertools import permutations, product
from collections import defaultdict, deque
from scipy.linalg import expm

EPSILON = 1e-10


class TAGNetwork(SR):
    def __init__(self, num_row, num_column, gamma=0.995, seed=0):
        super(TAGNetwork, self).__init__(num_row, num_column)
        np.random.seed(seed)
        self.gamma = gamma
        self.t_threshold = 0.15
        self.experienced_transitions = None
        self.initialize_experienced_transitions()
        self.env_border_transitions = self.return_border_transitions()
        left_singvecs, right_singvecs, singvals = self.decompose_map_svd()
        self.G = left_singvecs.T
        self.G_inv = right_singvecs
        self.lambdas = singvals
        self.G_unexplored = None
        self.G_inv_unexplored = None
        self.lambdas_unexplored = None
        self.one_step_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.two_step_deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # LU, RU, LD, RD
        self.update_G_unexplored()
        self.b_per_v = [(0, 2), (1, 2), (0, 3), (1, 3)]
        self.v_per_b = [[0, 2], [1, 3], [0, 1], [2, 3]]
        self.offset_per_v = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.next_a_drctn = {
            # a, drctn, row/column
            (0, -1, 0): [(2, -1), (0, 1)],
            (0, 1, 0): [(0, -1), (2, 1)],
            (1, -1, 0): [(3, -1), (1, 1)],
            (1, 1, 0): [(1, -1), (3, 1)],
            (2, -1, 0): [(0, -1), (2, 1)],
            (2, 1, 0): [(2, -1), (0, 1)],
            (3, -1, 0): [(1, -1), (3, 1)],
            (3, 1, 0): [(3, -1), (1, 1)],
            (0, -1, 1): [(1, -1), (0, 1)],
            (0, 1, 1): [(0, -1), (1, 1)],
            (1, -1, 1): [(0, -1), (1, 1)],
            (1, 1, 1): [(1, -1), (0, 1)],
            (2, -1, 1): [(3, -1), (2, 1)],
            (2, 1, 1): [(2, -1), (3, 1)],
            (3, -1, 1): [(2, -1), (3, 1)],
            (3, 1, 1): [(3, -1), (2, 1)],
        }
        self.vertex_nodes = None
        self.deadend_nodes = None
        self.edge_nodes = None
        self.vertex_corresp = None
        self.deadend_corresp = None
        self.edge_corresp = None
        self.skeleton_graph = None
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.topology_classifier = TransformerClassifier(
                input_dim=32, output_dim=4, num_layers=6, nhead=8
            ).to(
                self.device
            )  # pretrained classifier
            self.topology_classifier.load_state_dict(
                torch.load("data/outer_loop/model_weights_dict_v_6_8_0.001_10_cpu.pth")
            )
            self.topology_classifier.to(self.device)
            self.topology_classifier.eval()
        except FileNotFoundError:
            print("Topology classifier not loaded")
            self.topology_classifier = None

    def return_skeleton_diffusionmap(self):
        self.skeletonize_map()
        diffusionmap = self.return_successormap()
        return avg_nodewise(diffusionmap, self.vertex_nodes, self.deadend_nodes)

    def update_G_unexplored(self):
        T_unexplored = np.zeros((self.num_states, self.num_states))
        structure_head_border_stack_cache = self.structure_head_border_stack()
        for s in range(self.num_states):
            for a in range(4):
                dr, dc = self.one_step_deltas[a]
                new_r, new_c = self.state_to_index(s)
                if 0 <= new_r + dr < self.num_row and 0 <= new_c + dc < self.num_column:
                    new_s = self.index_to_state(new_r + dr, new_c + dc)
                    if (
                        structure_head_border_stack_cache[s, new_s] >= self.t_threshold
                        or not self.experienced_transitions[s, a]
                    ):
                        T_unexplored[s, new_s] = 1.0
        row_sums = T_unexplored.sum(axis=1)
        non_zero_mask = row_sums != 0
        T_unexplored_normalized = np.zeros_like(T_unexplored)
        T_unexplored_normalized[non_zero_mask] = (
            T_unexplored[non_zero_mask] / row_sums[non_zero_mask][:, np.newaxis]
        )
        T_unexplored = T_unexplored_normalized
        U, S, Vh = np.linalg.svd(np.eye(self.num_states) - self.gamma * T_unexplored)
        self.G_unexplored = Vh.T
        self.G_inv_unexplored = U.T
        self.lambdas_unexplored = 1 / S

    def calculate_actionable_transitions(self):
        T = self.structure_head_border_stack()
        _, borders_exclude_unexplored = self.structure_head_corner_stack(
            include_unexplored=False
        )
        actionable_transitions = (1 - borders_exclude_unexplored).astype(bool)
        for i in range(T.shape[0]):
            if np.sum(T[i]) < EPSILON:
                actionable_transitions[i] = False
        return actionable_transitions

    def find_explorable_states(self):
        actionable_transitions = self.calculate_actionable_transitions()
        experienced_transitions = self.experienced_transitions
        num_states = experienced_transitions.shape[0]
        result_array = np.zeros(num_states)  # Initialize the array with zeros
        for state in range(num_states):
            if actionable_transitions[state].any() or (
                np.sum(self.sr[state]) > 1 + EPSILON
                and experienced_transitions[state].any()
            ):  # Condition 1: Any action is possible or (the state is not a block and any transition is experienced)
                result_array[state] = 1
            elif not experienced_transitions[
                state
            ].all():  # Condition 2: Any element in the experienced transition is False
                result_array[state] = 0.5
            elif (experienced_transitions + self.env_border_transitions)[
                state
            ].all() and not actionable_transitions[
                state
            ].any():  # Condition 3: All directions are blocked
                result_array[state] = 0
            else:  # There should be no such case
                raise ValueError(
                    f"{state}, {experienced_transitions[state]}, {actionable_transitions[state]}"
                )
        return result_array

    def initialize_experienced_transitions(self):
        self.experienced_transitions = np.zeros(
            (self.num_states, 4), dtype=bool
        )  # 1이면 transition을 경험한 것

    def return_border_transitions(self):
        env_border_transitions = np.zeros((self.num_states, 4), dtype=bool)
        for r in range(self.num_row):
            env_border_transitions[self.index_to_state(r, 0), 0] = True
            env_border_transitions[self.index_to_state(r, self.num_column - 1), 1] = (
                True
            )
        for c in range(self.num_column):
            env_border_transitions[self.index_to_state(0, c), 2] = True
            env_border_transitions[self.index_to_state(self.num_row - 1, c), 3] = True
        return env_border_transitions

    def generate_skeleton_image(self, include_unexplored=False):
        image_original = np.ones((self.num_row * 3, self.num_column * 3))
        for r in range(self.num_row * 3):
            image_original[r, 0] = 0
            image_original[r, self.num_column * 3 - 1]
        for c in range(self.num_column * 3):
            image_original[0, c] = 0
            image_original[self.num_row * 3 - 1, c] = 0
        explorable_states = self.find_explorable_states()
        explorable_states_threshold = 0.3 if include_unexplored else 0.7
        corners, borders = self.structure_head_corner_stack(include_unexplored)
        for s in range(self.num_states):
            r, c = self.state_to_index(s)
            if explorable_states[s] < explorable_states_threshold:  # if block
                for dr in range(3):
                    for dc in range(3):
                        image_original[3 * r + dr, 3 * c + dc] = 0
            else:  # if not block
                for a in range(4):
                    if borders[s, a]:
                        if a == 0:
                            for dr in range(3):
                                image_original[3 * r + dr, 3 * c] = 0
                        elif a == 1:
                            for dr in range(3):
                                image_original[3 * r + dr, 3 * c + 2] = 0
                        elif a == 2:
                            for dc in range(3):
                                image_original[3 * r, 3 * c + dc] = 0
                        else:
                            for dc in range(3):
                                image_original[3 * r + 2, 3 * c + dc] = 0
                    if corners[s, a] < 0:
                        if a == 0:
                            image_original[3 * r, 3 * c] = 0
                        elif a == 1:
                            image_original[3 * r, 3 * c + 2] = 0
                        elif a == 2:
                            image_original[3 * r + 2, 3 * c] = 0
                        else:
                            image_original[3 * r + 2, 3 * c + 2] = 0
        image = preprocess_image(image_original)
        skeleton = skeletonize(image)
        return image_original, image, skeleton

    def skeletonize_map(self, preserved_states=None):  # ground-truth skeletonization
        image_original, image, skeleton = self.generate_skeleton_image(
            include_unexplored=False
        )
        image_1x1 = merge_3x3_to_1x1(image_original)
        if (1 - image_1x1).astype(bool).all():
            self.vertex_nodes = []
            self.deadend_nodes = []
            self.edge_nodes = []
            self.vertex_corresp = []
            self.deadend_corresp = []
            self.edge_corresp = []
            self.skeleton_graph = nx.MultiGraph()
        else:
            closest_skeleton_indices = map_pixels_to_skeleton_optimized(
                image_1x1, skeleton
            )
            G = create_graph_from_boolean_matrix(skeleton)
            preserved_nodes = []
            closest_preserved_states = {}
            if preserved_states is not None:
                for s in preserved_states:
                    r, c = self.state_to_index(s)
                    matching_keys = [
                        key
                        for key, value in closest_skeleton_indices.items()
                        if (r, c) in value
                    ]
                    if len(matching_keys) == 1:
                        preserved_nodes.append(matching_keys[0])
                    elif len(matching_keys) > 1:
                        """Matching key 개수가 여러 개인 경우, degree가 가장 큰 node를 선택"""
                        degrees = [G.degree[key] for key in matching_keys]
                        max_degree = np.max(degrees)
                        max_elements = [
                            matching_keys[i]
                            for i in range(len(matching_keys))
                            if degrees[i] == max_degree
                        ]
                        if len(max_elements) == 1:
                            preserved_nodes.append(max_elements[0])
                        else:
                            """Degree가 가장 큰 node가 여러 개인 경우, 가장 matching되는 state가 많은 node를 선택"""
                            max_matching = np.max(
                                [
                                    len(closest_skeleton_indices[key])
                                    for key in max_elements
                                ]
                            )
                            max_matching_elements = [
                                max_elements[i]
                                for i in range(len(max_elements))
                                if len(closest_skeleton_indices[max_elements[i]])
                                == max_matching
                            ]
                            preserved_nodes.append(max_matching_elements[0])
                    else:
                        raise ValueError("No matching skeletal nodes found")
                    closest_preserved_states[s] = (
                        preserved_nodes[-1][0] // 3,
                        preserved_nodes[-1][1] // 3,
                    )
            else:
                preserved_states = []
            assert len(preserved_nodes) == len(preserved_states)
            simplified_G, unskeletonized_edges_dict, unskeletonized_nodes_dict = (
                simplify_graph_traverse(G, closest_skeleton_indices, preserved_nodes)
            )
            assert np.array(
                [s in simplified_G.nodes for s in closest_preserved_states.values()]
            ).all()
            total_nodes = []
            for edges in unskeletonized_edges_dict.values():
                for edge in edges:
                    total_nodes.extend(edge)
            for nodes in unskeletonized_nodes_dict.values():
                total_nodes.extend(nodes)

            (
                vertex_nodes,
                deadend_nodes,
                edge_nodes,
                vertex_corresp,
                deadend_corresp,
                edge_corresp,
            ) = segment_nodes(
                simplified_G,
                unskeletonized_nodes_dict,
                unskeletonized_edges_dict,
                num_column=image.shape[1] // 3,
            )

            # Create a new MultiGraph
            simplified_G_onethird = nx.MultiGraph()

            # Add all nodes from the original graph
            simplified_G_onethird.add_nodes_from(simplified_G.nodes(data=True))

            # Add all edges with weights divided by 3
            for u, v, key, data in simplified_G.edges(data=True, keys=True):
                # Get the original weight (default to 1 if not present)
                original_weight = data.get("weight", 1)
                # Divide the weight by 3
                new_weight = original_weight / 3
                # Add the edge to the new graph with the updated weight
                simplified_G_onethird.add_edge(
                    u, v, key=key, **{**data, "weight": new_weight}
                )

            self.vertex_nodes = vertex_nodes
            self.deadend_nodes = deadend_nodes
            self.edge_nodes = edge_nodes
            self.vertex_corresp = vertex_corresp
            self.deadend_corresp = deadend_corresp
            self.edge_corresp = edge_corresp
            self.skeleton_graph = simplified_G_onethird
        if preserved_states is not None:
            return closest_preserved_states

    def skeletonize_map_biologically(
        self, return_attn=False
    ):  # biological skeletonization using a Transformer encoder
        if self.topology_classifier is None:
            return None
        else:
            corners, _ = self.structure_head_corner_stack(include_unexplored=False)
            corners = (
                torch.tensor(corners.reshape(1, 10, 10, -1)).float().to(self.device)
            )
            transformed_corners = transform_corner(corners)
            blocks = (
                torch.tensor(
                    np.array(
                        [
                            (
                                1
                                if np.sum(self.structure_head_border_stack()[s]) < 0.1
                                else 0
                            )
                            for s in range(self.num_states)
                        ]
                    )
                )
                .float()
                .to(self.device)
                .reshape(1, 10, 10, 1)
            )
            if return_attn:
                output, attn = self.topology_classifier(
                    transformed_corners,
                    torch.zeros(1, 10, 10, 4).float().to(self.device),
                    torch.zeros(1, 10, 10, 100).float().to(self.device),
                    torch.zeros(1, 10, 10, 9).float().to(self.device),
                    blocks,
                    return_attn,
                )
                _, preds = torch.max(output, dim=-1)
                return preds.cpu().numpy().reshape(100), attn
            else:
                with torch.no_grad():
                    _, preds = torch.max(
                        self.topology_classifier(
                            transformed_corners,
                            torch.zeros(1, 10, 10, 4).float().to(self.device),
                            torch.zeros(1, 10, 10, 100).float().to(self.device),
                            torch.zeros(1, 10, 10, 9).float().to(self.device),
                            blocks,
                            return_attn,
                        ),
                        dim=-1,
                    )
                return preds.cpu().numpy().reshape(100)

    def policy_head_place_stack(self):
        return self.G @ np.diag(self.lambdas) @ self.G_inv

    def structure_head_border_stack(self):
        return (
            np.identity(self.num_states)
            - self.G_inv.T @ np.diag(self.lambdas**-1) @ self.G.T
        ) / self.gamma

    def structure_head_corner_stack(self, include_unexplored=True):
        stack_1_output = self.structure_head_border_stack()
        stack_1_output = self.structure_1_output_postprocessing(
            stack_1_output, include_unexplored
        )
        skip_connect = stack_1_output[:, np.newaxis, :]
        one_step_t = self.structure_2_skip_connect_linear_layer(skip_connect)
        borders = np.multiply(
            1 - one_step_t,
            np.sum(stack_1_output, axis=1, keepdims=True) >= self.t_threshold,
        )  # Border cell

        self_attn_output = (np.matmul(stack_1_output, stack_1_output))[:, np.newaxis, :]
        two_step_t = self.structure_2_value_layer(self_attn_output)

        one_indices = np.array(
            [
                np.eye(4)[:, [0, 2]],
                np.eye(4)[:, [1, 2]],
                np.eye(4)[:, [0, 3]],
                np.eye(4)[:, [1, 3]],
            ]
        )
        one_step_t_rep = np.repeat(one_step_t[np.newaxis, :, :], 4, axis=0)
        one_step_adj_t = np.matmul(one_step_t_rep, one_indices)
        one_step_result = (
            np.sum(one_step_adj_t, axis=2).T - 1
        ) 
        # one_step_result = xnor_layer(one_step_adj_t)
        two_step_result = not_layer(
            two_step_t
        )  
        combined_result = np.multiply(one_step_result, two_step_result)
        # combined_result = np.squeeze(and_layer(np.squeeze(np.stack((one_step_result, two_step_result), axis=2))).T) # n x 4 (num corners)
        corners = np.multiply(
            combined_result,
            np.sum(stack_1_output, axis=1, keepdims=True) >= self.t_threshold,
        )
        return corners, borders

    def structure_1_output_postprocessing(self, x, include_unexplored):
        x = x >= self.t_threshold
        if not include_unexplored:
            return np.array(x)
        x_masked = np.zeros((self.num_states, self.num_states), dtype=bool)
        for s in range(self.num_states):
            r, c = self.state_to_index(s)
            for a in range(4):
                dr, dc = self.one_step_deltas[a]
                if 0 <= r + dr < self.num_row and 0 <= c + dc < self.num_column:
                    new_s = self.index_to_state(r + dr, c + dc)
                    if self.experienced_transitions[s, a]:
                        x_masked[s, new_s] = x[s, new_s]
                    else:
                        x_masked[s, new_s] = True
        return np.array(
            x_masked
        )  # exploration mask가 0이면 기본 transition rule대로 따르도록

    def structure_2_skip_connect_linear_layer(self, x):
        skip_linears = []
        for s in range(self.num_states):
            skip_linear = []
            r, c = self.state_to_index(s)
            for dr, dc in self.one_step_deltas:
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < self.num_row and 0 <= new_c < self.num_column:
                    new_s = self.index_to_state(new_r, new_c)
                    skip_linear.append(
                        [1 if x == new_s else 0 for x in range(self.num_states)]
                    )
                else:
                    skip_linear.append([0 for x in range(self.num_states)])
            skip_linears.append(np.array(skip_linear).T)
        skip_linears = np.array(skip_linears)
        return np.squeeze(np.matmul(x, skip_linears))

    def structure_2_value_layer(self, x):
        layer_values = []
        for s in range(self.num_states):
            layer_value = []
            r, c = self.state_to_index(s)
            for dr, dc in self.two_step_deltas:
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < self.num_row and 0 <= new_c < self.num_column:
                    new_s = self.index_to_state(new_r, new_c)
                    layer_value.append(
                        [1 if x == new_s else 0 for x in range(self.num_states)]
                    )
                else:
                    layer_value.append([0 for x in range(self.num_states)])
            layer_values.append(np.array(layer_value).T)
        layer_values = np.array(layer_values)
        return np.squeeze(np.matmul(x, layer_values))

    def save_map(self, directory):
        np.save(directory + "pbc_" + self.index + ".npy", self.sr)
        np.save(
            directory + "experienced_transitions_" + self.index + ".npy",
            self.experienced_transitions,
        )

    def load_map(self, directory, load_pbc_exclusive=False):
        if load_pbc_exclusive:
            self.sr = np.load(directory + "pbc_" + self.index + ".npy")
            if os.path.exists(
                directory + "experienced_transitions_" + self.index + ".npy"
            ):
                self.experienced_transitions = np.load(
                    directory + "experienced_transitions_" + self.index + ".npy"
                )
            else:
                self.set_experienced_transitions_to_true()
        else:
            self.sr = np.load(directory + "sr_" + self.index + ".npy")
            self.set_experienced_transitions_to_true()
        self.update_grid()

    def set_experienced_transitions_to_true(self):
        self.initialize_experienced_transitions()
        for s in range(self.num_states):
            r, c = self.state_to_index(s)
            for a in range(4):
                dr, dc = self.one_step_deltas[a]
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < self.num_row and 0 <= new_c < self.num_column:
                    self.experienced_transitions[s, a] = True

    def set_map(self, new_sr, row_indices=None, column_indices=None):
        super().set_map(new_sr, row_indices, column_indices)
        self.update_grid()
        self.set_experienced_transitions_to_true()

    def update_grid(self):
        left_singvecs, right_singvecs, singvals = self.decompose_map_svd()
        self.G = left_singvecs.T
        self.G_inv = right_singvecs
        self.lambdas = singvals

    def visualize_corner(self, directory=None, blocks=None, display=False):
        corners, _ = self.structure_head_corner_stack()
        vis = np.zeros((self.num_row * 2, self.num_column * 2))
        colorval = [3, 1, 5, 7]
        s_padding = [(0, 0), (0, 1), (1, 0), (1, 1)]
        hsv = matplotlib.colormaps["hsv"].resampled(9)
        new_cmap = hsv(range(9))
        black = [0.0, 0.0, 0.0, 1.0]
        white = [1.0, 1.0, 1.0, 1.0]
        new_cmap[0] = white
        new_cmap[8] = black
        new_cmap = colors.ListedColormap(new_cmap)
        for s in range(self.num_states):
            if np.sum(np.abs(corners[s])):
                r, c = self.state_to_index(s)
                for a in range(4):
                    if corners[s, a]:
                        dr, dc = s_padding[a]
                        vis[r * 2 + dr, c * 2 + dc] = colorval[a]
            else:
                r, c = self.state_to_index(s)
                if blocks is None:
                    vis[2 * r, 2 * c] = np.nan
                    vis[2 * r + 1, 2 * c] = np.nan
                    vis[2 * r, 2 * c + 1] = np.nan
                    vis[2 * r + 1, 2 * c + 1] = np.nan
                else:
                    if s in blocks:
                        vis[2 * r, 2 * c] = 8
                        vis[2 * r + 1, 2 * c] = 8
                        vis[2 * r, 2 * c + 1] = 8
                        vis[2 * r + 1, 2 * c + 1] = 8
        plt.matshow(vis, cmap=new_cmap, vmin=0, vmax=8)
        plt.tick_params(
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
            bottom=False,
            top=False,
            labeltop=False,
        )
        plt.grid(False)
        if display:
            plt.show()
        else:
            if directory is None:
                plt.savefig("display/corner_{}.png".format(self.index))
            else:
                plt.savefig(directory + "corner_{}.png".format(self.index))
            plt.close("all")

    def visualize_border(self, directory=None, blocks=None, display=False):
        _, borders = self.structure_head_corner_stack()
        vis = np.ones((self.num_row * 2, self.num_column * 2))
        s_paddings = [
            [(0, 0), (1, 0)],
            [(0, 1), (1, 1)],
            [(0, 0), (0, 1)],
            [(1, 0), (1, 1)],
        ]
        colorval = [4, 0, 2, 6]
        hsv = matplotlib.colormaps["hsv"].resampled(9)
        new_cmap = hsv(range(9))
        black = [0.0, 0.0, 0.0, 1.0]
        white = [1.0, 1.0, 1.0, 1.0]
        new_cmap[1] = white
        new_cmap[8] = black
        new_cmap = colors.ListedColormap(new_cmap)
        for s in range(self.num_states):
            if np.sum(np.abs(borders[s])):
                r, c = self.state_to_index(s)
                for a in range(4):
                    if borders[s, a]:
                        s_padding = s_paddings[a]
                        for dr, dc in s_padding:
                            if vis[r * 2 + dr, c * 2 + dc] != 1:
                                vis[r * 2 + dr, c * 2 + dc] = np.nan
                            else:
                                vis[r * 2 + dr, c * 2 + dc] = colorval[a]
            else:
                r, c = self.state_to_index(s)
                if blocks is None:
                    vis[2 * r, 2 * c] = np.nan
                    vis[2 * r + 1, 2 * c] = np.nan
                    vis[2 * r, 2 * c + 1] = np.nan
                    vis[2 * r + 1, 2 * c + 1] = np.nan
                else:
                    if s in blocks:
                        vis[2 * r, 2 * c] = 8
                        vis[2 * r + 1, 2 * c] = 8
                        vis[2 * r, 2 * c + 1] = 8
                        vis[2 * r + 1, 2 * c + 1] = 8
        plt.matshow(vis, cmap=new_cmap, vmin=0, vmax=8)
        plt.tick_params(
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
            bottom=False,
            top=False,
            labeltop=False,
        )
        plt.grid(False)
        if display:
            plt.show()
        else:
            if directory is None:
                plt.savefig("display/border_{}.png".format(self.index))
            else:
                plt.savefig(directory + "border_{}.png".format(self.index))
            plt.close("all")


class TAGUpdate(TAGNetwork):
    def __init__(self, num_row, num_column, gamma=0.995, seed=0):
        super(TAGUpdate, self).__init__(num_row, num_column, gamma, seed)
        self.reverse_action = {0: 1, 1: 0, 2: 3, 3: 2}

    def reset_map(self):
        self.sr = np.identity(self.num_states)
        self.update_grid()

    def woodbury_update(self, state, prob_diff):
        self.sr = (
            self.sr
            - 1
            / (1 + np.dot(self.gamma * prob_diff, self.sr[:, state]))
            * self.sr[:, state].reshape(-1, 1)
            @ (self.gamma * prob_diff.reshape(1, -1))
            @ self.sr
        )

    def update_map(self, current_state, action, next_state, bidirection=True):
        if self.sr[current_state][current_state] == 1:
            self.sr[current_state][current_state] = 1 / (1 - self.gamma)
        if not self.experienced_transitions[current_state, action]:
            self.experienced_transitions[current_state, action] = True
        if current_state != next_state:
            if (
                self.structure_head_border_stack()[current_state, next_state]
                < self.t_threshold
            ):
                """
                처음에 모든 transition이 불가능한 것으로 initialize 되어있다는 가정임
                """
                current_legal_states_hot = (
                    self.structure_head_border_stack()[current_state]
                    >= self.t_threshold
                )  # step function
                if bidirection:
                    if not self.experienced_transitions[
                        next_state, self.reverse_action[action]
                    ]:
                        self.experienced_transitions[
                            next_state, self.reverse_action[action]
                        ] = True
                    next_legal_states_hot = (
                        self.structure_head_border_stack()[next_state]
                        >= self.t_threshold
                    )  # step function
                    if self.sr[next_state][next_state] == 1:
                        self.sr[next_state][next_state] = 1 / (1 - self.gamma)

                if 0 < np.sum(current_legal_states_hot) <= 3:
                    current_original_rw_prob = current_legal_states_hot / np.sum(
                        current_legal_states_hot
                    )  # softmax
                    current_legal_states_hot[next_state] = True
                    current_new_rw_prob = current_legal_states_hot / np.sum(
                        current_legal_states_hot
                    )  # softmax
                    current_prob_diff = current_original_rw_prob - current_new_rw_prob
                    self.woodbury_update(current_state, current_prob_diff)
                    if bidirection:
                        if np.sum(next_legal_states_hot) == 0:
                            next_prob_diff = np.zeros(self.num_states)
                            next_prob_diff[current_state] = -1
                            next_prob_diff[next_state] = 1
                        else:
                            next_original_rw_prob = next_legal_states_hot / np.sum(
                                next_legal_states_hot
                            )
                            if next_legal_states_hot[next_state]:
                                next_legal_states_hot[next_state] = False
                            next_legal_states_hot[current_state] = True
                            next_new_rw_prob = next_legal_states_hot / np.sum(
                                next_legal_states_hot
                            )
                            next_prob_diff = next_original_rw_prob - next_new_rw_prob
                        self.woodbury_update(next_state, next_prob_diff)
                elif np.sum(current_legal_states_hot) == 0:
                    current_prob_diff = np.zeros(self.num_states)
                    current_prob_diff[current_state] = 1
                    current_prob_diff[next_state] = -1
                    self.woodbury_update(current_state, current_prob_diff)
                    if bidirection:
                        next_prob_diff = -current_prob_diff
                        self.woodbury_update(next_state, next_prob_diff)
                else:
                    raise ValueError()

                left_singvecs, right_singvecs, singvals = self.decompose_map_svd()
                self.G = left_singvecs.T
                self.G_inv = right_singvecs
                self.lambdas = singvals

        else:
            if not self.experienced_transitions[current_state, action]:
                self.experienced_transitions[current_state, action] = True
            current_row, current_column = self.state_to_index(current_state)
            if (0 <= current_row + self.one_step_deltas[action][0] < self.num_row) and (
                0 <= current_column + self.one_step_deltas[action][1] < self.num_column
            ):
                block_state = self.index_to_state(
                    current_row + self.one_step_deltas[action][0],
                    current_column + self.one_step_deltas[action][1],
                )
                if bidirection:
                    """
                    block에서 여기로 오는 transition도 경험한 것으로 처리
                    """
                    if not self.experienced_transitions[
                        block_state, self.reverse_action[action]
                    ]:
                        self.experienced_transitions[
                            block_state, self.reverse_action[action]
                        ] = True
                if (
                    self.structure_head_border_stack()[current_state, block_state]
                    >= self.t_threshold
                ):
                    block_legal_states_hot = (
                        self.structure_head_border_stack()[block_state]
                        >= self.t_threshold
                    )
                    block_original_rw_prob = block_legal_states_hot / np.sum(
                        block_legal_states_hot
                    )
                    block_legal_states_hot[current_state] = False
                    if np.sum(block_legal_states_hot):
                        block_new_rw_prob = block_legal_states_hot / np.sum(
                            block_legal_states_hot
                        )
                        block_connection_lost = False
                    else:
                        block_new_rw_prob = np.zeros(self.num_states)
                        block_connection_lost = True
                    block_prob_diff = block_original_rw_prob - block_new_rw_prob
                    self.woodbury_update(block_state, block_prob_diff)
                    if block_connection_lost:
                        self.sr[block_state] = np.array(
                            [
                                1.0 if x == block_state else 0.0
                                for x in range(self.num_states)
                            ]
                        )
                    if bidirection:
                        curr_legal_states_hot = (
                            self.structure_head_border_stack()[current_state]
                            >= self.t_threshold
                        )
                        curr_original_rw_prob = curr_legal_states_hot / np.sum(
                            curr_legal_states_hot
                        )
                        curr_legal_states_hot[block_state] = False
                        if np.sum(curr_legal_states_hot):
                            curr_new_rw_prob = curr_legal_states_hot / np.sum(
                                curr_legal_states_hot
                            )
                            curr_connection_lost = False
                        else:
                            curr_new_rw_prob = np.zeros(self.num_states)
                            curr_connection_lost = True
                        curr_prob_diff = curr_original_rw_prob - curr_new_rw_prob
                        self.woodbury_update(current_state, curr_prob_diff)
                        if curr_connection_lost:
                            self.sr[current_state] = np.array(
                                [
                                    1.0 if x == current_state else 0.0
                                    for x in range(self.num_states)
                                ]
                            )

                    left_singvecs, right_singvecs, singvals = self.decompose_map_svd()
                    self.G = left_singvecs.T
                    self.G_inv = right_singvecs
                    self.lambdas = singvals
        """
        접근 불가능한 transition은 경험한 것으로 처리
        """
        self.update_G_unexplored()
        place_codes = (
            self.G_unexplored @ np.diag(self.lambdas_unexplored) @ self.G_inv_unexplored
        )[current_state]
        for s in range(self.num_states):
            if place_codes[s] < EPSILON:
                for a in range(4):
                    if not self.experienced_transitions[s, a]:
                        self.experienced_transitions[s, a] = True
        self.update_G_unexplored()


class TAGTransfer(TAGUpdate):
    def __init__(self, num_row, num_column, gamma=0.995, seed=0):
        super(TAGTransfer, self).__init__(num_row, num_column, gamma, seed)
        self.block_chunks = []
        self.ego_color = []  # L, LD, D, RD, R, RU, U, LU
        self.topological_prior = None
        self.exploration_target_transitions = np.zeros(
            (self.num_states, 4, self.num_states), dtype=bool
        )
        self.transition_count = np.zeros(
            (self.num_states, 4)
        )  # Count visits for each state-action pair
        self.total_counts = np.zeros(self.num_states)  # Total counts for each state

    def update_map(self, current_state, action, next_state, bidirection=True):
        super().update_map(current_state, action, next_state, bidirection)
        self.transition_count[current_state][action] += 1
        self.total_counts[current_state] += 1
        if current_state != next_state:
            self.transition_count[next_state][self.reverse_action[action]] += 1
            self.total_counts[next_state] += 1
            if self.exploration_target_transitions[current_state, action, next_state]:
                self.exploration_target_transitions[
                    current_state, action, next_state
                ] = False
        else:
            if self.exploration_target_transitions[current_state, action, :].any():
                self.exploration_target_transitions[current_state, action, :] = False

    def set_topological_prior(self, topological_prior: nx.MultiGraph):
        self.topological_prior = topological_prior

    def find_subgraph_alignments(self, partial_graph):
        GM = isomorphism.GraphMatcher(self.topological_prior, partial_graph)
        subgraph_alignments = [mapping for mapping in GM.subgraph_isomorphisms_iter()]
        return subgraph_alignments

    def align_including_and_excluding_unexplored(
        self, G_including_unexplored, G_excluding_unexplored
    ):
        alignment = {}
        aligned_nodes = set()
        euclidean_distance = lambda point1, point2: math.sqrt(
            (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
        )

        while True:
            new_alignments = False
            for nodes_excluding in G_excluding_unexplored.nodes():
                if nodes_excluding in alignment:
                    continue

                if nodes_excluding in G_including_unexplored.nodes():
                    alignment[nodes_excluding] = nodes_excluding
                    aligned_nodes.add(nodes_excluding)
                    new_alignments = True
                else:
                    distances = [
                        (
                            nodes_including,
                            euclidean_distance(nodes_excluding, nodes_including),
                        )
                        for nodes_including in G_including_unexplored.nodes()
                        if nodes_including not in aligned_nodes
                    ]
                    if not distances:
                        continue

                    min_dist = min(distances, key=lambda x: x[1])
                    min_count = sum(1 for _, dist in distances if dist == min_dist[1])

                    if min_count > 1:
                        continue
                    else:
                        closest_node = min_dist[0]
                        if closest_node not in aligned_nodes:
                            alignment[nodes_excluding] = closest_node
                            aligned_nodes.add(closest_node)
                            new_alignments = True

            if not new_alignments:
                break

        # Handle remaining unaligned nodes
        for nodes_excluding in G_excluding_unexplored.nodes():
            if nodes_excluding not in alignment:
                distances = [
                    (
                        nodes_including,
                        euclidean_distance(nodes_excluding, nodes_including),
                    )
                    for nodes_including in G_including_unexplored.nodes()
                ]
                if distances:
                    closest_node = min(distances, key=lambda x: x[1])[0]
                    alignment[nodes_excluding] = closest_node

        return alignment

    def compare_including_vs_excluding_unexplored(self):
        _, _, skeletonize_including_unexplored = self.generate_skeleton_image(
            include_unexplored=True
        )
        image_original, _, skeletonize_excluding_unexplored = (
            self.generate_skeleton_image(include_unexplored=False)
        )
        G_including_unexplored = create_graph_from_boolean_matrix(
            skeletonize_including_unexplored
        )
        G_excluding_unexplored = create_graph_from_boolean_matrix(
            skeletonize_excluding_unexplored
        )
        alignment = self.align_including_and_excluding_unexplored(
            G_including_unexplored, G_excluding_unexplored
        )  # exploration 안한 영역을 포함 / 불포함한 skeleton을 align
        # Compare the number of edges of both aligned nodes
        mismatches_over = []
        mismatches_under = []
        for node_excluding, node_including in alignment.items():
            degree_excluding = G_excluding_unexplored.degree(node_excluding)
            degree_including = G_including_unexplored.degree(node_including)
            if (
                degree_excluding < degree_including
            ):  # 포함했을 때 edge가 더 많은 경우와 적은 경우를 나누어서 해당하는 노드와 degree를 저장
                mismatches_over.append((node_excluding, degree_excluding))
            if degree_excluding > degree_including:
                mismatches_under.append((node_excluding, degree_excluding))
        return (
            mismatches_over,
            mismatches_under,
            G_excluding_unexplored,
            image_original,
            skeletonize_excluding_unexplored,
        )

    def find_plausible_alignments(self):
        (
            mismatches_over,
            mismatches_under,
            G_excluding_unexplored,
            image_original,
            skeleton,
        ) = self.compare_including_vs_excluding_unexplored()
        image_1x1 = merge_3x3_to_1x1(image_original)
        closest_skeleton_indices = map_pixels_to_skeleton_optimized(image_1x1, skeleton)

        edge_mismatches_over = []  # mismatch node들 후처리
        vertex_mismatches_over = []
        edge_mismatches_over_1x1 = []
        vertex_mismatches_over_1x1 = []
        for node_excluding, degree_excluding in mismatches_over:
            if degree_excluding == 2:
                if node_excluding not in edge_mismatches_over:
                    edge_mismatches_over.append(node_excluding)
                    edge_mismatches_over_1x1.append(
                        (
                            node_excluding[0] // 3,
                            node_excluding[1] // 3,
                        )
                    )
            else:
                if node_excluding not in vertex_mismatches_over:
                    vertex_mismatches_over.append(node_excluding)
                    vertex_mismatches_over_1x1.append(
                        (
                            node_excluding[0] // 3,
                            node_excluding[1] // 3,
                        )
                    )
        edge_mismatches_under = []
        vertex_mismatches_under = []
        edge_mismatches_under_1x1 = []
        vertex_mismatches_under_1x1 = []
        for node_excluding, degree_excluding in mismatches_under:
            if degree_excluding == 2:
                if node_excluding not in edge_mismatches_under:
                    edge_mismatches_under.append(node_excluding)
                    edge_mismatches_under_1x1.append(
                        (
                            node_excluding[0] // 3,
                            node_excluding[1] // 3,
                        )
                    )
            else:
                if node_excluding not in vertex_mismatches_under:
                    vertex_mismatches_under.append(node_excluding)
                    vertex_mismatches_under_1x1.append(
                        (
                            node_excluding[0] // 3,
                            node_excluding[1] // 3,
                        )
                    )

        all_subsets = []
        for r in range(
            np.min((len(edge_mismatches_over) + 1, 3))
        ):  # edge가 더 발생할 가능성이 있는 곳들에 대해서 combination
            subsets = list(combinations(edge_mismatches_over, r))
            all_subsets.extend(subsets)

        valid_alignments = []
        corresponding_subgraphs = []

        for (
            subset
        ) in all_subsets:  # edge가 더 발생할 node 선택 경우의 수 한 가지씩 처리
            G_copy = copy.deepcopy(G_excluding_unexplored)
            simplified_G, _, _ = simplify_graph_traverse(
                G_copy, closest_skeleton_indices, list(subset)
            )  # edge가 더 발생할 node를 남기고 graph contraction (unexplored 제외 graph에서)
            if simplified_G is None:
                raise ValueError(f"None simplified_G at {self.index}")
            if not nx.is_connected(simplified_G):
                raise ValueError("simplified_G is not connected")
            alignments_subset = self.find_subgraph_alignments(
                simplified_G
            )  # topological prior와의 alignment를 계산
            if len(alignments_subset):  # 기존에서 확장을 하면 되는 경우
                for alignment in alignments_subset:
                    valid = True
                    for full_node, partial_node in alignment.items():
                        if self.topological_prior.degree(
                            full_node
                        ) != simplified_G.degree(
                            partial_node
                        ):  # prior와 subset의 degree가 다른 node가 기존 계산했던 mismatch에 포함되어야만 함
                            if (
                                partial_node not in edge_mismatches_over_1x1
                                and partial_node not in vertex_mismatches_over_1x1
                            ):
                                valid = False
                                break
                    if valid:
                        valid_alignments.append(alignment)
                        corresponding_subgraphs.append(simplified_G)
        return valid_alignments, corresponding_subgraphs, vertex_mismatches_under_1x1

    def extract_subgraph_from_alignment(self, alignment):
        """
        Extracts a subgraph from full_graph using the node alignment mapping.

        Parameters:
        - full_graph: The full nx.MultiGraph from which to extract the subgraph.
        - alignment: A dictionary where keys are nodes of the full_graph and values are the aligned subgraph nodes.

        Returns:
        - aligned_subgraph: The nx.MultiGraph formed by the nodes mapped by the alignment.
        """
        aligned_nodes = list(alignment.keys())
        aligned_subgraph = self.topological_prior.subgraph(
            aligned_nodes
        ).copy()  # Create the subgraph from the aligned nodes
        return aligned_subgraph

    def rank_subgraphs(self, valid_alignments, corresponding_subgraphs, alpha=0.5):

        gw_distances = []

        # Iterate over each valid alignment and its corresponding subgraph
        for alignment, corresponding_subgraph in zip(
            valid_alignments, corresponding_subgraphs
        ):
            # Step 1: Extract the aligned subgraph from the full skeleton graph
            aligned_subgraph = self.extract_subgraph_from_alignment(alignment)

            # Step 2: Calculate the GW distance between the aligned subgraph and the corresponding subgraph
            gw_distance, _ = compute_gw_distance_graph(
                aligned_subgraph, corresponding_subgraph, alpha=alpha
            )

            # Step 3: Store the result as a tuple (alignment, corresponding_subgraph, gw_distance)
            gw_distances.append((alignment, corresponding_subgraph, gw_distance))

        # Step 4: Sort the subgraphs by GW distance (ascending order)
        ranked_subgraphs = sorted(gw_distances, key=lambda x: x[2])

        return ranked_subgraphs

    def set_exploration_target(self, state):
        if self.topological_prior is None:
            pass
        else:
            self.skeletonize_map()
            if self.skeleton_graph.number_of_nodes() >= 3:
                GM = isomorphism.MultiGraphMatcher(
                    self.topological_prior, self.skeleton_graph
                )
                if not GM.is_isomorphic():

                    (
                        valid_alignments,
                        corresponding_subgraphs,
                        vertex_mismatches_under,
                    ) = self.find_plausible_alignments()
                    if (
                        len(valid_alignments) == 0
                    ):  # 너무 불필요한 vertex 가지가 많은 경우 (exploration target이 설정될 수 없는 경우)
                        corners, _ = self.structure_head_corner_stack(
                            include_unexplored=False
                        )
                        for (
                            vertex
                        ) in (
                            vertex_mismatches_under
                        ):  # 불필요한 가지를 제거해나가기 위해 그쪽으로 이동
                            min_dist = 1000
                            min_s = None
                            for s in range(self.num_states):
                                if (corners[s] > EPSILON).any():
                                    r, c = self.state_to_index(s)
                                    dist = math.sqrt(
                                        (r - vertex[0]) ** 2 + (c - vertex[1]) ** 2
                                    )
                                    if dist < min_dist:
                                        min_dist = dist
                                        min_s = s
                            if min_s is not None:
                                min_r, min_c = self.state_to_index(min_s)
                                for a in range(4):
                                    if corners[min_s][a] > EPSILON:
                                        dr, dc = self.two_step_deltas[a]
                                        if not self.experienced_transitions[
                                            self.index_to_state(min_r + dr, min_c),
                                            self.one_step_deltas.index((0, dc)),
                                        ]:
                                            self.exploration_target_transitions[
                                                self.index_to_state(min_r + dr, min_c),
                                                self.one_step_deltas.index((0, dc)),
                                                self.index_to_state(
                                                    min_r + dr, min_c + dc
                                                ),
                                            ] = True
                                        if not self.experienced_transitions[
                                            self.index_to_state(min_r, min_c + dc),
                                            self.one_step_deltas.index((dr, 0)),
                                        ]:
                                            self.exploration_target_transitions[
                                                self.index_to_state(min_r, min_c + dc),
                                                self.one_step_deltas.index((dr, 0)),
                                                self.index_to_state(
                                                    min_r + dr, min_c + dc
                                                ),
                                            ] = True

                    else:  # alignment가 있는 경우
                        ranked_subgraphs = self.rank_subgraphs(
                            valid_alignments, corresponding_subgraphs
                        )  # 그래프 간의 gw distance를 계산함
                        # Step 1: Extract the third element from each tuple
                        gw_dist = [t[2] for t in ranked_subgraphs]
                        # Step 2: Find the minimum value among these third elements
                        min_gw_dist = min(gw_dist)
                        # Step 3: Filter the tuples that have this minimum value as their third element
                        filtered_tuples = [
                            t for t in ranked_subgraphs if t[2] == min_gw_dist
                        ]
                        candidiate_states = []
                        for (
                            alignment,
                            corresponding_subgraph,
                            gw_distance,
                        ) in filtered_tuples:
                            for full_node, partial_node in alignment.items():
                                if (
                                    self.topological_prior.degree(full_node)
                                    - corresponding_subgraph.degree(partial_node)
                                    > 0
                                ):
                                    partial_node_state = self.index_to_state(
                                        partial_node[0], partial_node[1]
                                    )
                                    candidiate_states.append(partial_node_state)
                        place_cells = self.sr[:, state]
                        next_determined_state = max(
                            candidiate_states,
                            key=lambda s: place_cells[s],
                        )
                        for a in range(4):
                            if not (
                                self.experienced_transitions
                                + self.env_border_transitions
                            )[next_determined_state, a]:
                                next_r, next_c = self.state_to_index(
                                    next_determined_state
                                )
                                dr, dc = self.one_step_deltas[a]
                                self.exploration_target_transitions[
                                    next_determined_state,
                                    a,
                                    self.index_to_state(next_r + dr, next_c + dc),
                                ] = True
                if (
                    GM.is_isomorphic()
                    or (1 - self.exploration_target_transitions).astype(bool).all()
                ):
                    states_with_unexperienced_transitions = [
                        s
                        for s in range(self.num_states)
                        if (
                            (
                                self.experienced_transitions
                                + self.env_border_transitions
                            )[s]
                            == False
                        ).any()
                    ]
                    if len(states_with_unexperienced_transitions):
                        place_cells = self.sr[:, state]
                        state_with_highest_value = max(
                            states_with_unexperienced_transitions,
                            key=lambda s: place_cells[s],
                        )  ## 설정하고 보니 접근이 불가능한 곳이라면? - 이 관련 처리가 되어있는지? - 항상 접근 가능한 곳으로 세팅되도록 설계되어있음
                        unexperienced_transition_indices = [
                            i
                            for i, x in enumerate(
                                (
                                    self.experienced_transitions
                                    + self.env_border_transitions
                                )[state_with_highest_value]
                            )
                            if not x
                        ]
                        for a in unexperienced_transition_indices:
                            r, c = self.state_to_index(state_with_highest_value)
                            dr, dc = self.one_step_deltas[a]
                            if (
                                0 <= r + dr < self.num_row
                                and 0 <= c + dc < self.num_column
                            ):
                                self.exploration_target_transitions[
                                    state_with_highest_value,
                                    a,
                                    self.index_to_state(r + dr, c + dc),
                                ] = True

    def calculate_next_state_prob(self, placefield, state):
        prob = [0, 0, 0, 0]
        row, column = self.state_to_index(state)
        _, borders_exclude_unexplored = self.structure_head_corner_stack(
            include_unexplored=False
        )
        actionable_transitions = (1 - borders_exclude_unexplored).astype(bool)
        if 0 <= column - 1 < self.num_column and not (
            self.experienced_transitions[state, 0]
            and not actionable_transitions[state, 0]
        ):
            prob[0] = placefield[self.index_to_state(row, column - 1)]
        if 0 <= column + 1 < self.num_column and not (
            self.experienced_transitions[state, 1]
            and not actionable_transitions[state, 1]
        ):
            prob[1] = placefield[self.index_to_state(row, column + 1)]
        if 0 <= row - 1 < self.num_row and not (
            self.experienced_transitions[state, 2]
            and not actionable_transitions[state, 2]
        ):
            prob[2] = placefield[self.index_to_state(row - 1, column)]
        if 0 <= row + 1 < self.num_row and not (
            self.experienced_transitions[state, 3]
            and not actionable_transitions[state, 3]
        ):
            prob[3] = placefield[self.index_to_state(row + 1, column)]
        return prob

    def sample_action_exploration(self, state):
        if (
            (self.experienced_transitions + self.env_border_transitions) == False
        ).any():
            if self.topological_prior is not None:
                if (1 - self.exploration_target_transitions).astype(bool).all():
                    self.set_exploration_target(state)  # target이 없으면 세팅

            if (
                self.topological_prior is None
                or (1 - self.exploration_target_transitions).astype(bool).all()
            ):  # Prior 없거나 세팅을 시도해도 없으면
                if np.sum(self.transition_count[state]) == 0:
                    return np.random.choice(range(4))
                else:
                    return np.argmin(
                        self.transition_count[state]
                    )  # Least sampled action 선택

            else:  # 세팅이 되었다면
                state_list, action_list, next_state_list = np.where(
                    self.exploration_target_transitions
                )
                if state not in state_list:
                    target_states = np.array(
                        [1 if x in state_list else 0 for x in range(self.num_states)]
                    )
                    sr_including_unexplored = (
                        self.G_unexplored
                        @ np.diag(self.lambdas_unexplored)
                        @ self.G_inv_unexplored
                    )
                    exploration_placefield = np.matmul(
                        sr_including_unexplored, target_states.reshape(-1, 1)
                    ).flatten()
                    prob = self.calculate_next_state_prob(exploration_placefield, state)
                else:
                    state_indices = np.where(state_list == state)[0]
                    prob = np.array(
                        [
                            1.0 if x in action_list[state_indices] else 0.0
                            for x in range(4)
                        ]
                    )
                return self.choose_action(prob)
        else:
            return np.random.choice(range(4))


class TAGDecisionMaking(TAGTransfer):
    def __init__(self, num_row, num_column, gamma=0.995, seed=0):
        self.gamma_decision_making = 0.1
        super(TAGDecisionMaking, self).__init__(
            num_row, num_column, self.gamma_decision_making, seed
        )
        self.gamma_topological_planning = gamma
        self.start = None
        self.goal = None
        self.subgoals = []
        corners, _ = self.structure_head_corner_stack()
        self.outward_corners = np.where(np.sum(corners > 0, axis=1) > 0)[0]
        self.start_mapping = None
        self.goal_mapping = None
        self.subgoal_mappings = None
        self.next_goal = None
        self.goal_sequences = None
        self.states_in_focus_mapping = None
        self.placefield = None

    def rule_out_unapproachable_subgoals(self):
        test_sr = SRMB(10, 10)
        blocks = []
        for s in range(self.num_states):
            if np.sum(self.sr, axis=1)[s] < 1 / (1 - self.gamma) - EPSILON:
                blocks.append(s)
        blocks.append(self.goal)
        test_sr.update_map_structure(blocks=blocks)
        approachable_states = test_sr.sr[self.start]
        new_subgoals = []
        for subgoal in self.subgoals:
            if approachable_states[subgoal] > EPSILON:
                new_subgoals.append(subgoal)
        self.subgoals = new_subgoals

    def skeletonize_map(self):
        """
        skeletonization 시에 버그를 피하기 위해 preserved_nodes를 추가하지 않음
        """
        self.rule_out_unapproachable_subgoals()
        super(TAGDecisionMaking, self).skeletonize_map()
        self.subgoal_mappings = {}
        for i, vertex_states in enumerate(self.vertex_nodes):
            if self.start in vertex_states:
                self.start_mapping = ("vertex", self.vertex_corresp[i])
            if self.goal in vertex_states:
                self.goal_mapping = ("vertex", self.vertex_corresp[i])
            for subgoal in self.subgoals:
                if subgoal in vertex_states:
                    self.subgoal_mappings[subgoal] = ("vertex", self.vertex_corresp[i])
        for i, deadend_states in enumerate(self.deadend_nodes):
            if self.start in deadend_states:
                self.start_mapping = ("deadend", self.deadend_corresp[i])
            if self.goal in deadend_states:
                self.goal_mapping = ("deadend", self.deadend_corresp[i])
            for subgoal in self.subgoals:
                if subgoal in deadend_states:
                    self.subgoal_mappings[subgoal] = (
                        "deadend",
                        self.deadend_corresp[i],
                    )
        for i, edge_states in enumerate(self.edge_nodes):
            if self.start in edge_states:
                self.start_mapping = ("edge", self.edge_corresp[i])
            if self.goal in edge_states:
                self.goal_mapping = ("edge", self.edge_corresp[i])
            for subgoal in self.subgoals:
                if subgoal in edge_states:
                    self.subgoal_mappings[subgoal] = ("edge", self.edge_corresp[i])
        assert self.start_mapping is not None
        assert self.goal_mapping is not None

    def reset(self):
        super(TAGDecisionMaking, self).reset()
        self.start_mapping = None
        self.goal_mapping = None
        self.subgoal_mappings = None
        self.next_goal = None
        self.goal_sequences = None
        self.states_in_focus_mapping = None
        self.placefield = None

    def set_start(self, start):
        self.start = start

    def set_w(self, w, subgoals=None):
        self.w = w
        if subgoals is None:
            subgoals = []
        if np.sum(w > 0) == 1:
            self.goal = np.where(w > 0)[0][0]
        elif np.sum(w > 0) > 1:
            if np.sum(w > 0) - len(subgoals) != 1:
                raise ValueError("Invalid number of subgoals")
            else:
                self.subgoals = subgoals
                self.goal = np.setdiff1d(np.where(w > 0)[0], subgoals)[0]
        else:
            pass

    def set_placefield(self, state):
        if self.goal_sequences is None:
            self.determine_goal_order(state)
        if self.next_goal is None:
            self.next_goal = self.goal_sequences.pop(0)
        self.placefield = (
            self.sr @ np.eye(self.num_states)[:, self.next_goal]
        ).flatten()
        if len(self.goal_sequences) == 0:
            self.goal_sequences = None
        else:  # next goal not a goal
            self.placefield[self.goal] = 0  # not yet ready to visit goal

    def sample_action(self, state):
        if state == self.next_goal:
            self.next_goal = None
            self.placefield = None
        if self.placefield is None:
            self.set_placefield(state)
        if self.goal_sequences is not None:
            if state in self.goal_sequences:
                self.goal_sequences.pop(self.goal_sequences.index(state))
        prob = self.calculate_next_state_prob(self.placefield, state)
        return self.choose_action(prob)

    def find_all_paths(self, start, goal, path=[], visited=set(), total_weight=0):
        """
        Recursively find all paths from 'start' to 'goal', visiting each node at most once,
        while ensuring the path only moves 'closer' to the goal in terms of cumulative weight.
        """
        path = path + [start]  # Extend the current path with the current node
        visited.add(start)  # Mark the current node as visited

        # If the start is the goal, we have found a valid path
        if start == goal:
            return [(path, total_weight)]

        paths = []  # Store all valid paths from this point

        # Explore all neighbors (and their multiple edges) of the current node
        for neighbor in self.skeleton_graph.neighbors(start):
            if neighbor not in visited:
                # Explore all edges between the current node and the neighbor
                for key, edge_data in self.skeleton_graph[start][neighbor].items():
                    edge_weight = edge_data.get(
                        "weight", 1
                    )  # Default weight is 1 if not present

                    # Recur only if this edge brings us closer (in terms of cumulative weight)
                    new_total_weight = total_weight + edge_weight
                    new_paths = self.find_all_paths(
                        neighbor, goal, path, visited.copy(), new_total_weight
                    )

                    # Collect all valid paths from this recursion
                    paths.extend(new_paths)

        return paths

    def flatten(self, nested_list):
        """
        Recursively flatten a nested list or tuple into a single flat list,
        but preserve tuples with exactly two elements.
        """
        flat_list = []
        for item in nested_list:
            if isinstance(item, (list, tuple)) and not (
                isinstance(item, tuple) and len(item) == 2
            ):
                flat_list.extend(self.flatten(item))  # Recursively flatten
            else:
                flat_list.append(item)
        return flat_list

    def calculate_min_path_weights(self, path_graph):
        """
        Calculate the minimum path weights for all node pairs across all paths.
        One-step edge weights are taken from the skeleton graph, while multi-step paths
        use the minimum cumulative weight across all possible paths.
        """
        min_edge_weights = {}  # Dictionary to store the minimum weights for node pairs

        # Step 1: Use Dijkstra's algorithm to compute shortest paths between all node pairs
        for node in path_graph.nodes:
            # Compute shortest paths from 'node' to all other nodes in the path_graph
            lengths = nx.single_source_dijkstra_path_length(
                path_graph, node, weight="weight"
            )

            # Store the minimum weights for all reachable pairs
            for target, weight in lengths.items():
                min_edge_weights[(node, target)] = weight
                min_edge_weights[(target, node)] = weight  # Store reverse direction

        return min_edge_weights  # Return the dictionary of minimum path weights

    def calculate_subgoal_order_value(self, subgoals, gamma):
        """
        Calculate the value for a given order of subgoals.
        Each subgoal is a tuple (subgoal, weight, reward).
        """
        max_value = -float("inf")  # Initialize with a very small value
        best_order = None  # Store the best order of subgoals

        # Iterate through all possible orders (permutations) of the subgoals
        for order in permutations(subgoals):
            subgoal_sequence = []
            value = 0  # Initialize value for the current order
            accumulated_timesteps = 0  # Initialize accumulated timesteps

            # Calculate the value for the current order
            for i, (subgoal, weight, reward) in enumerate(order):
                subgoal_sequence.append(subgoal)
                # Calculate the reward contribution for this subgoal
                discounted_value = (
                    gamma ** (2 * accumulated_timesteps + weight)
                ) * reward
                value += discounted_value

                # Accumulate the timesteps for future calculations
                accumulated_timesteps += weight

            # Check if this order has the highest value so far
            if value > max_value:
                max_value = value
                best_order = subgoal_sequence

        # Return the best order and its corresponding value
        return (
            best_order,
            max_value,
            2 * accumulated_timesteps,
        )  # Double the accumulated timesteps

    def calculate_trajectory_value(
        self, trajectory, subgoal_path_mappings, min_edge_weights, gamma=0.9
    ):
        """
        Calculate the total value of a given trajectory using dynamic programming.
        """
        total_value = 0
        accumulated_timesteps = 0
        sequence_list = []

        # Iterate through the trajectory, accumulating value and discounting based on distance
        for i in range(1, len(trajectory) - 1):
            previous_node = trajectory[i - 1]
            current_node = trajectory[i]

            # Get the minimum distance (timestep) between the two nodes
            if previous_node == current_node:
                timestep = 0
            else:
                edge = (previous_node, current_node)
                timestep = min_edge_weights[edge]
            accumulated_timesteps += timestep

            # Add the subgoal reward if it exists
            if current_node in subgoal_path_mappings.keys():
                subgoal_rewards_list = subgoal_path_mappings[current_node]

                # Filter subgoals from the list (only those with type 'subgoal')
                subgoals = [
                    (s, w, r) for s, w, t, r in subgoal_rewards_list if t == "subgoal"
                ]

                # Use nested DP to determine the best order of subgoals
                if subgoals:
                    best_order, max_subgoal_value, subgoal_timesteps = (
                        self.calculate_subgoal_order_value(subgoals, gamma)
                    )
                    total_value += (gamma**accumulated_timesteps) * max_subgoal_value
                    accumulated_timesteps += (
                        subgoal_timesteps  # Add the timesteps for the subgoals
                    )
                    sequence_list.extend(best_order)

        previous_node = trajectory[-2]
        current_node = trajectory[-1]  # goal state
        sequence_list.append(current_node)

        # Get the minimum distance (timestep) between the two nodes
        if previous_node == current_node:
            timestep = 0
        else:
            edge = (previous_node, current_node)
            timestep = min_edge_weights[edge]
        accumulated_timesteps += timestep

        # Add the goal reward at the end of the trajectory
        for _, _, t, r in subgoal_path_mappings[self.goal_mapping]:
            if t == "goal":
                goal_reward = r
                break
        goal_discounted_value = (gamma**accumulated_timesteps) * goal_reward
        total_value += goal_discounted_value

        return total_value, sequence_list

    def plot_graph(self, graph, highlight_nodes=None, istuple=True):
        """
        Plot a NetworkX graph with optional node highlighting.

        Parameters:
        - graph: The NetworkX graph to plot.
        - highlight_nodes: A list of nodes to highlight (optional).
        - istuple: Whether node coordinates are tuples (row, column). Defaults to True.
        """
        # Set default highlight nodes to an empty list if not provided
        if highlight_nodes is None:
            highlight_nodes = []

        # Extract node positions: (column, -row) for intuitive plotting
        if istuple:
            pos = {node: (node[1], node[0]) for node in graph.nodes}
        else:
            pos = {node: self.state_to_index(node)[::-1] for node in graph.nodes}

        # Set node colors: Highlighted nodes get a different color
        node_colors = [
            "lightcoral" if node in highlight_nodes else "skyblue"
            for node in graph.nodes
        ]

        # Plot the graph with custom positions and node colors
        plt.figure(figsize=(8, 8))
        nx.draw(
            graph,
            pos=pos,
            with_labels=True,
            node_color=node_colors,
            node_size=500,
            font_size=10,
            edge_color="gray",
        )

        # Invert y-axis for intuitive row-column display
        plt.gca().invert_yaxis()
        plt.title("Graph with Highlighted Nodes")

    def debug_plot(self, path_graph, subgoal_path_mappings):

        whole_mapping = [self.start_mapping, self.goal_mapping]
        whole_mapping.extend(list(self.subgoal_mappings.keys()))
        self.plot_graph(self.skeleton_graph, whole_mapping, True)
        plt.savefig(f"display/skeleton_graph.png")
        plt.close()
        self.plot_graph(path_graph, list(subgoal_path_mappings.keys()), True)
        plt.savefig("display/path_graph.png")
        plt.close()

    def nearest_node_from_paths(self, target_node, path_nodes):
        """
        Find the nearest node from 'path_nodes' that is directly or indirectly connected
        to 'target_node' on the graph, leveraging the branch structure.

        Parameters:
            target_node: The node for which we want the nearest node.
            path_nodes: A list or set of nodes from the known paths.

        Returns:
            nearest: The closest node from 'path_nodes' connected to 'target_node'.
            weight: The cumulative weight of the path connecting them.
        """
        visited = set()  # Track visited nodes to avoid cycles
        queue = deque([(target_node, 0)])  # Store (current_node, accumulated_weight)
        nearest = None
        min_weight = float("inf")

        while queue:
            current_node, current_weight = queue.popleft()

            # Mark the current node as visited
            if current_node in visited:
                continue
            visited.add(current_node)

            # Check if the current node is in path_nodes
            if current_node in path_nodes:
                if current_weight < min_weight:
                    min_weight = current_weight
                    nearest = current_node

            # Explore neighbors of the current node
            for neighbor in self.skeleton_graph.neighbors(current_node):
                if neighbor not in visited:
                    # Check all edges between current_node and neighbor (MultiGraph case)
                    for _, edge_data in self.skeleton_graph[current_node][
                        neighbor
                    ].items():
                        weight = edge_data.get(
                            "weight", 1
                        )  # Default weight is 1 if not specified
                        queue.append((neighbor, current_weight + weight))

        return nearest, min_weight

    def determine_goal_order(self, state):
        self.skeletonize_map()
        paths = []
        if self.start_mapping[0] in ["vertex", "deadend"]:  # start on vertex or deadend
            start = self.start_mapping[1]
            if self.goal_mapping[0] in [
                "vertex",
                "deadend",
            ]:  # goal on vertex or deadend
                goal = self.goal_mapping[1]
                paths.extend(self.find_all_paths(start, goal))
            else:
                goal_node_1, goal_node_2 = self.goal_mapping[1]
                paths.extend(self.find_all_paths(start, goal_node_1))
                paths.extend(self.find_all_paths(start, goal_node_2))
        else:  # start on edge
            start_node_1, start_node_2 = self.start_mapping[1]
            if self.goal_mapping[0] in [
                "vertex",
                "deadend",
            ]:  # goal on vertex or deadend
                goal = self.goal_mapping[1]
                paths.extend(self.find_all_paths(start_node_1, goal))
                paths.extend(self.find_all_paths(start_node_2, goal))
            else:
                goal_node_1, goal_node_2 = self.goal_mapping[1]
                paths.extend(self.find_all_paths(start_node_1, goal_node_1))
                paths.extend(self.find_all_paths(start_node_1, goal_node_2))
                paths.extend(self.find_all_paths(start_node_2, goal_node_1))
                paths.extend(self.find_all_paths(start_node_2, goal_node_2))

        all_nodes_in_paths = []
        for path, _ in paths:
            all_nodes_in_paths.extend(path)
        all_nodes_in_paths = list(set(all_nodes_in_paths))

        path_graph = self.skeleton_graph.subgraph(
            all_nodes_in_paths
        )  # path가 지나가는 모든 edge node들로 구성된 subgraph

        # Extract all unique nodes from the paths
        path_nodes = set()
        max_path_length = -1
        for path, weight in paths:
            path_nodes.update(path)  # Add all nodes from the current path
            if len(path) > max_path_length:
                max_path_length = len(path)

        step_subgoal_mapping = {
            i: [] for i in range(max_path_length * 2 + 1)
        }  # path node뿐 아니라 사이에 낀 edge도 같이 고려

        # Convert to a list (optional) and print the result
        path_nodes = list(path_nodes)
        if len(path_nodes):

            for subgoal, (
                subgoal_type,
                subgoal_corresp,
            ) in self.subgoal_mappings.items():
                if subgoal_type in ["vertex", "deadend"]:
                    subgoal_node = subgoal_corresp
                    if subgoal_node in path_nodes:
                        path_location_list = []
                        for path, _ in paths:
                            if subgoal_node in path:
                                path_location_list.append(path.index(subgoal_node))
                        min_path_location = min(path_location_list)
                        step_subgoal_mapping[min_path_location * 2].append(subgoal)
                    else:
                        nearest, _ = self.nearest_node_from_paths(
                            subgoal_node, path_nodes
                        )
                        path_location_list = []
                        for path, _ in paths:
                            if nearest in path:
                                path_location_list.append(path.index(nearest))
                        assert len(path_location_list) > 0
                        min_path_location = min(path_location_list)
                        step_subgoal_mapping[min_path_location * 2].append(subgoal)
                else:
                    subgoal_node_1, subgoal_node_2 = subgoal_corresp
                    path_locations = []
                    for subgoal_node in [subgoal_node_1, subgoal_node_2]:
                        if subgoal_node in path_nodes:
                            path_location_list = []
                            for path, _ in paths:
                                if subgoal_node in path:
                                    path_location_list.append(path.index(subgoal_node))
                            min_path_location = min(path_location_list)
                            path_locations.append(min_path_location)
                        else:
                            nearest, _ = self.nearest_node_from_paths(
                                subgoal_node, path_nodes
                            )
                            path_location_list = []
                            for path, _ in paths:
                                if nearest in path:
                                    path_location_list.append(path.index(nearest))
                            assert len(path_location_list) > 0
                            min_path_location = min(path_location_list)
                            path_locations.append(min_path_location)
                    step_subgoal_mapping[min(path_locations) * 2 + 1].append(subgoal)

            current_state = state

            low_level_seq = []

            for i in range(max_path_length):
                i_current = i * 2
                i_between = i * 2 + 1
                i_next = i * 2 + 2
                while True:
                    candidate_subgoals = []
                    candidate_subgoals.extend(step_subgoal_mapping[i_current])
                    candidate_subgoals.extend(step_subgoal_mapping[i_between])
                    candidate_subgoals.extend(step_subgoal_mapping[i_next])
                    if len(candidate_subgoals):
                        current_placefield = self.sr[:, current_state].flatten()
                        highest_subgoal = None
                        highest_placefield = -1
                        for subgoal in candidate_subgoals:
                            if current_placefield[subgoal] > highest_placefield:
                                highest_subgoal = subgoal
                                highest_placefield = current_placefield[subgoal]
                        low_level_seq.append(highest_subgoal)
                        if highest_subgoal in step_subgoal_mapping[i_current]:
                            step_subgoal_mapping[i_current].remove(highest_subgoal)
                        elif highest_subgoal in step_subgoal_mapping[i_between]:
                            step_subgoal_mapping[i_between].remove(highest_subgoal)
                        else:
                            step_subgoal_mapping[i_next].remove(highest_subgoal)
                        current_state = highest_subgoal
                    else:
                        break
        else:
            low_level_seq = []
            current_state = state
            candidate_subgoals = copy.deepcopy(self.subgoals)
            while True:
                if len(candidate_subgoals):
                    current_placefield = self.sr[:, current_state].flatten()
                    highest_subgoal = None
                    highest_placefield = -1
                    for subgoal in candidate_subgoals:
                        if current_placefield[subgoal] > highest_placefield:
                            highest_subgoal = subgoal
                            highest_placefield = current_placefield[subgoal]
                    low_level_seq.append(highest_subgoal)
                    candidate_subgoals.remove(highest_subgoal)
                    current_state = highest_subgoal
                else:
                    break
        low_level_seq.append(self.goal)
        self.goal_sequences = low_level_seq


class TAGZeroShot(SR):
    def __init__(self, num_row, num_column, graph_kernel, graph_param, alpha_state_to_node, alpha_node_to_state, stochastic_action=False):
        super(TAGZeroShot, self).__init__(num_row, num_column, stochastic_action)
        self.graph_kernel = graph_kernel
        self.graph_param = graph_param
        self.alpha_state_to_node = alpha_state_to_node
        self.alpha_node_to_state = alpha_node_to_state

    def update_map(
        self, 
        G: nx.Graph,
        closest_skeleton_indices: dict,
        blocks: list[int],
        walls: list[tuple[int, int]],
    ) -> np.ndarray:
        """
        Compose a grid-world kernel by chaining three kernels:
        1) state -> graph-node via grid RBF
        2) graph-node -> graph-node via chosen graph kernel
        3) graph-node -> state via grid RBF

        Parameters:
        -----------
        G : networkx.Graph
            Skeleton graph.
        graph_kernel : {'rbf', 'heat'}
            Which graph kernel to use.
        graph_param : float
            Alpha (for RBF) or t (for heat).
        alpha_state_to_node : float
            RBF parameter from state to node.
        alpha_node_to_state : float
            RBF parameter from node to state.
        closest_skeleton_indices : dict
            Mapping graph-node -> list of grid-state (row,col) tuples.
        num_row, num_col : int
            Grid dimensions.

        Returns:
        --------
        K_grid : np.ndarray, shape=(num_row*num_col, num_row*num_col)
            Kernel matrix over all grid states.
        """
        # 1) compute graph-level kernel
        if self.graph_kernel == "rbf":
            K_graph, nodes_map = self.compute_graph_rbf_kernel_matrix(G)
        elif self.graph_kernel == "heat":
            K_graph, nodes_map = self.compute_graph_heat_kernel_matrix(G)
        else:
            raise ValueError("graph_kernel must be 'rbf' or 'heat'")

        # 2) compose kernels
        for node in G.nodes():
            node_coord = (node[0] // 3, node[1] // 3) 
            for node2 in G.nodes():
                node2_coord = (node2[0] // 3, node2[1] // 3)
                # graph-node -> graph-node kernel value
                k_graph = K_graph[nodes_map[node], nodes_map[node2]]
                # get all grid states closest to each node
                for s1 in closest_skeleton_indices[node]:
                    idx1 = self.index_to_state(s1[0], s1[1])
                    # state -> node kernel
                    k1 = self.compute_grid_rbf_kernel(s1, node_coord)
                    for s2 in closest_skeleton_indices[node2]:
                        idx2 = self.index_to_state(s2[0], s2[1])
                        # node -> state kernel
                        if self.is_visible(s1, s2, blocks, walls):
                            self.sr[idx1, idx2] = self.compute_grid_rbf_kernel(s1, s2)
                        else:
                            k3 = self.compute_grid_rbf_kernel(node2_coord, s2)
                            self.sr[idx1, idx2] = k1 * k_graph * k3

    def is_visible(self, s, t, blocks, walls):
        """
        s, t: (r,c) 정수 좌표
        blocks: set of blocked state indices
        walls: set of frozenset({i,j}) for each wall edge
        """
        if s == t:
            return True

        p0 = (s[0]+0.5, s[1]+0.5)
        p1 = (t[0]+0.5, t[1]+0.5)
        for (rf, cf) in bresenham_line(p0, p1, n_samples=50):

            ri, ci = int(rf), int(cf)
            if ri<0 or ri>=self.num_row or ci<0 or ci>=self.num_column:
                return False
            idx = self.index_to_state(ri, ci)
            if idx in blocks:
                return False

        pts = bresenham_line(p0, p1, n_samples=50)
        prev_cell = (int(pts[0][0]), int(pts[0][1]))
        for rf, cf in pts[1:]:
            cur_cell = (int(rf), int(cf))
            if cur_cell != prev_cell:
                i = self.index_to_state(prev_cell[0], prev_cell[1]) 
                j = self.index_to_state(cur_cell[0], cur_cell[1])
                test_tuple = (i, j) if i < j else (j, i)
                if test_tuple in walls:
                    return False
                prev_cell = cur_cell
        return True

            
    def compute_graph_rbf_kernel_matrix(self, G: nx.Graph) -> (np.ndarray, dict):
        """
        Compute the RBF (Gaussian) kernel matrix based on graph distances.

        Parameters:
        ----------
        G : networkx.Graph
            Weighted or unweighted graph. Edge weights are used if present.

        Returns:
        -------
        K : np.ndarray, shape=(n, n)
            RBF kernel matrix where K[i, j] = exp(-alpha * d_G(v_i, v_j)).
        nodes_map : dict
            Mapping from graph node to matrix index.
        """
        # Compute shortest-path distances
        all_pairs = dict(nx.floyd_warshall(G, weight="weight"))
        nodes = list(G.nodes())
        n = len(nodes)
        K = np.zeros((n, n))
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                d = all_pairs[u].get(v, np.inf)
                if np.isinf(d):
                    K[i, j] = 0.0
                else:
                    K[i, j] = np.exp(-self.graph_param * d)
        nodes_map = {node: idx for idx, node in enumerate(nodes)}
        return K, nodes_map


    def compute_graph_heat_kernel_matrix(self, G: nx.Graph) -> (np.ndarray, dict):
        """
        Compute the heat kernel matrix e^{-t L} for a graph.

        Parameters:
        ----------
        G : networkx.Graph
            Weighted or unweighted graph. Edge weights are used if present.
        Returns:
        -------
        H : np.ndarray, shape=(n, n)
            Heat kernel matrix given by the matrix exponential of -t*L.
        nodes_map : dict
            Mapping from graph node to matrix index.
        """
        nodes = list(G.nodes())
        n = len(nodes)
        # Build adjacency matrix
        A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
        # Degree matrix
        degrees = np.sum(A, axis=1)
        D = np.diag(degrees)
        # Combinatorial Laplacian
        L = D - A
        # Heat kernel via matrix exponential
        H = expm(-self.graph_param * L)
        nodes_map = {node: idx for idx, node in enumerate(nodes)}
        return H / np.max(H), nodes_map


    def compute_grid_rbf_kernel(self, point1: tuple, point2: tuple,) -> float:
        """
        Compute the RBF (Gaussian) kernel between two grid-world coordinates.

        Parameters
        ----------
        point1 : tuple of float or int
            The (x, y) coordinates of the first point.
        point2 : tuple of float or int
            The (x, y) coordinates of the second point.
        alpha : float
            The kernel width parameter (must be >= 0).

        Returns
        -------
        float
            The RBF kernel value exp(-alpha * ||point1 - point2||_2).
        """
        # Convert to arrays
        p = np.array(point1, dtype=float)
        q = np.array(point2, dtype=float)
        # Euclidean distance
        dist = np.linalg.norm(p - q)
        # RBF kernel
        return np.exp(-self.alpha_state_to_node * dist)


def bresenham_line(p0, p1, n_samples=100):
    r0, c0 = p0
    r1, c1 = p1
    rs = np.linspace(r0, r1, n_samples)
    cs = np.linspace(c0, c1, n_samples)
    return list(zip(rs, cs))


def and_layer(x):
    return np.matmul(x, np.array([1, 1])[:, np.newaxis]) - 1 > 0

def or_layer(x):
    return np.matmul(x, np.array([2, 2])[:, np.newaxis]) - 1 > 0


def nor_layer(x):
    return np.matmul(x, np.array([-1, -1])[:, np.newaxis]) + 1 > 0

def xnor_layer(x):
    return or_layer(np.squeeze(np.stack((and_layer(x), nor_layer(x)), axis=2))) > 0

def not_layer(x):
    return -x + 1
