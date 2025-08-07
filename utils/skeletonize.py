from operator import is_
from networkx import draw
from modules.env import MazeEnvironment
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib.collections import LineCollection
from collections import defaultdict
from tqdm import tqdm
from skimage.morphology import skeletonize
import networkx as nx
import copy
from sklearn.neighbors import NearestNeighbors
from modules.base import SRMB
import colorsys
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import skimage.morphology
import pickle
import multiprocessing as mp
import seaborn as sns
import os
from custom_skeletonization._skeletonize import skeletonize_custom

junction_color = "#416788"
deadend_color = "#646464"
edge_color = "#E1DAD2"


def plot_maze_image():
    env = MazeEnvironment(10, 10)
    for ratio_index in tqdm(range(1, 11)):
        for index in range(1, 101):
            env.set_index("{}_{}".format(ratio_index, index))
            env.load_map("data/env/train/")
            env.visualize(
                display=False,
                directory="data/skeletonize/maze_image/",
                puddle=False,
                trajectory=False,
                no_startgoal=True,
            )


def migrate_graph_to_multigraph(G):
    # Create a new MultiGraph
    MG = nx.MultiGraph()

    # Add all nodes from the original Graph
    MG.add_nodes_from(G.nodes(data=True))

    # Add all edges from the original Graph
    MG.add_edges_from(G.edges(data=True))

    # Set the weight of all edges to 1
    for u, v, key in MG.edges(keys=True):
        MG[u][v][key]['weight'] = 1

    return MG


def create_graph_from_boolean_matrix(matrix):
    # matrix = merge_3x3_to_1x1(matrix)
    rows = len(matrix)
    cols = len(matrix[0])
    G = nx.Graph()

    # Directions for horizontal and vertical connections
    hv_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Directions for diagonal connections with corresponding conditions
    diag_directions = [
        ((-1, -1), [(-1, 0), (0, -1)]),
        ((-1, 1), [(-1, 0), (0, 1)]),
        ((1, -1), [(1, 0), (0, -1)]),
        ((1, 1), [(1, 0), (0, 1)]),
    ]

    # Add nodes and edges
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j]:  # If there's a node
                G.add_node((i, j))
                hv_connected = set()

                # Check horizontal and vertical connections
                for d in hv_directions:
                    ni, nj = i + d[0], j + d[1]
                    if 0 <= ni < rows and 0 <= nj < cols and matrix[ni][nj]:
                        G.add_edge((i, j), (ni, nj))
                        hv_connected.add((d[0], d[1]))

                # Check diagonal connections if necessary
                for diag, hv_req in diag_directions:
                    ni, nj = i + diag[0], j + diag[1]
                    if 0 <= ni < rows and 0 <= nj < cols and matrix[ni][nj]:
                        # Check if all required hv directions are missing
                        if all((h[0], h[1]) not in hv_connected for h in hv_req):
                            G.add_edge((i, j), (ni, nj))
    # G = merge_nodes_by_division(G)
    MG = migrate_graph_to_multigraph(G)

    return MG


def is_bigger_than_left(tuple1, tuple2):
    if tuple1[0] < tuple2[0]:
        return True
    elif tuple1[0] == tuple2[0]:
        return tuple1[1] < tuple2[1]
    else:
        return False


def merge_nodes_by_division(G):
    new_G = nx.MultiGraph()
    node_mapping = {}

    # Map old nodes to new nodes
    for node in G.nodes:
        r, c = node
        new_node = (r // 3, c // 3)
        if new_node not in new_G:
            new_G.add_node(new_node)
        node_mapping[node] = new_node

    # Add edges to the new graph
    for u, v, data in G.edges(data=True):
        new_u = node_mapping[u]
        new_v = node_mapping[v]
        if new_u != new_v or (new_u == new_v and u == v):
            new_G.add_edge(new_u, new_v, **data)

    return new_G


def has_node_with_degree_not_equal_to_2(G):
    for node in G.nodes:
        if G.degree(node) != 2:
            return True
    return False


def simplify_graph_traverse(G, closest_skeleton_indices, preserved_nodes=None):
    if preserved_nodes is None:
        preserved_nodes = []
    edge_dict = {}
    is_not_torus = has_node_with_degree_not_equal_to_2(G)
    contracted_preserved_nodes = []
    while True:
        if G.number_of_nodes() == 1:
            break
        elif G.number_of_nodes() == 0:
            return None, None, None
        elif G.number_of_nodes() == 2:
            node1, node2 = list(G.nodes)
            if G.degree(node1) == 1 and G.degree(node2) == 1:
                merged_skel_indices = []
                num_merged_nodes = 0
                new_node_r_sum = 0
                new_node_c_sum = 0

                edge_key = (node1, node2) if is_bigger_than_left(node1, node2) else (node2, node1)
                edges = edge_dict.pop(edge_key, [])
                for edge in edges:
                    for node in edge:
                        skel_indices = closest_skeleton_indices.pop(node)
                        merged_skel_indices.extend(skel_indices)
                        new_node_r_sum += node[0]
                        new_node_c_sum += node[1]
                        num_merged_nodes += 1
                for node in G.nodes:
                    skel_indices = closest_skeleton_indices.pop(node)
                    merged_skel_indices.extend(skel_indices)
                    num_merged_nodes += 1
                    new_node_r_sum += node[0]
                    new_node_c_sum += node[1]
                G.remove_node(node1)
                G.remove_node(node2)
                assert node1 not in preserved_nodes
                assert node2 not in preserved_nodes
                new_node_r = new_node_r_sum // num_merged_nodes
                new_node_c = new_node_c_sum // num_merged_nodes
                new_node = (new_node_r, new_node_c)
                closest_skeleton_indices[new_node] = merged_skel_indices
                G.add_node(new_node)
                # edge도 다 포함해서 weighted average
                # closest skel point에 두 node와 edge 모두 모아서 합치기
        contraction_made = False
        for node in G.nodes:
            if (is_not_torus and G.degree(node) != 2) or (not is_not_torus and G.degree(node) == 2):
                # Find a neighbor with exactly two edges
                node_neighbors = list(G.neighbors(node))
                for neighbor in node_neighbors:
                    if G.degree(neighbor) == 2:
                        # Follow along the neighbor
                        path = [neighbor]
                        current = neighbor
                        start = node
                        need_contract = True
                        while True:
                            neighbors = list(G.neighbors(current))
                            next_node = None
                            for n in neighbors:
                                if n not in path and G.degree(n) == 2 and n != start:
                                    next_node = n
                                    break
                            if next_node is None:
                                if len(neighbors) != 2:  # 지금이 end 한 step 이전이어야 해서 edge여야 함
                                    need_contract = False
                                    break
                                end = (
                                    neighbors[0]
                                    if (
                                        neighbors[0] not in path
                                        and neighbors[0] != start
                                    )
                                    else neighbors[1]
                                )
                                if G.degree(end) == 2:  # loop일 때
                                    end = start
                                break
                            path.append(next_node)
                            current = next_node   
                        if need_contract:    
                            preserved_nodes_in_path = [n for n in path if n in preserved_nodes]
                            preserved_nodes_in_path_copy = preserved_nodes_in_path.copy()
                            preserved_nodes_in_path_copy.sort()
                            if len(preserved_nodes_in_path) > 0:
                                if (
                                    preserved_nodes_in_path_copy
                                    not in contracted_preserved_nodes
                                ):
                                    contracted_preserved_nodes.append(preserved_nodes_in_path_copy)
                                    if len(preserved_nodes_in_path) == len(path):
                                        for i, n in enumerate(path):
                                            if i == 0:
                                                key = (
                                                    (start, n)
                                                    if is_bigger_than_left(start, n)
                                                    else (n, start)
                                                )
                                                if key in edge_dict:
                                                    edge_dict[key].append([])
                                                else:
                                                    edge_dict[key] = [[]]
                                            else:
                                                key = (
                                                    (path[i - 1], n)
                                                    if is_bigger_than_left(path[i - 1], n)
                                                    else (n, path[i - 1])
                                                )
                                                if key in edge_dict:
                                                    edge_dict[key].append([])
                                                else:
                                                    edge_dict[key] = [[]]
                                        key = (
                                            (path[-1], end)
                                            if is_bigger_than_left(path[-1], end)
                                            else (end, path[-1])
                                        )
                                        if key in edge_dict:
                                            edge_dict[key].append([])
                                        else:
                                            edge_dict[key] = [[]]
                                        continue
                                    else:
                                        for n in path:
                                            if n not in preserved_nodes_in_path:
                                                G.remove_node(n)
                                                assert n not in preserved_nodes
                                        for i, n in enumerate(preserved_nodes_in_path):
                                            if i == 0:
                                                weight = path.index(n) + 1
                                                if weight > 1:
                                                    G.add_edge(start, n, weight=weight)
                                                key = (
                                                    (start, n)
                                                    if is_bigger_than_left(start, n)
                                                    else (n, start)
                                                )
                                                if key in edge_dict:
                                                    edge_dict[key].append(path[:path.index(n)])
                                                else:
                                                    edge_dict[key] = [path[:path.index(n)]]
                                            else:
                                                weight = path.index(n) - path.index(preserved_nodes_in_path[i - 1])
                                                if weight > 1:
                                                    G.add_edge(preserved_nodes_in_path[i - 1], n, weight=weight)
                                                key = (
                                                    (preserved_nodes_in_path[i - 1], n)
                                                    if is_bigger_than_left(preserved_nodes_in_path[i - 1], n)
                                                    else (n, preserved_nodes_in_path[i - 1])
                                                )
                                                if key in edge_dict:
                                                    edge_dict[key].append(path[path.index(preserved_nodes_in_path[i - 1]) + 1:path.index(n)])
                                                else:
                                                    edge_dict[key] = [path[path.index(preserved_nodes_in_path[i - 1]) + 1:path.index(n)]]
                                        weight = len(path) - path.index(preserved_nodes_in_path[-1])
                                        if weight > 1:
                                            G.add_edge(preserved_nodes_in_path[-1], end, weight=weight)
                                        key = (
                                            (preserved_nodes_in_path[-1], end)
                                            if is_bigger_than_left(preserved_nodes_in_path[-1], end)
                                            else (end, preserved_nodes_in_path[-1])
                                        )
                                        if key in edge_dict:
                                            edge_dict[key].append(path[path.index(preserved_nodes_in_path[-1]) + 1:])
                                        else:
                                            edge_dict[key] = [path[path.index(preserved_nodes_in_path[-1]) + 1:]]
                                        contraction_made = True
                                        break
                            else:
                                weight = len(path) + 1
                                G.add_edge(start, end, weight=weight)
                                for n in path:
                                    G.remove_node(n)
                                    assert n not in preserved_nodes
                                key = (
                                    (start, end)
                                    if is_bigger_than_left(start, end)
                                    else (end, start)
                                )
                                if key in edge_dict:
                                    edge_dict[key].append(path)
                                else:
                                    edge_dict[key] = [path]
                                contraction_made = True
                                break
                if contraction_made:
                    break
                else:
                    for neighbor in G.neighbors(node):
                        key = (
                            (node, neighbor)
                            if is_bigger_than_left(node, neighbor)
                            else (neighbor, node)
                        )
                        if key not in edge_dict:
                            edge_dict[key] = [[]]
        if not contraction_made:
            break
    for key in edge_dict:
        num_edges = G.number_of_edges(key[0], key[1]) if G.has_edge(key[0], key[1]) else G.number_of_edges(key[1], key[0])
        if num_edges != len(edge_dict[key]):
            for _ in range(num_edges - len(edge_dict[key])):
                edge_dict[key].append([])
    num_edge_dict_nodes = 0
    for key in edge_dict:
        num_edge_dict_nodes += len(edge_dict[key])
    if num_edge_dict_nodes != G.number_of_edges():
        return None, None, None

    num_skeleton_nodes = len(closest_skeleton_indices)
    num_contracted_nodes = 0
    for edge in edge_dict.values():
        for path in edge:
            num_contracted_nodes += len(list(set(path)))
    num_nodes = num_contracted_nodes + G.number_of_nodes()
    assert num_nodes == num_skeleton_nodes

    new_edge_dict = {}
    unskeletonized_edge_list = {}
    for key in edge_dict:
        new_key = ((key[0][0] // 3, key[0][1] // 3), (key[1][0] // 3, key[1][1] // 3))
        if new_key not in new_edge_dict:
            new_edge_dict[new_key] = []
        new_edge_dict[new_key].extend(edge_dict[key])
        if new_key not in unskeletonized_edge_list:
            unskeletonized_edge_list[new_key] = []
        for path in edge_dict[key]:
            states = []
            for node in path:
                for closest_node in closest_skeleton_indices[node]:
                    if closest_node not in new_key:
                        states.append(closest_node)
            unskeletonized_edge_list[new_key].append(list(dict.fromkeys(states)))

    unskeletonized_nodes_dict = {}
    for node in G.nodes():
        new_node = (node[0] // 3, node[1] // 3)
        if new_node in unskeletonized_nodes_dict:
            unskeletonized_nodes_dict[new_node].extend(closest_skeleton_indices[node])
            unskeletonized_nodes_dict[new_node] = list(set(unskeletonized_nodes_dict[new_node]))
        else:
            unskeletonized_nodes_dict[new_node] = list(set(closest_skeleton_indices[node]))

    G_1x1 = merge_nodes_by_division(G)

    return G_1x1, unskeletonized_edge_list, unskeletonized_nodes_dict, new_edge_dict


def preprocess_image(image):
    image_to_modify = np.copy(image)
    '''for r in range(1, image.shape[0] - 1):
        for c in range(1, image.shape[1] - 1):
            if (
                image[r, c]
                and not image[r, c - 1]
                and not image[r - 1, c]
            ):
                image_to_modify[r, c] = False
            if (
                image[r, c]
                and not image[r, c + 1]
                and not image[r - 1, c]
            ):
                image_to_modify[r, c] = False
            if (
                image[r, c]
                and not image[r, c - 1]
                and not image[r + 1, c]
            ):
                image_to_modify[r, c] = False
            if (
                image[r, c]
                and not image[r, c + 1]
                and not image[r + 1, c]
            ):
                image_to_modify[r, c] = False'''
    return image_to_modify


def merge_3x3_to_1x1(matrix):
    # Get the size of the original matrix
    size_3n_row = matrix.shape[0]
    size_3n_col = matrix.shape[1]
    # Calculate the size of the new matrix
    size_n_row = size_3n_row // 3
    size_n_col = size_3n_col // 3

    # Initialize the new n x n matrix with False
    new_matrix = np.zeros((size_n_row, size_n_col), dtype=bool)

    # Iterate over the 3n x 3n matrix in steps of 3
    for i in range(size_n_row):
        for j in range(size_n_col):
            # Extract the 3x3 sub-matrix
            sub_matrix = matrix[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3]
            # If any element in the sub-matrix is True, set the corresponding element in the new matrix to True
            if np.any(sub_matrix):
                new_matrix[i, j] = True

    return new_matrix


def map_pixels_to_skeleton_optimized(image, skeleton_3x3):
    # Get the indices of the non-zero (or white) pixels in the skeleton
    skeleton_indices_3x3 = np.argwhere(skeleton_3x3 > 0)

    # Prepare an array for all white pixel indices in the original image
    white_image_indices = np.argwhere(image > 0)
    white_image_indices_3x3 = white_image_indices * 3 + 1

    # Use NearestNeighbors from scikit-learn for efficient distance computation
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(white_image_indices_3x3)

    # Find the closest skeleton pixel for each white image pixel
    distances, indices = neigh.kneighbors(skeleton_indices_3x3)

    # Assuming closest_skeleton_indices should store both row and column indices of the closest skeleton point
    # Initialize it with an additional dimension for the indices
    closest_skeleton_indices = {}

    # Correctly assign the closest skeleton row and column indices for each white pixel
    for i, (r, c) in enumerate(skeleton_indices_3x3):
        closest_white_image_point = tuple(white_image_indices[indices[i]][0])
        if (r, c) not in closest_skeleton_indices:
            closest_skeleton_indices[(r, c)] = []
        closest_skeleton_indices[(r, c)].append(closest_white_image_point)

    # Add missing white pixels to the closest_skeleton_indices
    neigh_reverse = NearestNeighbors(n_neighbors=1)
    neigh_reverse.fit(skeleton_indices_3x3)
    distances_reverse, indices_reverse = neigh_reverse.kneighbors(white_image_indices_3x3)
    for i, (r, c) in enumerate(white_image_indices):
        closest_skeleton_point = tuple(skeleton_indices_3x3[indices_reverse[i]][0])
        assert closest_skeleton_point in closest_skeleton_indices
        if (r, c) not in closest_skeleton_indices[closest_skeleton_point]:
            closest_skeleton_indices[closest_skeleton_point].append((r, c))

    whites_in_skeleton = []
    for key in closest_skeleton_indices:
        whites_in_skeleton.extend(closest_skeleton_indices[key])
    assert len(list(set(whites_in_skeleton))) == len(white_image_indices)

    return closest_skeleton_indices


def build_skeleton_dict(skeleton: np.ndarray, label_map: np.ndarray) -> dict:
    """
    skeleton: (H,W) bool mask of the final skeleton
    label_map: (H,W) int array, each entry is linear index = sk_r*W + sk_c
    returns: dict mapping (sk_r,sk_c) -> [(i//3, j//3), ...]
    """
    H, W = label_map.shape
    result = {}
    # for every original pixel (i,j):
    for i in range(H):
        for j in range(W):
            # find which skeleton pixel it maps to
            lin = label_map[i, j]
            sk_r, sk_c = divmod(lin, W)
            # skip if that skel position is actually background
            if not skeleton[sk_r, sk_c]:
                continue
            key = (sk_r, sk_c)
            val = (i // 3, j // 3)
            # add to dict, avoiding duplicates
            if key not in result:
                result[key] = [val]
            else:
                if val not in result[key]:
                    result[key].append(val)
    return result


def assign_pixels_to_graph_elements(closest_skeleton_indices, skeleton_point_to_node):
    pixel_to_graph_element = {}
    for skeleton_point, pixels in closest_skeleton_indices.items():
        if skeleton_point in skeleton_point_to_node:
            # The skeleton point maps directly to a node
            node = skeleton_point_to_node[skeleton_point]
            for pixel in pixels:
                pixel_to_graph_element[pixel] = ("node", node)
        else:
            # Additional logic needed to assign pixels to edges if necessary
            pass
    return pixel_to_graph_element


def create_skeleton_image(env):
    image_original = generate_image(env)
    image = preprocess_image(image_original)
    skeleton = skeletonize(image)
    return image_original, image, skeleton


def visualize_label_map(
    skeleton: np.ndarray,
    label_map: np.ndarray,
    output_path: str = "./display/label_map.png",
):
    """
    skeleton: (H,W) bool mask
    label_map: (H,W) int array of linear IDs (row*W+col)
    output_path: where to save the PNG
    """
    H, W = label_map.shape

    # 1) 스켈레톤 픽셀 자신에게 줄 고유 ID 맵
    skel_ids = np.arange(H * W, dtype=np.int32).reshape(H, W)

    # 2) id_map: 스켈레톤은 자신의 ID, 나머지는 label_map 에 기록된 ID
    id_map = np.where(skeleton, skel_ids, label_map)

    # 3) 0…1 정규화
    norm = id_map.astype(float) / float(id_map.max() if id_map.max() > 0 else 1)

    # 4) rainbow cmap 적용 (RGBA), 우리는 RGB 채널만 쓸 거예요
    cmap = plt.cm.rainbow
    color_img = cmap(norm)[..., :3]  # shape (H,W,3)

    # 5) 디렉토리 없으면 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 6) 플롯 & 저장
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(color_img, interpolation="nearest")
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def skeletonize_env(env, with_mapping=False):
    image_original, image, skeleton = create_skeleton_image(env)
    image_1x1 = merge_3x3_to_1x1(image_original)
    if with_mapping:
        skeleton_with_mapping, label_map = skeletonize_custom(image)
        closest_skeleton_indices = build_skeleton_dict(skeleton_with_mapping, label_map)
    else:
        closest_skeleton_indices = map_pixels_to_skeleton_optimized(image_1x1, skeleton)

    whites_in_skeleton = []
    for key in closest_skeleton_indices:
        whites_in_skeleton.extend(closest_skeleton_indices[key])
    assert len(list(set(whites_in_skeleton))) == len(env.nonblocks)

    G = create_graph_from_boolean_matrix(skeleton)
    original_G = copy.deepcopy(G)
    simplified_G, unskeletonized_edges_dict, unskeletonized_nodes_dict, edge_dict = simplify_graph_traverse(
        G, closest_skeleton_indices
    )
    if simplified_G is None:
        return None, None, None, None, None, None

    total_nodes = []
    for edges in unskeletonized_edges_dict.values():
        for edge in edges:
            total_nodes.extend(edge)
    for nodes in unskeletonized_nodes_dict.values():
        total_nodes.extend(nodes)
    assert len(env.nonblocks) == len(list(set(total_nodes)))

    vertex_nodes, deadend_nodes, edge_nodes, vertex_corresp, deadend_corresp, edge_corresp = segment_nodes(
        simplified_G, unskeletonized_nodes_dict, unskeletonized_edges_dict, num_column=image.shape[1] // 3
    )
    all_nodes = []
    for nodes in vertex_nodes:
        all_nodes.extend(nodes)
    for nodes in deadend_nodes:
        all_nodes.extend(nodes)
    for nodes in edge_nodes:
        all_nodes.extend(nodes)
    assert len(env.nonblocks) == len(list(set(all_nodes)))
    if len(vertex_nodes) == 0 and len(deadend_nodes) == 0 and len(edge_nodes) == 0:
        deadend_nodes.extend(list(env.nonblocks))
        deadend_corresp.append(list(simplified_G.nodes())[0])

    return vertex_nodes, deadend_nodes, edge_nodes, simplified_G, image_original, skeleton, vertex_corresp, deadend_corresp, edge_corresp, edge_dict, original_G, closest_skeleton_indices


def darken_color(color, factor):
    """
    Darkens the given color by a specified factor.

    Parameters:
    - color: The color to darken. Can be specified as a named color, hex code, or RGB tuple.
    - factor: A value between 0 and 1, where 0 means no change and 1 means completely darkened.

    Returns:
    - The darkened color in RGB format.
    """
    '''# Convert the color to RGB format
    rgb = mcolors.to_rgb(color)'''

    # Convert RGB to HLS
    rgb = color
    h, lvar, s = colorsys.rgb_to_hls(*rgb)
    # Decrease the lightness
    lvar = max(0, lvar * (1 - factor))
    # Convert back to RGB
    darkened_rgb = colorsys.hls_to_rgb(h, lvar, s)
    # Return the darkened color
    return mcolors.to_hex(darkened_rgb)


def draw_scatter_plot_skeleton(
    datapoints,
    vertex_nodes,
    deadend_nodes,
    edge_nodes,
    nonblocks,
    directory="display/scatter",
    save=False,
    sizes_per_type=None,
    num_row=10,
    num_column=10,
    type="eigenmap",
    ax=None,
    fig=None,
    azim=-15, 
    elev=0,
    axis_off=True,

):
    # Colors for the scatter plot points
    colors = ['white' for _ in range(len(nonblocks))]
    sizes = [10 for _ in range(len(nonblocks))]
    for v_index in range(len(vertex_nodes)):
        # Calculate darkness factor based on position in list
        factor = v_index / len(vertex_nodes) / 2
        for node in vertex_nodes[v_index]:
            # colors[np.where(nonblocks == node)[0][0]] = darken_color("lightpink", factor)
            colors[np.where(nonblocks == node)[0][0]] = junction_color
            sizes[np.where(nonblocks == node)[0][0]] = 20 if sizes_per_type is None else sizes_per_type[0]
    for d_index in range(len(deadend_nodes)):
        factor = d_index / len(deadend_nodes) / 2
        for node in deadend_nodes[d_index]:
            '''colors[np.where(nonblocks == node)[0][0]] = darken_color(
                sns.color_palette("pastel", 8)[4], factor
            )'''
            colors[np.where(nonblocks == node)[0][0]] = deadend_color
            sizes[np.where(nonblocks == node)[0][0]] = 10 if sizes_per_type is None else sizes_per_type[1]
    for e_index in range(len(edge_nodes)):
        factor = e_index / len(edge_nodes) / 2
        for node in edge_nodes[e_index]:
            '''colors[np.where(nonblocks == node)[0][0]] = darken_color(
                sns.color_palette("pastel", 8)[2], factor
            )'''
            if not isinstance(node, list):
                colors[np.where(nonblocks == node)[0][0]] = edge_color
                sizes[np.where(nonblocks == node)[0][0]] = 3 if sizes_per_type is None else sizes_per_type[2]
            elif (isinstance(node, list) and len(node) > 0):
                for n in node:
                    colors[np.where(nonblocks == n)[0][0]] = edge_color
                    sizes[np.where(nonblocks == n)[0][0]] = 3 if sizes_per_type is None else sizes_per_type[2]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        ax_given = False
    else:
        ax_given = True
    # nodesize = 10 if type == 'eigenmap' else 15

    x = datapoints[:, 0]
    y = datapoints[:, 1]
    if len(datapoints[0]) >= 3:
        z = datapoints[:, 2]
        assert fig is not None
        ax_pos = ax.get_position()
        ax.remove()  # 기존의 2D 축 제거
        ax = fig.add_axes(ax_pos, projection="3d")
        ax.scatter(x, y, z, c=colors, s=sizes)
        ax.view_init(azim=azim, elev=elev)
    else:
        ax.scatter(x, y, c=colors, s=sizes)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if axis_off:
        ax.axis("off")

    # Save or show the plot
    if ax_given:
        return ax
    else:
        if save:
            plt.savefig(directory + ".png")
            plt.close()
        else:
            plt.show()


def generate_image(env):
    copied_env = copy.deepcopy(env)
    copied_env.scale_map(3)
    image = np.ones((copied_env.num_row, copied_env.num_column), dtype=bool)
    for r in range(copied_env.num_row):
        image[r, 0] = 0
        image[r, copied_env.num_column - 1] = 0
    for c in range(copied_env.num_column):
        image[0, c] = 0
        image[copied_env.num_row - 1, c] = 0
    for b in copied_env.blocks:
        br, bc = copied_env.state_to_index(b)
        image[br, bc] = 0
        if br - 1 >= 0:
            image[br - 1, bc] = 0
        if bc - 1 >= 0:
            image[br, bc - 1] = 0
        if br + 1 < copied_env.num_row:
            image[br + 1, bc] = 0
        if bc + 1 < copied_env.num_column:
            image[br, bc + 1] = 0
    for w in copied_env.walls:
        w1, w2 = w
        w1r, w1c = copied_env.state_to_index(w1)
        w2r, w2c = copied_env.state_to_index(w2)
        image[w1r, w1c] = 0
        image[w2r, w2c] = 0
    return image


def plot_state_segmentation(env, vertex_nodes, deadend_nodes, edge_nodes, index):
    vis = np.zeros((env.num_row, env.num_column))
    for block in env.blocks:
        brow, bcol = env.state_to_index(block)
        vis[brow, bcol] = 1
    # Define base colors
    colors = {
        'white': (1, 1, 1),
        'black': (0, 0, 0),
        'vertex_color': sns.color_palette("pastel", 8)[3],
        'deadend_color': sns.color_palette("pastel", 8)[4],
        'edge_color': sns.color_palette("pastel", 8)[2]
    }
    nodes = [vertex_nodes, deadend_nodes, edge_nodes]
    # Create a list of colors for the colormap
    color_list = [colors['white'], colors['black']]
    for base_index, base_color in enumerate(
        ["vertex_color", "deadend_color", "edge_color"]
    ):
        for i in range(len(nodes[base_index])):  # Assuming 5 levels of darkening
            color_list.append(darken_color(colors[base_color], factor=i * (0.5 / len(nodes[base_index]))))
            for node in nodes[base_index][i]:
                row, col = env.state_to_index(node)
                vis[row, col] = len(color_list) - 1
    cmap = LinearSegmentedColormap.from_list('custom_cmap', color_list, N=len(color_list))
    plt.matshow(vis, cmap=cmap, vmin=0, vmax=len(color_list) - 1)
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
    for w in env.walls:
        w_1, w_2 = w
        wrow_1, wcol_1 = env.state_to_index(w_1)
        wrow_2, wcol_2 = env.state_to_index(w_2)
        if wrow_1 == wrow_2:
            plt.vlines(
                (wcol_1 + wcol_2) / 2,
                wrow_1 - 1 / 2,
                wrow_1 + 1 / 2,
                colors="black",
            )
        elif wcol_1 == wcol_2:
            plt.hlines(
                (wrow_1 + wrow_2) / 2,
                wcol_1 - 1 / 2,
                wcol_1 + 1 / 2,
                colors="black",
            )
        else:
            raise ValueError()
    plt.savefig(f"data/skeletonize/maze_node_image/{index}.png")
    plt.close()


def plot_skeleton(image, skeleton, index):
    plt.imshow(- skeleton.astype(float) * 1 / 3 + image.astype(float), cmap="gray")
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
    plt.savefig(f"data/skeletonize/skeleton_image/{index}.png")
    plt.close()


def save_adj_matrix(G, index):
    adj_matrix = nx.to_numpy_array(G)
            # Alternatively, save the adjacency matrix to a binary file
    np.save(f"data/skeletonize/skeleton_adjmat/{index}.npy", adj_matrix)


def plot_basis_10_train():
    sizes_per_type = [500, 400, 300]
    erroneous_indices = []
    for ratio_index in tqdm(range(1, 11)):
        for index in tqdm(range(1, 101)):
            env = MazeEnvironment(10, 10)
            env.set_index("{}_{}".format(ratio_index, index))
            env.load_map("data/env/train/")
            image = generate_image(env)
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
            ) = skeletonize_env(env)
            with open(f'data/skeletonize/nodes/train/v_{ratio_index}_{index}.pkl', 'wb') as f:
                pickle.dump(vertex_nodes, f)
            with open(
                f"data/skeletonize/nodes/train/e_{ratio_index}_{index}.pkl", "wb"
            ) as f:
                pickle.dump(edge_nodes, f)
            with open(f'data/skeletonize/nodes/train/d_{ratio_index}_{index}.pkl', 'wb') as f:
                pickle.dump(deadend_nodes, f)
            with open(f'data/skeletonize/nodes/train/vc_{ratio_index}_{index}.pkl', 'wb') as f:
                pickle.dump(vertex_corresp, f)
            with open(
                f"data/skeletonize/nodes/train/ec_{ratio_index}_{index}.pkl", "wb"
            ) as f:
                pickle.dump(edge_corresp, f)
            with open(f'data/skeletonize/nodes/train/dc_{ratio_index}_{index}.pkl', 'wb') as f:
                pickle.dump(deadend_corresp, f)
            with open(f'data/skeletonize/edges/train/ed_{ratio_index}_{index}.pkl', 'wb') as f:
                pickle.dump(edge_dict, f)
            with open(f'data/skeletonize/skeleton_graph/train/skel_indices_{ratio_index}_{index}.pkl', 'wb') as f:
                pickle.dump(closest_skeleton_indices, f)
            nx.write_graphml(
                original_G,
                f"data/skeletonize/skeleton_graph/train/original_{ratio_index}_{index}.graphml",
            )

            if simplified_G is None:
                erroneous_indices.append((ratio_index, index))
                continue

            nx.write_graphml(simplified_G, f"data/skeletonize/skeleton_graph/train/{ratio_index}_{index}.graphml")

            plot_skeleton(image, skeleton, "train/{}_{}".format(ratio_index, index))
            draw_graph_with_curved_edges(simplified_G, "train/{}_{}".format(ratio_index, index))
            save_adj_matrix(simplified_G, "train/{}_{}".format(ratio_index, index))
            plot_state_segmentation(env, vertex_nodes, deadend_nodes, edge_nodes, "train/{}_{}".format(ratio_index, index))

            for bias in [True, False]:
                rd_prob = 0.5 if not bias else 0.9
                bias_str = "biased" if bias  else "unbiased"

                srmb = SRMB(10, 10)
                srmb.set_index("{}_{}".format(ratio_index, index))
                srmb.load_map_structure("data/env/train/", rd_prob=rd_prob)
                eigenvecs, eigenvals = srmb.decompose_map_eigen()
                eigenmaps = np.diag(eigenvals) @ eigenvecs
                X = eigenmaps[1:11][:, env.nonblocks].T
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                tsne = TSNE(
                    n_components=2, perplexity=10.0, learning_rate=10.0, random_state=42
                )
                X_tsne = tsne.fit_transform(X_scaled)
                draw_scatter_plot_skeleton(
                    eigenmaps[1:4][:, env.nonblocks].T,
                    vertex_nodes,
                    deadend_nodes,
                    edge_nodes,
                    env.nonblocks,
                    save=True,
                    directory="data/skeletonize/skeleton_scatter/train/{}_{}_{}".format(
                        ratio_index, index, bias_str
                    ),
                    sizes_per_type=sizes_per_type,
                )
                draw_scatter_plot_skeleton(
                    X_tsne,
                    vertex_nodes,
                    deadend_nodes,
                    edge_nodes,
                    env.nonblocks,
                    save=True,
                    directory="data/skeletonize/skeleton_scatter_tsne/train/{}_{}_{}".format(
                        ratio_index, index, bias_str
                    ),
                    sizes_per_type=sizes_per_type,
                )
    for ratio_index, index in erroneous_indices:
        print(f"Error occurred at {ratio_index}_{index}")


def plot_basis_10_test():
    sizes_per_type = [500, 400, 300]
    erroneous_indices = []
    for index in tqdm(range(1, 1001)):
        env = MazeEnvironment(10, 10)
        env.set_index("{}".format(index))
        env.load_map("data/env/test/")
        image = generate_image(env)
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
        ) = skeletonize_env(env)
        with open(f'data/skeletonize/nodes/test/v_{index}.pkl', 'wb') as f:
            pickle.dump(vertex_nodes, f)
        with open(
            f"data/skeletonize/nodes/test/e_{index}.pkl", "wb"
        ) as f:
            pickle.dump(edge_nodes, f)
        with open(f'data/skeletonize/nodes/test/d_{index}.pkl', 'wb') as f:
            pickle.dump(deadend_nodes, f)
        with open(f'data/skeletonize/nodes/test/vc_{index}.pkl', 'wb') as f:
            pickle.dump(vertex_corresp, f)
        with open(
            f"data/skeletonize/nodes/test/ec_{index}.pkl", "wb"
        ) as f:
            pickle.dump(edge_corresp, f)
        with open(f'data/skeletonize/nodes/test/dc_{index}.pkl', 'wb') as f:
            pickle.dump(deadend_corresp, f)
        with open(f'data/skeletonize/edges/test/ed_{index}.pkl', 'wb') as f:
            pickle.dump(edge_dict, f)
        with open(
            f"data/skeletonize/skeleton_graph/test/skel_indices_{index}.pkl",
            "wb",
        ) as f:
            pickle.dump(closest_skeleton_indices, f)
        nx.write_graphml(
            original_G,
            f"data/skeletonize/skeleton_graph/test/original_{index}.graphml",
        )

        if simplified_G is None:
            erroneous_indices.append((index))
            continue

        nx.write_graphml(simplified_G, f"data/skeletonize/skeleton_graph/test/{index}.graphml")
        plot_skeleton(image, skeleton, "test/{}".format(index))
        draw_graph_with_curved_edges(simplified_G, "test/{}".format(index))
        save_adj_matrix(simplified_G, "test/{}".format(index))
        plot_state_segmentation(env, vertex_nodes, deadend_nodes, edge_nodes, "test/{}".format(index))

        for bias in [True, False]:
            rd_prob = 0.5 if not bias else 0.9
            bias_str = "biased" if bias  else "unbiased"

            srmb = SRMB(10, 10)
            srmb.set_index("{}".format(index))
            srmb.load_map_structure("data/env/test/", rd_prob=rd_prob)
            eigenvecs, eigenvals = srmb.decompose_map_eigen()
            eigenmaps = np.diag(eigenvals) @ eigenvecs
            X = eigenmaps[1:11][:, env.nonblocks].T
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            tsne = TSNE(
                n_components=2, perplexity=10.0, learning_rate=10.0, random_state=42
            )
            X_tsne = tsne.fit_transform(X_scaled)
            draw_scatter_plot_skeleton(
                eigenmaps[1:4][:, env.nonblocks].T,
                vertex_nodes,
                deadend_nodes,
                edge_nodes,
                env.nonblocks,
                save=True,
                directory="data/skeletonize/skeleton_scatter/test/{}_{}".format(
                    index, bias_str
                ),
                sizes_per_type=sizes_per_type,
            )
            draw_scatter_plot_skeleton(
                X_tsne,
                vertex_nodes,
                deadend_nodes,
                edge_nodes,
                env.nonblocks,
                save=True,
                directory="data/skeletonize/skeleton_scatter_tsne/test/{}_{}".format(
                    index, bias_str
                ),
                sizes_per_type=sizes_per_type,
            )
    for ratio_index, index in erroneous_indices:
        print(f"Error occurred at {ratio_index}_{index}")



def draw_graph_with_curved_edges(
    G,
    index,
    rad: float = 0.2,
    self_loop_rad: float = 1,
    save: bool = True,
    pos: dict = None,
    ax=None,
    set_margin: bool = False,
    node_size: float = 500,
    edge_width: float = 7,
    use_distance_color: bool = False,
    n_samples: int = 50,
    source_node=None,
):
    """
    Draw a graph G with curved edges. By default (use_distance_color=False),
    nodes are colored by degree (junction/ dead‐end/ edge) using global
    junction_color, deadend_color, edge_color, and edges are drawn uniformly.
    
    If use_distance_color=True, nodes are colored by geodesic distance from
    `index` (jet colormap, red=closest, blue=farthest), and edges are drawn
    as smooth gradients between endpoint colors.

    Parameters
    ----------
    G : networkx Graph or DiGraph (or MultiGraph/MultiDiGraph)
        The graph to draw. May be directed or undirected.
    index : node
        If use_distance_color=True, this is the source node for distance
        calculations. If False, still used for saving filename.
    rad : float, optional (default=0.2)
        Base curvature factor for parallel edges. Higher `rad` → more curvature.
    self_loop_rad : float, optional (default=1)
        Curvature radius for self‐loop edges.
    save : bool, optional (default=True)
        If True and ax was None, save the figure to
        "data/skeletonize/skeleton_graph/{index}.png"; otherwise show.
    pos : dict, optional (default=None)
        A dict mapping each node to a 2‐tuple (x, y). If None, nodes are placed
        by matrix‐coordinate assumption: node is tuple (i,j) → pos=(j, -i).
    ax : matplotlib.axes.Axes, optional (default=None)
        Axes on which to draw. If None, a new figure/axes is created.
    set_margin : bool, optional (default=False)
        If True, add a 13% margin around the extents of `pos`.
    node_size : float, optional (default=500)
        Size of each node marker.
    edge_width : float, optional (default=7)
        Base width for each edge (segments inherit this).
    use_distance_color : bool, optional (default=False)
        If True, color nodes by their geodesic distance from `index` using
        the jet colormap, and draw edges with a gradient from source‐color to
        target‐color. If False, use original degree‐based coloring and uniform
        edge_color (global).
    n_samples : int, optional (default=50)
        Number of sample‐points along each edge curve for a smooth gradient
        (only used if use_distance_color=True).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the graph was drawn (unless saved and closed).
    """

    # 1) Prepare Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        ax_given = False
    else:
        ax_given = True

    # 2) Determine positions
    if pos is None:
        # Assume nodes are (i, j) tuples
        pos = {node: (node[1], -node[0]) for node in G.nodes()}

    # 3) Optional margin
    if set_margin:
        x_vals, y_vals = zip(*pos.values())
        x_margin = (max(x_vals) - min(x_vals)) * 0.13
        y_margin = (max(y_vals) - min(y_vals)) * 0.13
        ax.set_xlim(min(x_vals) - x_margin, max(x_vals) + x_margin)
        ax.set_ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)

    # 4) For curved edges, compute parallel groups and curvature mapping
    try:
        edge_iter = G.edges(keys=True)
        multi_edge = True
    except TypeError:
        # Not a MultiGraph: no keys
        edge_iter = [(u, v, None) for (u, v) in G.edges()]
        multi_edge = False

    parallel_groups = defaultdict(list)
    for u, v, key in edge_iter:
        grp = (u, v) if u <= v else (v, u)
        parallel_groups[grp].append((u, v, key))

    # Compute curvature for each (u,v,key)
    edge2rad = {}
    for grp_nodes, edge_list in parallel_groups.items():
        count = len(edge_list)
        rad_list = []
        if count == 1:
            rad_list = [0.0]
        else:
            if count % 2 == 1:
                # odd
                for i in range(-1, count - 1):
                    if i < 0:
                        r = 0.0
                    else:
                        r = rad * (i // 2 + 1)
                    sign = (i % 2) * 2 - 1
                    rad_list.append(r * sign)
            else:
                # even
                for i in range(count):
                    r = rad * (i // 2 + 1)
                    sign = (i % 2) * 2 - 1
                    rad_list.append(r * sign)
        for idx_edge, edge_id in enumerate(edge_list):
            edge2rad[edge_id] = rad_list[idx_edge]

    # 5) Node coloring
    if use_distance_color:
        # 5a) Distance‐based coloring
        G_und = G
        dist_dict = dict(nx.shortest_path_length(G_und, source=source_node, weight='weight'))
        if len(dist_dict) == 0:
            raise ValueError(f"Source node {index} not in graph or no reachable nodes.")
        max_dist = max(dist_dict.values())
        norm_dist = {}
        for node in G.nodes():
            if node in dist_dict:
                norm_dist[node] = dist_dict[node] / max_dist if max_dist > 0 else 0.0
            else:
                norm_dist[node] = 1.0  # unreachable → farthest (blue)
        cmap = cm.get_cmap("jet").reversed()
        node_colors = [cmap(norm_dist[node]) for node in G.nodes()]
    else:
        # 5b) Original degree‐based coloring using globals:
        #   junction_color if degree>2, deadend_color if degree==1, else "black"
        node_edge_count = {node: 0 for node in G.nodes()}
        for u, v in G.edges():
            node_edge_count[u] += 1
            node_edge_count[v] += 1
        node_colors = []
        for node, count in node_edge_count.items():
            if count > 2:
                node_colors.append(junction_color)
            elif count == 1:
                node_colors.append(deadend_color)
            else:
                node_colors.append("black")
        norm_dist = None

    # 6) Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        node_color=node_colors,
        ax=ax
    )

    # 7) Edge drawing
    if use_distance_color:
        # 7a) Draw gradient edges
        for grp_nodes, edge_list in parallel_groups.items():
            for (u, v, key) in edge_list:
                p0 = np.array(pos[u], dtype=float)
                p2 = np.array(pos[v], dtype=float)
                color_u = np.array(cmap(norm_dist[u]))
                color_v = np.array(cmap(norm_dist[v]))
                curv = edge2rad[(u, v, key)]

                if u == v:
                    # self‐loop
                    theta = np.linspace(0, 2 * np.pi, n_samples)
                    loop_radius = self_loop_rad * 0.1
                    cx, cy = p0
                    xs = cx + loop_radius * np.cos(theta)
                    ys = cy + loop_radius * np.sin(theta)
                    points = np.vstack([xs, ys]).T
                else:
                    if np.isclose(curv, 0.0):
                        ts = np.linspace(0.0, 1.0, n_samples)
                        points = np.outer(1 - ts, p0) + np.outer(ts, p2)
                    else:
                        m = 0.5 * (p0 + p2)
                        d = p2 - p0
                        if np.allclose(d, 0):
                            ts = np.linspace(0.0, 1.0, n_samples)
                            points = np.outer(1 - ts, p0) + np.outer(ts, p2)
                        else:
                            perp = np.array([-d[1], d[0]])
                            perp_norm = perp / np.linalg.norm(perp)
                            cp = m + perp_norm * (curv * np.linalg.norm(d))
                            ts = np.linspace(0.0, 1.0, n_samples)
                            one_minus_t = 1 - ts
                            points = (
                                np.outer(one_minus_t**2, p0)
                                + np.outer(2 * one_minus_t * ts, cp)
                                + np.outer(ts**2, p2)
                            )
                segments = np.stack([points[:-1], points[1:]], axis=1)
                t_vals = np.linspace(norm_dist[u], norm_dist[v], n_samples - 1)

                seg_colors = [cmap(t) for t in t_vals]
                lc = LineCollection(
                    segments,
                    colors=seg_colors,
                    linewidths=edge_width,
                    zorder=1,
                    capstyle="round"
                )
                ax.add_collection(lc)
    else:
        # 7b) Original edge drawing (uniform edge_color)
        # First, build count of parallel edges to determine curvature usage
        num_edges = {}
        for u, v, key in edge_iter:
            pair = (u, v) if (u, v) in num_edges else (v, u) if (v, u) in num_edges else None
            if pair is not None:
                num_edges[pair] += 1
            else:
                num_edges[(u, v)] = 1

        for u, v in G.edges():
            pair = (u, v) if (u, v) in num_edges else (v, u)
            num_edge = num_edges.get(pair, 1)
            if num_edge > 1:
                if num_edge % 2:
                    for i in range(-1, num_edge - 1):
                        rad_curved = rad * (i // 2 + 1) if i >= 0 else 0
                        sign = (i % 2) * 2 - 1
                        nx.draw_networkx_edges(
                            G,
                            pos,
                            edgelist=[(u, v)],
                            connectionstyle=f"arc3,rad={rad_curved * sign}",
                            edge_color=edge_color,
                            width=edge_width,
                            arrows=True,
                            ax=ax
                        )
                else:
                    for i in range(num_edge):
                        rad_curved = rad * (i // 2 + 1)
                        sign = (i % 2) * 2 - 1
                        nx.draw_networkx_edges(
                            G,
                            pos,
                            edgelist=[(u, v)],
                            connectionstyle=f"arc3,rad={rad_curved * sign}",
                            edge_color=edge_color,
                            width=edge_width,
                            arrows=True,
                            ax=ax
                        )
            else:
                if u != v:
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=[(u, v)],
                        edge_color=edge_color,
                        width=edge_width,
                        ax=ax
                    )
                else:
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=[(u, v)],
                        connectionstyle=f"arc3,rad={self_loop_rad}",
                        edge_color="red",
                        width=edge_width,
                        arrows=True,
                        ax=ax
                    )

    # 8) Final cleanup
    ax.set_axis_off()

    if ax_given:
        return ax, norm_dist
    else:
        if save:
            plt.savefig(f"data/skeletonize/skeleton_graph/{index}.png", bbox_inches="tight", pad_inches=0)
            plt.close()
        else:
            plt.show()
        return ax


def segment_nodes(G, unskeletonized_nodes_dict, unskeletonized_edges_dict, num_column=10):
    vertex_corresp = [
        n for n in G.nodes() if G.degree(n) > 2
    ]
    vertex_nodes = [
        [s[0] * num_column + s[1] for s in unskeletonized_nodes_dict[n]]
        for n in G.nodes() if G.degree(n) > 2
    ]
    vertex_pair = list(zip(vertex_corresp, vertex_nodes))
    vertex_pair.sort()
    vertex_corresp, vertex_nodes = zip(*vertex_pair) if len(vertex_pair) else ([], [])

    deadend_corresp = [
        n for n in G.nodes() if G.degree(n) <= 1
    ]
    deadend_nodes = [
        [s[0] * num_column + s[1] for s in unskeletonized_nodes_dict[n]]
        for n in G.nodes() if G.degree(n) <= 1
    ]
    deadend_pair = list(zip(deadend_corresp, deadend_nodes))
    deadend_pair.sort()
    deadend_corresp, deadend_nodes = (
        zip(*deadend_pair) if len(deadend_pair) else ([], [])
    )
    edge_corresp = [key for key in unskeletonized_edges_dict]
    edge_nodes = []
    for key in unskeletonized_edges_dict:
        edge_node_list = []
        for path in unskeletonized_edges_dict[key]:
            '''
            현 시점에서는 multiple edge의 경우 이를 구분하지 않음.
            '''
            edge_node_list.extend(
                [s[0] * num_column + s[1] for s in path]
            )
        edge_nodes.append(edge_node_list)
    '''edge_nodes = [
        [n[0] * num_column + n[1] for n in path]
        for key in unskeletonized_edges_dict
        for path in unskeletonized_edges_dict[key]
    ]'''
    edge_corresp.extend([(n, n) for n in G.nodes() if G.degree(n) == 2])
    for n in G.nodes():
        if G.degree(n) == 2:
            edge_nodes.append([s[0] * num_column + s[1] for s in unskeletonized_nodes_dict[n]])
    edge_pair = list(zip(edge_corresp, edge_nodes))
    edge_pair.sort()
    edge_corresp, edge_nodes = zip(*edge_pair) if len(edge_pair) else ([], [])
    return vertex_nodes, deadend_nodes, edge_nodes, vertex_corresp, deadend_corresp, edge_corresp


def create_skeleton_image_parametric(env, max_iter=None):
    """
    env 로부터 raw image → preprocess → skeletonize(thin) 수행.
    max_iter=None 이면 skimage.morphology.skeletonize,
    max_iter=k 이면 skimage.morphology.thin(image, max_iter=k).
    """
    image_original = generate_image(env)
    image = preprocess_image(image_original)
    if max_iter is not None:
        # Zhang–Suen thinning을 k회만 수행
        skeleton = skimage.morphology.thin(image, max_num_iter=max_iter)
    else:
        # 완전한 스켈레톤
        skeleton = skimage.morphology.skeletonize(image)
    return image_original, image, skeleton


def skeletonize_env_parametric(env, max_iter=None):
    """
    기존 skeletonize_env와 동일한 로직을 타되,
    스켈레톤 생성 부분만 create_skeleton_image_parametric 으로 대체.
    """
    # 1) 스켈레톤 생성 (iteration 조절)
    image_original, image, skeleton = create_skeleton_image_parametric(env, max_iter)

    # 2) 1×1 그리드로 축소
    image_1x1 = merge_3x3_to_1x1(image_original)

    # 3) 스켈레톤↔원본 픽셀 매핑
    closest_skeleton_indices = map_pixels_to_skeleton_optimized(image_1x1, skeleton)

    # 4) 그래프 생성 & 단순화
    G = create_graph_from_boolean_matrix(skeleton)
    original_G = copy.deepcopy(G)
    simplified_G, unskeletonized_edges_dict, unskeletonized_nodes_dict, edge_dict = (
        simplify_graph_traverse(G, closest_skeleton_indices)
    )
    if simplified_G is None:
        return (None,) * 12

    # 5) 노드 분류
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

    # 6) 리턴 (원본 env → 이미지 → 스켈레톤 → 그래프)
    return (
        vertex_nodes,
        deadend_nodes,
        edge_nodes,
        simplified_G,
        image_original,
        skeleton,
        vertex_corresp,
        deadend_corresp,
        edge_corresp,
        edge_dict,
        original_G,
        closest_skeleton_indices,
    )


def test():
    env_1 = MazeEnvironment(13, 13)
    env_1.set_index("vertical_concat")
    env_2 = MazeEnvironment(13, 13)
    env_2.set_index("horizontal_concat")
    block_1 = [(0, 1), (1, 0), (1, 1), (2, 1)]
    block_2 = [(0, 0), (0, 1), (0, 2)]
    block_3 = [(0, 0), (1, 0), (2, 0), (2, 1)]
    block_4 = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    blocks = [block_1, block_2, block_3, block_4]
    block_1_shift_1 = (4, 2)
    block_2_shift_1 = (4, 8)
    block_3_shift_1 = (7, 3)
    block_4_shift_1 = (5, 8)
    block_shifts_1 = [
        block_1_shift_1,
        block_2_shift_1,
        block_3_shift_1,
        block_4_shift_1,
    ]
    block_1_shift_2 = (2, 5)
    block_2_shift_2 = (3, 7)
    block_3_shift_2 = (8, 5)
    block_4_shift_2 = (10, 7)
    block_shifts_2 = [
        block_1_shift_2,
        block_2_shift_2,
        block_3_shift_2,
        block_4_shift_2,
    ]
    env_1_blocks = np.concatenate(
        [
            [
                env_1.index_to_state(
                    br + block_shifts_1[i][0], bc + block_shifts_1[i][1]
                )
                for br, bc in blocks[i]
            ]
            for i in range(4)
        ]
    )
    env_1.update_map(env_1_blocks)
    env_1.visualize(directory="./display/")

    env_2_blocks = np.concatenate(
        [
            [
                env_2.index_to_state(
                    br + block_shifts_2[i][0], bc + block_shifts_2[i][1]
                )
                for br, bc in blocks[i]
            ]
            for i in range(4)
        ]
    )
    env_2.update_map(env_2_blocks)
    env_2.visualize(directory="./display/")

    (
        vertex_nodes_1,
        deadend_nodes_1,
        edge_nodes_1,
        simplified_G_1,
        image_1,
        skeleton_1,
        vertex_corresp_1,
        deadend_corresp_1,
        edge_corresp_1,
        edge_dict_1,
        original_G,
        closest_skeleton_indices,
    ) = skeletonize_env(env_1)
    plot_skeleton(image_1, skeleton_1, "vertical_concat")
    draw_graph_with_curved_edges(simplified_G_1, "vertical_concat")

    (
        vertex_nodes_2,
        deadend_nodes_2,
        edge_nodes_2,
        simplified_G_2,
        image_2,
        skeleton_2,
        vertex_corresp_2,
        deadend_corresp_2,
        edge_corresp_2,
        edge_dict_2,
        original_G,
        closest_skeleton_indices,
    ) = skeletonize_env(env_2)
    plot_skeleton(image_2, skeleton_2, "horizontal_concat")
    draw_graph_with_curved_edges(simplified_G_2, "horizontal_concat")


def plot_basis_10_train_parametric():
    sizes_per_type = [500, 400, 300]
    erroneous_indices = []
    for ratio_index in tqdm(range(1, 11)):
        for index in tqdm(range(1, 101)):
            env = MazeEnvironment(10, 10)
            env.set_index("{}_{}".format(ratio_index, index))
            env.load_map("data/env/train/")
            for max_iter in range(1, 21):
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
                ) = skeletonize_env_parametric(env, max_iter=max_iter)
                with open(
                    f"data/skeletonize/nodes/train_parametric/v_{ratio_index}_{index}_{max_iter}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(vertex_nodes, f)
                with open(
                    f"data/skeletonize/nodes/train_parametric/e_{ratio_index}_{index}_{max_iter}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(edge_nodes, f)
                with open(
                    f"data/skeletonize/nodes/train_parametric/d_{ratio_index}_{index}_{max_iter}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(deadend_nodes, f)
                with open(
                    f"data/skeletonize/nodes/train_parametric/vc_{ratio_index}_{index}_{max_iter}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(vertex_corresp, f)
                with open(
                    f"data/skeletonize/nodes/train_parametric/ec_{ratio_index}_{index}_{max_iter}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(edge_corresp, f)
                with open(
                    f"data/skeletonize/nodes/train_parametric/dc_{ratio_index}_{index}_{max_iter}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(deadend_corresp, f)
                with open(
                    f"data/skeletonize/edges/train_parametric/ed_{ratio_index}_{index}_{max_iter}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(edge_dict, f)
                with open(
                    f"data/skeletonize/skeleton_graph/train_parametric/skel_indices_{ratio_index}_{index}_{max_iter}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(closest_skeleton_indices, f)
                nx.write_graphml(
                    original_G,
                    f"data/skeletonize/skeleton_graph/train_parametric/original_{ratio_index}_{index}_{max_iter}.graphml",
                )

                if simplified_G is None:
                    erroneous_indices.append((ratio_index, index, max_iter))
                    continue

                nx.write_graphml(
                    simplified_G,
                    f"data/skeletonize/skeleton_graph/train_parametric/{ratio_index}_{index}_{max_iter}.graphml",
                )

                plot_skeleton(
                    image,
                    skeleton,
                    f"train_parametric/{ratio_index}_{index}_{max_iter}",
                )
                draw_graph_with_curved_edges(
                    simplified_G, f"train_parametric/{ratio_index}_{index}_{max_iter}"
                )
                save_adj_matrix(
                    simplified_G, f"train_parametric/{ratio_index}_{index}_{max_iter}",
                )
                plot_state_segmentation(
                    env,
                    vertex_nodes,
                    deadend_nodes,
                    edge_nodes,
                    f"train_parametric/{ratio_index}_{index}_{max_iter}",
                )

                for bias in [True, False]:
                    rd_prob = 0.5 if not bias else 0.9
                    bias_str = "biased" if bias else "unbiased"

                    srmb = SRMB(10, 10)
                    srmb.set_index("{}_{}".format(ratio_index, index))
                    srmb.load_map_structure(
                        "data/env/train_parametric/", rd_prob=rd_prob
                    )
                    eigenvecs, eigenvals = srmb.decompose_map_eigen()
                    eigenmaps = np.diag(eigenvals) @ eigenvecs
                    X = eigenmaps[1:11][:, env.nonblocks].T
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    tsne = TSNE(
                        n_components=2, perplexity=10.0, learning_rate=10.0, random_state=42
                    )
                    X_tsne = tsne.fit_transform(X_scaled)
                    draw_scatter_plot_skeleton(
                        eigenmaps[1:4][:, env.nonblocks].T,
                        vertex_nodes,
                        deadend_nodes,
                        edge_nodes,
                        env.nonblocks,
                        save=True,
                        directory=f"data/skeletonize/skeleton_scatter/train_parametric/{ratio_index}_{index}_{bias_str}_{max_iter}",
                        sizes_per_type=sizes_per_type,
                    )
                    draw_scatter_plot_skeleton(
                        X_tsne,
                        vertex_nodes,
                        deadend_nodes,
                        edge_nodes,
                        env.nonblocks,
                        save=True,
                        directory=f"data/skeletonize/skeleton_scatter_tsne/train_parametric/{ratio_index}_{index}_{bias_str}_{max_iter}",
                        sizes_per_type=sizes_per_type,
                    )
        for ratio_index, index, max_iter in erroneous_indices:
            print(f"Error occurred at {ratio_index}_{index}_{max_iter}")


def _process_one_env_parametric(ratio_index, index):
    """
    plot_basis_10_train_parametric 내부의
    for max_iter in range(1,21): ... 블록만 따로 뽑아서 실행합니다.
    """
    env = MazeEnvironment(10, 10)
    env.set_index(f"{ratio_index}_{index}")
    env.load_map("data/env/train/")
    sizes_per_type = [500, 400, 300]

    for max_iter in range(1, 21):
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
        ) = skeletonize_env_parametric(env, max_iter=max_iter)

        # (아래 저장 로직은 원본 그대로 복사)
        with open(
            f"data/skeletonize/nodes/train_parametric/"
            f"v_{ratio_index}_{index}_{max_iter}.pkl",
            "wb",
        ) as f:
            pickle.dump(vertex_nodes, f)
        # … 이하 e_, d_, vc_, ec_, dc_, ed_ 파일 저장 …

        nx.write_graphml(
            original_G,
            f"data/skeletonize/skeleton_graph/"
            f"train_parametric/original_{ratio_index}_{index}_{max_iter}.graphml",
        )
        if simplified_G:
            nx.write_graphml(
                simplified_G,
                f"data/skeletonize/skeleton_graph/"
                f"train_parametric/{ratio_index}_{index}_{max_iter}.graphml",
            )

            plot_skeleton(
                image,
                skeleton,
                f"train_parametric/{ratio_index}_{index}_{max_iter}",
            )
            draw_graph_with_curved_edges(
                simplified_G,
                f"train_parametric/{ratio_index}_{index}_{max_iter}",
            )
            save_adj_matrix(
                simplified_G,
                f"train_parametric/{ratio_index}_{index}_{max_iter}",
            )
            plot_state_segmentation(
                env,
                vertex_nodes,
                deadend_nodes,
                edge_nodes,
                f"train_parametric/{ratio_index}_{index}_{max_iter}",
            )

            for bias in (True, False):
                rd_prob = 0.9 if bias else 0.5
                bias_str = "biased" if bias else "unbiased"
                srmb = SRMB(10, 10)
                srmb.set_index(f"{ratio_index}_{index}")
                srmb.load_map_structure(
                    "data/env/train_parametric/",
                    rd_prob=rd_prob,
                )
                eigenvecs, eigenvals = srmb.decompose_map_eigen()
                eigenmaps = np.diag(eigenvals) @ eigenvecs
                X = eigenmaps[1:11][:, env.nonblocks].T
                scaler = StandardScaler()
                X_tsne = TSNE(
                    n_components=2,
                    perplexity=10.0,
                    learning_rate=10.0,
                    random_state=42,
                ).fit_transform(scaler.fit_transform(X))

                draw_scatter_plot_skeleton(
                    eigenmaps[1:4][:, env.nonblocks].T,
                    vertex_nodes,
                    deadend_nodes,
                    edge_nodes,
                    env.nonblocks,
                    save=True,
                    directory=(
                        f"data/skeletonize/skeleton_scatter/"
                        + f"train_parametric/{ratio_index}_"
                        + f"{index}_{bias_str}_{max_iter}"
                    ),
                    sizes_per_type=sizes_per_type,
                )
                draw_scatter_plot_skeleton(
                    X_tsne,
                    vertex_nodes,
                    deadend_nodes,
                    edge_nodes,
                    env.nonblocks,
                    save=True,
                    directory=(
                        f"data/skeletonize/skeleton_scatter_tsne/"
                        + f"train_parametric/{ratio_index}_"
                        + f"{index}_{bias_str}_{max_iter}"
                    ),
                    sizes_per_type=sizes_per_type,
                )


def plot_basis_10_train_parametric_mp(processes=None):
    """
    multiprocessing.Pool을 이용해
    모든 (ratio_index, index) 조합에 대해
    _process_one_env_parametric을 병렬 실행합니다.
    """
    tasks = [(r, i) for r in range(1, 11) for i in range(1, 101)]
    processes = processes or mp.cpu_count()
    with mp.Pool(processes=processes) as pool:
        list(
            tqdm(
                pool.starmap(_process_one_env_parametric, tasks),
                total=len(tasks),
                desc="Parametric skeletonize (env-level)",
            )
        )


