import numpy as np
import networkx as nx
import dgh
from networkx.algorithms import isomorphism
import multiprocessing as mp
from functools import partial
from sklearn.manifold import Isomap
from ripser import ripser


def plot_H1_comparison_2(dgms1, thr1, dgms2, thr2, ax, labels=("Unbiased", "Biased")):
    lt1 = dgms1[1][:,1] - dgms1[1][:,0] - thr1
    lt1 = lt1[np.isfinite(lt1)]
    lt2 = dgms2[1][:,1] - dgms2[1][:,0] - thr2
    lt2 = lt2[np.isfinite(lt2)]

    n1 = len(lt1)
    n2 = len(lt2)

    centers = [1, 2]  

    face1, face2 = '#808080', '#C0C0C0'

    total_width = 0.1
    margin = 0.05  

    width1 = width2 = total_width

    def get_group_positions(center, n, width, margin):
        if n == 0:
            return []
        offset = ((n-1)*(width+margin))/2
        return [center - offset + i*(width+margin) for i in range(n)]

    positions1 = get_group_positions(centers[0], n1, width1, margin)
    for pos, val in zip(positions1, lt1):
        ax.bar(pos, val, width=width1, facecolor=face1, edgecolor='black', linewidth=1)

    positions2 = get_group_positions(centers[1], n2, width2, margin)
    for pos, val in zip(positions2, lt2):
        ax.bar(pos, val, width=width2, facecolor=face2, edgecolor='black', linewidth=1)

    for idx, n in enumerate([n1, n2]):
        ax.text(centers[idx], -0.08 * max([*lt1, *lt2, 1]), f"n={n}", ha='center', va='top', fontsize=11, color='black')

    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel("H1 Lifetime (relative)")
    ax.set_title("H1 Lifetime Comparison")


def compute_embedding(data_matrix, n_components=3, n_neighbors=10):
    iso = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    return iso.fit_transform(data_matrix)  # returns (n_samples x n_components)


def compute_persistent_homology(point_cloud, maxdim=2):
    result = ripser(point_cloud, maxdim=maxdim)
    return result["dgms"]  # list: [H0, H1, H2, ...]


def compute_null_lifetimes(
    data_matrix, n_null=200, n_components=3, n_neighbors=10, maxdim=2
):
    null_lf = {dim: [] for dim in range(maxdim + 1)}
    for _ in range(n_null):
        shuffled = np.random.permutation(data_matrix)
        emb = compute_embedding(shuffled, n_components, n_neighbors)
        dgms = compute_persistent_homology(emb, maxdim)
        for dim, dgm in enumerate(dgms):
            if dgm.size == 0:
                continue
            lifetimes = dgm[:, 1] - dgm[:, 0]
            finite_lts = lifetimes[np.isfinite(lifetimes)]
            null_lf[dim].extend(finite_lts.tolist())
    return null_lf


def find_robust_holes(dgms, null_lf, percentile=99.9):
    thresholds = {}
    robust = {}
    for dim, dgm in enumerate(dgms):
        lf_null = np.array(null_lf.get(dim, []))
        lf_null = lf_null[np.isfinite(lf_null)]
        if lf_null.size > 0:
            th = np.percentile(lf_null, percentile)
        else:
            th = np.inf
        thresholds[dim] = th

        lifetimes = dgm[:, 1] - dgm[:, 0] if dgm.size else np.array([])
        mask = np.isfinite(lifetimes) & (lifetimes > th)
        robust[dim] = dgm[mask]
    return thresholds, robust


def load_sr_eigenvectors(path):
    return np.load(path)


def compute_embedding(data_matrix, n_components=3, n_neighbors=10):
    iso = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    return iso.fit_transform(data_matrix)  # returns (n_samples x n_components)


def compute_persistent_homology(point_cloud, maxdim=2):
    result = ripser(point_cloud, maxdim=maxdim)
    return result["dgms"]  # list: [H0, H1, H2, ...]


def compute_null_lifetimes(
    data_matrix, n_null=200, n_components=3, n_neighbors=10, maxdim=2
):
    null_lf = {dim: [] for dim in range(maxdim + 1)}
    for _ in range(n_null):
        shuffled = np.random.permutation(data_matrix)
        emb = compute_embedding(shuffled, n_components, n_neighbors)
        dgms = compute_persistent_homology(emb, maxdim)
        for dim, dgm in enumerate(dgms):
            if dgm.size == 0:
                continue
            lifetimes = dgm[:, 1] - dgm[:, 0]
            finite_lts = lifetimes[np.isfinite(lifetimes)]
            null_lf[dim].extend(finite_lts.tolist())
    return null_lf


def find_robust_holes(dgms, null_lf, percentile=99.9):
    thresholds = {}
    robust = {}
    for dim, dgm in enumerate(dgms):
        lf_null = np.array(null_lf.get(dim, []))
        lf_null = lf_null[np.isfinite(lf_null)]
        if lf_null.size > 0:
            th = np.percentile(lf_null, percentile)
        else:
            th = np.inf
        thresholds[dim] = th

        lifetimes = dgm[:, 1] - dgm[:, 0] if dgm.size else np.array([])
        mask = np.isfinite(lifetimes) & (lifetimes > th)
        robust[dim] = dgm[mask]
    return thresholds, robust


def load_topology_graphs(
    base_path: str, ratio_max: int = 10, index_max: int = 100
) -> dict:
    topology_graphs = {}
    for ratio in range(1, ratio_max + 1):
        for idx in range(1, index_max + 1):
            path = f"{base_path}/{ratio}_{idx}.graphml"
            g = nx.MultiGraph(nx.read_graphml(path))
            topology_graphs[(ratio, idx)] = g
    return topology_graphs


def _init_worker(graphs):
    global topology_graphs_global
    topology_graphs_global = graphs


def _worker_isomorphic(pair):
    """
    Worker function for checking isomorphism between two graphs.
    Uses the global topology_graphs_global set via initializer.
    """
    i1, i2 = pair
    # Convert overall indices (0-based) to (ratio, index)
    r1, idx1 = divmod(i1, 100)
    r2, idx2 = divmod(i2, 100)
    # Adjust from 0-based to 1-based indexing
    r1 += 1
    idx1 += 1
    r2 += 1
    idx2 += 1

    g1 = topology_graphs_global[(r1, idx1)]
    g2 = topology_graphs_global[(r2, idx2)]

    # Skip pairs where either graph has only one node
    if g1.number_of_nodes() == 1 or g2.number_of_nodes() == 1:
        return None

    # Check isomorphism
    if nx.is_isomorphic(g1, g2):
        return ("isomorphic", pair)
    else:
        return ("non_isomorphic", pair)


def classify_isomorphic_pairs(pairs: list, topology_graphs: dict) -> (list, list):
    """
    Classify each pair of environment graphs as isomorphic or non-isomorphic.

    Returns two lists of tuple pairs: (isomorphic_pairs, non_isomorphic_pairs).

    - pairs: list of (overall_idx1, overall_idx2)
    - topology_graphs: dict mapping (ratio, index) to NetworkX graph
    """
    # Initialize worker pool with global graphs
    with mp.Pool(
        mp.cpu_count(), initializer=_init_worker, initargs=(topology_graphs,)
    ) as pool:
        results = pool.map(_worker_isomorphic, pairs)

    iso_pairs = []
    non_iso_pairs = []
    for res in results:
        if res is None:
            continue
        cat, pr = res
        if cat == "isomorphic":
            iso_pairs.append(pr)
        else:
            non_iso_pairs.append(pr)

    return iso_pairs, non_iso_pairs


def is_subgraph(graph_1, graph_2):
    GM = isomorphism.GraphMatcher(graph_1, graph_2)
    subgraph_alignments = [mapping for mapping in GM.subgraph_isomorphisms_iter()]
    if len(subgraph_alignments) > 0:
        return True
    GM = isomorphism.GraphMatcher(graph_2, graph_1)
    subgraph_alignments = [mapping for mapping in GM.subgraph_isomorphisms_iter()]
    if len(subgraph_alignments) > 0:
        return True
    return False


def process_pair(pair, topology_graph_dict, gh_dist_mat, gw_dist_mat, skel_gw_dist_mat):
    overall_index_1, overall_index_2 = pair
    ratio_index_1 = (overall_index_1) // 100 + 1
    index_1 = (overall_index_1) % 100 + 1
    ratio_index_2 = (overall_index_2) // 100 + 1
    index_2 = (overall_index_2) % 100 + 1

    graph_1 = topology_graph_dict[(ratio_index_1, index_1)]
    graph_2 = topology_graph_dict[(ratio_index_2, index_2)]

    gh_distance = gh_dist_mat[overall_index_1, overall_index_2]
    gw_distance = gw_dist_mat[overall_index_1, overall_index_2]
    skel_gw_distance = skel_gw_dist_mat[overall_index_1, overall_index_2]

    if gh_distance >= 0 and gw_distance >= 0 and not np.isnan(skel_gw_distance):
        if nx.is_isomorphic(graph_1, graph_2):
            return ("isomorphic", (overall_index_1, overall_index_2))
        else:
            if is_subgraph(graph_1, graph_2):
                return ("subgraph", (overall_index_1, overall_index_2))
            else:
                return ("non_isomorphic", (overall_index_1, overall_index_2))
    return None


def process_pair_wo_sub(
    pair, topology_graph_dict, gh_dist_mat, gw_dist_mat, skel_gw_dist_mat
):
    overall_index_1, overall_index_2 = pair
    ratio_index_1 = (overall_index_1) // 100 + 1
    index_1 = (overall_index_1) % 100 + 1
    ratio_index_2 = (overall_index_2) // 100 + 1
    index_2 = (overall_index_2) % 100 + 1

    graph_1 = topology_graph_dict[(ratio_index_1, index_1)]
    graph_2 = topology_graph_dict[(ratio_index_2, index_2)]

    gh_distance = gh_dist_mat[overall_index_1, overall_index_2]
    gw_distance = gw_dist_mat[overall_index_1, overall_index_2]
    skel_gw_distance = skel_gw_dist_mat[overall_index_1, overall_index_2]

    if gh_distance >= 0 and gw_distance >= 0 and not np.isnan(skel_gw_distance):
        if nx.is_isomorphic(graph_1, graph_2):
            return ("isomorphic", (overall_index_1, overall_index_2))
        else:
            return ("non_isomorphic", (overall_index_1, overall_index_2))
    return None


def identify_env_pairs_basis10_train(
    gh_sorted_pairs, gh_dist_mat, gw_dist_mat, skel_gw_dist_mat
):
    topology_graph_dict = {}
    for ratio_index in range(1, 11):
        for index in range(1, 101):
            topology_graph = nx.MultiGraph(
                nx.read_graphml(
                    f"data/skeletonize/skeleton_graph/train/{ratio_index}_{index}.graphml"
                )
            )
            topology_graph_dict[(ratio_index, index)] = topology_graph

    process_pair_partial = partial(
        process_pair,
        topology_graph_dict=topology_graph_dict,
        gh_dist_mat=gh_dist_mat,
        gw_dist_mat=gw_dist_mat,
        skel_gw_dist_mat=skel_gw_dist_mat,
    )

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_pair_partial, gh_sorted_pairs)

    isomorphic_pairs = []
    subgraph_pairs = []
    non_isomorphic_pairs = []

    for result in results:
        if result:
            category, pair = result
            if category == "isomorphic":
                isomorphic_pairs.append(pair)
            elif category == "subgraph":
                subgraph_pairs.append(pair)
            elif category == "non_isomorphic":
                non_isomorphic_pairs.append(pair)

    return isomorphic_pairs, subgraph_pairs, non_isomorphic_pairs


def identify_env_pairs_basis10_train_wo_subgraph(
    gh_sorted_pairs, gh_dist_mat, gw_dist_mat, skel_gw_dist_mat
):
    topology_graph_dict = {}
    for ratio_index in range(1, 11):
        for index in range(1, 101):
            topology_graph = nx.MultiGraph(
                nx.read_graphml(
                    f"data/skeletonize/skeleton_graph/train/{ratio_index}_{index}.graphml"
                )
            )
            topology_graph_dict[(ratio_index, index)] = topology_graph

    process_pair_partial = partial(
        process_pair_wo_sub,
        topology_graph_dict=topology_graph_dict,
        gh_dist_mat=gh_dist_mat,
        gw_dist_mat=gw_dist_mat,
        skel_gw_dist_mat=skel_gw_dist_mat,
    )

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_pair_partial, gh_sorted_pairs)

    isomorphic_pairs = []
    non_isomorphic_pairs = []

    for result in results:
        if result:
            category, pair = result
            if category == "isomorphic":
                isomorphic_pairs.append(pair)
            elif category == "non_isomorphic":
                non_isomorphic_pairs.append(pair)

    return isomorphic_pairs, non_isomorphic_pairs


def return_adjacent_nodes(state, edge_nodes, edge_corresp):
    for e_node, e_corresp in zip(edge_nodes, edge_corresp):
        if state in e_node:
            return e_corresp
    return None


def create_matchers(
    G1,
    G2,
    vertex_nodes_G1,
    vertex_corresp_G1,
    deadend_nodes_G1,
    deadend_corresp_G1,
    vertex_nodes_G2,
    vertex_corresp_G2,
    deadend_nodes_G2,
    deadend_corresp_G2,
    permutation_matrices,
):
    matchers = []

    for perm_matrix in permutation_matrices:
        matcher = {}

        # Convert permutation matrix to a node mapping between G1 and G2
        node_list_G1 = list(G1.nodes())
        node_list_G2 = list(G2.nodes())
        node_mapping = {
            node_list_G1[i]: node_list_G2[j] for i, j in zip(*np.nonzero(perm_matrix))
        }

        # Create matcher for vertex nodes
        for v_group_G1, v_corresp_G1 in zip(vertex_nodes_G1, vertex_corresp_G1):
            mapped_nodes_G2 = node_mapping[v_corresp_G1]
            for idx, nodes_G2 in enumerate(vertex_corresp_G2):
                if mapped_nodes_G2 == nodes_G2:
                    matcher[tuple(v_group_G1)] = tuple(vertex_nodes_G2[idx])
                    break

        # Create matcher for dead-end nodes
        for d_group_G1, d_corresp_G1 in zip(deadend_nodes_G1, deadend_corresp_G1):
            mapped_nodes_G2 = node_mapping[d_corresp_G1]
            for idx, nodes_G2 in enumerate(deadend_corresp_G2):
                if mapped_nodes_G2 == nodes_G2:
                    matcher[tuple(d_group_G1)] = tuple(deadend_nodes_G2[idx])
                    break

        matchers.append(matcher)

    return matchers


def generate_permutation_matrices(G1, G2):
    """
    A1_permuted = P.T @ A1 @ P
    """
    if isinstance(G1, nx.MultiGraph) and isinstance(G2, nx.MultiGraph):
        GM = nx.isomorphism.MultiGraphMatcher(G1, G2)
    else:
        GM = nx.isomorphism.GraphMatcher(G1, G2)

    permutation_matrices = []
    node_list_G1 = list(G1.nodes())
    node_list_G2 = list(G2.nodes())

    for mapping in GM.isomorphisms_iter():
        n = len(G1.nodes())
        permutation_matrix = np.zeros((n, n))

        # Correctly fill the permutation matrix
        for i, node in enumerate(node_list_G1):
            j = node_list_G2.index(mapping[node])
            permutation_matrix[i, j] = 1

        permutation_matrices.append(permutation_matrix)

    return permutation_matrices


def extract_path_dist(G):
    shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(G))

    # Step 1: Initialize a matrix to store the shortest distances
    num_nodes = len(G.nodes)
    distance_matrix = np.full((num_nodes, num_nodes), np.inf)

    # Step 2: Populate the matrix with the shortest path lengths
    for i, lengths in shortest_path_lengths.items():
        for j, length in lengths.items():
            distance_matrix[i, j] = length

    return distance_matrix


def extract_euclidean_dist(grid):
    num_states = grid.shape[0]
    distance_matrix = np.zeros((num_states, num_states))

    for i in range(num_states):
        for j in range(num_states):
            distance_matrix[i, j] = np.linalg.norm(grid[i] - grid[j])

    return distance_matrix


def calc_gh_dist_graph(G1, G2):
    """topology distance metric: Gromov-Hausdorff distance between the two graphs"""
    dist_mat_1 = extract_path_dist(G1)
    dist_mat_2 = extract_path_dist(G2)
    return dgh.upper(dist_mat_1, dist_mat_2, iter_budget=1000)


def calc_gh_dist_grid(grid_1, grid_2):
    """topology distance metric: Gromov-Hausdorff distance between the two graphs"""
    dist_mat_1 = extract_euclidean_dist(grid_1)
    dist_mat_2 = extract_euclidean_dist(grid_2)
    return dgh.upper(dist_mat_1, dist_mat_2, iter_budget=1000)


def avg_nodewise(matrix, vertex_nodes, deadend_nodes):
    """
    row: state, column: dim
    """
    averages = []

    if len(vertex_nodes) == 0 and len(deadend_nodes) == 0:
        return np.mean(matrix, axis=0, keepdims=True)

    # Process vertex_nodes
    for nodes in vertex_nodes:
        node_matrix = matrix[nodes, :]
        node_avg = np.mean(node_matrix, axis=0)
        averages.append(node_avg)

    # Process deadend_nodes
    for nodes in deadend_nodes:
        node_matrix = matrix[nodes, :]
        node_avg = np.mean(node_matrix, axis=0)
        averages.append(node_avg)

    return np.array(averages)


def cluster_states(states, num_columns):
    G = nx.Graph()

    # Add nodes
    for state in states:
        G.add_node(state)

    # Add edges based on connectivity
    for state in states:
        r, c = state // num_columns, state % num_columns
        neighbors = [
            (r - 1, c),  # Up
            (r + 1, c),  # Down
            (r, c - 1),  # Left
            (r, c + 1),  # Right
        ]
        for nr, nc in neighbors:
            neighbor_state = nr * num_columns + nc
            if (nr, nc) != (r, c) and neighbor_state in states:
                G.add_edge(state, neighbor_state)

    # Find connected components (clusters)
    clusters = [list(x) for x in list(nx.connected_components(G))]

    return clusters
