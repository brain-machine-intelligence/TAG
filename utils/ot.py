import numpy as np
from pyparsing import NotAny
from scipy.spatial import procrustes
import ot
from itertools import product
from scipy.optimize import minimize
from tqdm import tqdm
import networkx as nx
from scipy.linalg import svd
from scipy.spatial.distance import cdist


def extract_weighted_path_distances(graph):
    """
    Extract the shortest path distance matrix from the multigraph using edge weights.
    """
    # Compute shortest path lengths with edge weights
    shortest_path_lengths = dict(
        nx.all_pairs_dijkstra_path_length(graph, weight="weight")
    )

    # Create distance matrix
    num_nodes = len(graph.nodes)
    distance_matrix = np.full((num_nodes, num_nodes), np.inf)

    # Fill the matrix with shortest path distances
    for i, lengths in shortest_path_lengths.items():
        for j, length in lengths.items():
            distance_matrix[list(graph.nodes).index(i), list(graph.nodes).index(j)] = (
                length
            )

    return distance_matrix


def extract_edge_count_features(graph):
    """
    Extract the number of edges between node pairs in the multigraph as features.
    """
    num_nodes = len(graph.nodes)
    feature_matrix = np.zeros((num_nodes, num_nodes))

    for u, v, _ in graph.edges(keys=True):
        feature_matrix[list(graph.nodes).index(u), list(graph.nodes).index(v)] += 1

    return feature_matrix


def compute_feature_distance_matrix(features1, features2, metric="euclidean"):
    """
    Compute the pairwise distance matrix between features of nodes in graph 1 and graph 2.

    Parameters:
    - features1: Feature matrix for graph 1 (n1 x d).
    - features2: Feature matrix for graph 2 (n2 x d).
    - metric: Distance metric to use (e.g., 'euclidean', 'manhattan', etc.).

    Returns:
    - M: n1 x n2 matrix of pairwise feature distances.
    """
    from scipy.spatial.distance import cdist

    # Compute the pairwise distances between the feature vectors
    M = cdist(features1, features2, metric=metric)

    return M


def compute_gw_distance_graph(graph1, graph2):
    dist_matrix_1 = extract_weighted_path_distances(graph1)
    dist_matrix_2 = extract_weighted_path_distances(graph2)

    p1 = np.ones(dist_matrix_1.shape[0]) / dist_matrix_1.shape[0]
    p2 = np.ones(dist_matrix_2.shape[0]) / dist_matrix_2.shape[0]

    gw_dist, log = ot.gromov.gromov_wasserstein2(
        dist_matrix_1, dist_matrix_2, p1, p2, "square_loss", log=True
    )

    return gw_dist, log


def find_isomorphic_mapping(graph1, graph2):
    matcher = nx.isomorphism.GraphMatcher(graph1, graph2)
    if matcher.is_isomorphic():
        return matcher.mapping
    else:
        raise ValueError("Graphs are not isomorphic.")


def rotation_angle_from_matrix(R, criterion="sum"):
    """Compute the rotation angle from the rotation matrix R."""
    """# For 2D case, the angle can be directly obtained from the arccos of the trace
    # For higher dimensions, this gives a sense of the total rotation.
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Clip to avoid numerical issues
    return np.arccos(cos_theta)"""
    eigenvalues = np.linalg.eigvals(R)
    angles = np.angle(eigenvalues)
    angles = np.mod(angles, np.pi)
    angles = angles[angles > 1e-10]  # Filter out trivial angles near zero
    if criterion == "sum":
        return np.sum(angles)
    elif criterion == "max":
        return np.max(angles)
    elif criterion == "rms":
        return np.sqrt(np.mean(angles**2))
    else:
        raise ValueError("Unknown criterion")


def center_and_scale(matrix, predefined_norm=None):
    """
    Center and scale the given matrix as the Procrustes analysis does.

    Parameters:
    matrix (np.ndarray): The input matrix to be centered and scaled.

    Returns:
    np.ndarray: The centered and scaled matrix.
    """
    # Center the matrix by subtracting the mean of each column
    mean_matrix = np.mean(matrix, axis=0)
    centered_matrix = matrix - mean_matrix

    if predefined_norm is not None:
        scaled_matrix = centered_matrix / predefined_norm
        return scaled_matrix, predefined_norm
    else:
        # Compute the Frobenius norm of the centered matrix
        frobenius_norm = np.linalg.norm(centered_matrix, "fro")

        # Scale the matrix to have a Frobenius norm of 1
        if frobenius_norm > 0:
            scaled_matrix = centered_matrix / frobenius_norm
        else:
            scaled_matrix = centered_matrix
            frobenius_norm = 1

        return scaled_matrix, frobenius_norm


def center_and_scale_diffusionmap(
    matrix, direct_scale_factor=None, provided_norms=None
):
    """
    Center and scale the given matrix such that each dimension lies within [-1, 1] range,
    while preserving the ratio of norms between dimensions.

    Parameters:
    matrix (np.ndarray): The input matrix to be centered and scaled.
    direct_scale_factor (np.ndarray, optional): A vector for scaling each dimension directly.
    provided_norms (np.ndarray, optional): A vector of norms for each dimension. If given,
                                           it will be used instead of calculating norms.

    Returns:
    np.ndarray: The normalized manifold after scaling.
    np.ndarray: The direct scaling factors for each dimension.
    np.ndarray: The norms used for each dimension.
    """
    # Step 1: Center the manifold symmetrically for each dimension
    min_vals = np.min(matrix, axis=0)
    max_vals = np.max(matrix, axis=0)
    centered_manifold = matrix - (min_vals + (max_vals - min_vals) / 2)

    # Step 2: Divide each dimension by its range to fit within [-1, 1]
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1  # Avoid division by zero for constant dimensions
    normalized_manifold = centered_manifold / ranges

    # Step 3: If direct scaling factor is provided, apply it directly
    if direct_scale_factor is not None:
        scaled_manifold = centered_manifold * direct_scale_factor
        return (
            scaled_manifold,
            direct_scale_factor,
            provided_norms or np.linalg.norm(centered_manifold, axis=0),
        )

    # Step 4: Use provided norms if given, otherwise calculate them
    norms = (
        provided_norms
        if provided_norms is not None
        else np.linalg.norm(centered_manifold, axis=0)
    )

    # Step 5: Calculate scaling factors as norms / reference_norm
    reference_norm = np.max(norms)  # Use the maximum norm as the reference
    scaling_factors = norms / reference_norm  # Direct proportional scaling

    # Step 6: Apply the scaling factors to the normalized manifold
    normalized_manifold *= scaling_factors

    # Step 7: Compute the ratio of the range between normalized and centered manifolds
    new_ranges = np.max(
        np.abs(normalized_manifold), axis=0
    )  # Ranges after normalization
    direct_scale_factor = new_ranges / ranges  # Ratio between new and original ranges

    return (
        normalized_manifold,
        direct_scale_factor,
        norms,
    )  # 최종 manifold, centering만 하고 바로 normalize할 수 있는 것, 각 차원별 norm (sqrt eigenvalues)


def procrustes_manual(X, Y):
    H = X.T @ Y
    U, S, Vt = svd(H)
    R = Vt.T @ U.T

    # Ensure a proper rotation (det(R) = 1, not -1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def procrustes_without_connectivity(X, Y, original_states=None):
    # Step 1: Center and scale the matrices
    X_scaled, X_scale = center_and_scale(X)
    Y_scaled, Y_scale = center_and_scale(Y)

    # Step 2: Compute the optimal rotation using SVD
    if original_states is None:
        R = procrustes_manual(X_scaled, Y_scaled)
    else:
        R = procrustes_manual(X_scaled, Y_scaled[original_states])

    # Step 3: Rotate the points
    Y_rotated = Y_scaled @ R

    return X_scaled, Y_rotated, R, X_scale, Y_scale


def procrustes_with_connectivity(X, Y, graph1, graph2, corresp1, corresp2):
    # Step 1: Find isomorphic mapping ensuring connectivity is preserved
    corresp_list_1 = []
    corresp_list_1.extend(corresp1[0])
    corresp_list_1.extend(corresp1[1])
    corresp_list_2 = []
    corresp_list_2.extend(corresp2[0])
    corresp_list_2.extend(corresp2[1])

    if len(corresp_list_1) == 0 or len(corresp_list_2) == 0:
        Y_reordered = Y
    else:
        mapping = find_isomorphic_mapping(graph1, graph2)
        index_mapping = [-1 for _ in range(len(X))]
        for k, v in mapping.items():
            index_mapping[corresp_list_1.index(k)] = corresp_list_2.index(v)
        # Reorder Y to match the order of X based on the mapping
        Y_reordered = np.array([Y[index_mapping[i]] for i in range(len(X))])

    # Step 2: Center and scale the matrices
    X_scaled, X_scale = center_and_scale(X)
    Y_scaled, Y_scale = center_and_scale(Y_reordered)

    # Step 3: Compute the optimal rotation using SVD
    R = procrustes_manual(X_scaled, Y_scaled)

    # Step 4: Rotate the points
    Y_rotated = Y_scaled @ R

    return X_scaled, Y_rotated, R, X_scale, Y_scale


def sign_corrected_procrustes(
    X,
    Y,
    graph1=None,
    graph2=None,
    corresp1=None,
    corresp2=None,
    tol=1e-9,
    original_states=None,
):
    """
    X, Y 점 개수가 다르면 작동하지 않음
    """
    # Number of vectors
    n_vectors = Y.shape[1]

    # Initialize the best distance as infinity
    best_d = np.inf
    best_mtx2_wo_rotation = None
    best_mtx2 = None
    best_sign_configuration = None
    best_optimal_rotation_matrix = None
    min_rotation_angle = np.inf

    # Generate all possible sign configurations for n_vectors eigenvectors
    sign_configurations = np.array(list(product([1, -1], repeat=n_vectors)))

    for sign_config in sign_configurations:
        # Apply sign changes to Y
        Y_signed = Y * sign_config

        # Procrustes alignment
        if graph1 is None or (
            graph1 is not None and X.shape[0] != graph1.number_of_nodes()
        ):  # not a skeleton
            X_scaled, Y_rotated, R, X_scale, Y_scale = procrustes_without_connectivity(
                X, Y_signed, original_states
            )
        else:  # skeleton
            assert graph1 is not None and graph2 is not None
            X_scaled, Y_rotated, R, X_scale, Y_scale = procrustes_with_connectivity(
                X, Y_signed, graph1, graph2, corresp1, corresp2
            )

        # Calculate the disparity (sum of squared distances between aligned points)
        if original_states is None:
            disparity = np.sum((X_scaled - Y_rotated) ** 2)
        else:
            disparity = np.sum((X_scaled - Y_rotated[original_states]) ** 2)

        # Calculate the rotation angle from the rotation matrix
        rotation_angle = rotation_angle_from_matrix(R)

        # Update if we find a better alignment
        if disparity < best_d or (
            abs(disparity - best_d) <= tol and rotation_angle < min_rotation_angle
        ):
            best_d = disparity
            best_mtx2_wo_rotation = Y_signed
            best_mtx2 = Y_rotated
            best_sign_configuration = sign_config
            best_optimal_rotation_matrix = R
            min_rotation_angle = rotation_angle

    return (
        X_scaled,
        best_mtx2,
        best_mtx2_wo_rotation,
        best_d,
        best_sign_configuration,
        best_optimal_rotation_matrix,
        X_scale,
        Y_scale,
    )


def sign_corrected_rotatory_ot_cost_optimization(
    X, Y, node_cluster_1=None, node_cluster_2=None
):
    # Number of vectors
    n_vectors = Y.shape[1]

    # Initialize the best distance as infinity
    best_cost = np.inf
    best_mtx2 = None

    X_normalized, X_scale = center_and_scale(X)
    Y_normalized, Y_scale = center_and_scale(Y)

    # Generate all possible sign configurations for n_vectors eigenvectors
    sign_configurations = np.array(list(product([1, -1], repeat=n_vectors)))

    for sign_config_index in tqdm(range(len(sign_configurations))):
        sign_config = sign_configurations[sign_config_index]
        # Apply sign changes to Y
        Y_signed = Y_normalized * sign_config

        # Procrustes alignment
        initial_rotation_matrix = np.eye(n_vectors).flatten()
        result = minimize(
            rotatory_ot_cost,
            initial_rotation_matrix,
            args=(X_normalized, Y_signed, node_cluster_1, node_cluster_2),
            method="BFGS",
        )
        if result.success:
            optimal_rotation_matrix = result.x.reshape(n_vectors, n_vectors)
            optimal_rotation_matrix, _ = np.linalg.qr(optimal_rotation_matrix)

            transport_cost_val = rotatory_ot_cost(
                optimal_rotation_matrix,
                X_normalized,
                Y_signed,
                node_cluster_1,
                node_cluster_2,
            )

            if transport_cost_val < best_cost:
                best_transport_cost = transport_cost_val
                best_sign_configuration = sign_config
                best_optimal_rotation_matrix = optimal_rotation_matrix
                m2_normalized, m2_scale = center_and_scale(Y_signed)
                m2_rotated = m2_normalized.dot(optimal_rotation_matrix.T)
                best_mtx2 = m2_rotated

    assert best_mtx2 is not None
    return (
        X_normalized,
        best_mtx2,
        best_transport_cost,
        best_sign_configuration,
        best_optimal_rotation_matrix,
    )


def rotatory_ot_cost(
    flat_rotation_matrix, m1, m2, node_cluster_1=None, node_cluster_2=None
):
    dim = m1.shape[1]
    # Reshape the flat rotation matrix to its original shape
    rotation_matrix = flat_rotation_matrix.reshape(dim, dim)

    # Ensure the matrix is orthogonal
    rotation_matrix, _ = np.linalg.qr(rotation_matrix)

    m2_rotated = m2.dot(rotation_matrix.T)
    return calc_ot_cost(m1.real, m2_rotated.real, node_cluster_1, node_cluster_2)


def calc_ot_cost(m1, m2, node_cluster_1=None, node_cluster_2=None):
    """manifold input as numpy"""
    high_cost = 1e9
    cost_matrix_original = ot.dist(m1, m2, metric="euclidean")
    if node_cluster_1 is None or node_cluster_2 is None:
        cost_matrix = cost_matrix_original
    else:
        cost_matrix = np.ones(cost_matrix_original.shape) * high_cost
        for list_1, list_2 in zip(node_cluster_1, node_cluster_2):
            for i in list_1:
                for j in list_2:
                    cost_matrix[i, j] = cost_matrix_original[i, j]
    num_points_1 = m1.shape[0]
    num_points_2 = m2.shape[0]
    weights_1 = np.ones((num_points_1,)) / num_points_1
    weights_2 = np.ones((num_points_2,)) / num_points_2
    transport_cost = ot.emd2(weights_1, weights_2, cost_matrix)
    return transport_cost


def calc_ot_post_procrustes(
    m1,
    m2,
    graph1=None,
    graph2=None,
    corresp1=None,
    corresp2=None,
    node_cluster_1=None,
    node_cluster_2=None,
    original_states=None,
):
    """
    manifold_1이 원래
    manifold_2가 manifold_1에 fit 되는 것
    row: num_points, column: num_dim
    """
    if node_cluster_1 is not None or node_cluster_2 is not None:
        raise NotImplementedError("Transporting node-wise not yet implemented.")
    (
        m1_stand,
        m2_trans,
        m2_stand,
        disparity,
        best_sign_configuration,
        best_optimal_rotation_matrix,
        m1_scale,
        m2_scale,
    ) = sign_corrected_procrustes(
        m1, m2, graph1, graph2, corresp1, corresp2, original_states=original_states
    )
    ot_cost = calc_ot_cost(m1_stand, m2_trans, node_cluster_1, node_cluster_2)
    return (
        m1_stand,
        m2_trans,
        m2_stand,
        ot_cost,
        disparity,
        best_sign_configuration,
        best_optimal_rotation_matrix,
        m1_scale,
        m2_scale,
    )


def calc_ot_without_rotation(m1, m2):
    """
    Calculate OT cost using manual sign correction, scaling, and centering.
    Shift - baseline용
    """

    # Center and scale m1 and m2
    m1_normalized, m1_scale = center_and_scale(m1)
    m2_normalized, m2_scale = center_and_scale(m2)

    # Number of dimensions (columns) in m1 and m2
    n_vectors = m2_normalized.shape[1]

    # Initialize the best OT cost as infinity
    best_ot_cost = np.inf
    best_m2_trans = None
    best_sign_configuration = None

    # Generate all possible sign configurations for n_vectors
    sign_configurations = np.array(list(product([1, -1], repeat=n_vectors)))

    for sign_config in sign_configurations:
        # Apply sign changes to m2_scaled
        m2_signed = m2_normalized * sign_config

        # Calculate OT cost
        ot_cost = calc_ot_cost(m1_normalized, m2_signed, None, None)

        # Update if we find a better sign configuration
        if ot_cost < best_ot_cost:
            best_ot_cost = ot_cost
            best_m2_trans = m2_signed
            best_sign_configuration = sign_config

    return best_ot_cost


"""
X_scaled, best_mtx2, best_d, best_sign_configuration, best_optimal_rotation_matrix, X_scale, Y_scale
"""


def calc_minimum_ot(m1, m2, node_cluster_1=None, node_cluster_2=None):
    if m1.shape[0] == m2.shape[0]:
        if m1.shape[0] == 1:
            return m1, m2, 0, np.array([1]), np.array([[0]])
        (
            X_normalized,
            best_mtx2,
            best_transport_cost,
            best_sign_configuration,
            best_optimal_rotation_matrix,
        ) = calc_ot_post_procrustes(m1, m2, node_cluster_1, node_cluster_2)
    else:
        (
            X_normalized,
            best_mtx2,
            best_transport_cost,
            best_sign_configuration,
            best_optimal_rotation_matrix,
        ) = sign_corrected_rotatory_ot_cost_optimization(
            m1, m2, node_cluster_1, node_cluster_2
        )
    return (
        X_normalized,
        best_mtx2,
        best_transport_cost,
        best_sign_configuration,
        best_optimal_rotation_matrix,
    )


def filter_non_zero_samples(X):
    """
    Filter out rows in X that are close to zero (based on a small threshold)
    and return the filtered array along with the original indices.

    Parameters:
    - X: numpy array of shape (n_samples_X, n_features)

    Returns:
    - filtered_X: numpy array with non-zero rows retained
    - original_indices: array of original indices of the retained rows
    """
    epsilon = 1e-10  # Define a small threshold value
    non_zero_mask = ~np.all(np.abs(X) < epsilon, axis=1)
    filtered_X = X[non_zero_mask]
    original_indices = np.where(non_zero_mask)[0]
    return filtered_X, original_indices


def compute_pairwise_distance_matrix(X, Y=None):
    """
    Compute the pairwise Euclidean distance matrix between X and Y.
    If Y is None, compute the pairwise distances within X.
    """
    if Y is None:
        return np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)
    else:
        return np.linalg.norm(X[:, np.newaxis] - Y[np.newaxis, :], axis=2)


def construct_cost_matrix(X, Y, matcher):
    """
    Constructs the cost matrix for the optimal transport plan based on the matcher.

    Parameters:
    - X: Dataset X (n_samples_X, n_features)
    - Y: Dataset Y (n_samples_Y, n_features)
    - matcher: Dictionary mapping state indices in X to state indices in Y

    Returns:
    - M: The cost matrix for the selected states
    - selected_indices_X: List of selected indices in X
    - selected_indices_Y: List of selected indices in Y
    """
    selected_points_X = []
    selected_points_Y = []

    for x_indices, y_indices in matcher.items():
        selected_points_X.extend(x_indices)
        selected_points_Y.extend(y_indices)

    # Remove duplicates and sort the indices
    selected_points_X = sorted(set(selected_points_X))
    selected_points_Y = sorted(set(selected_points_Y))

    # Construct the cost matrix
    M = np.zeros((len(selected_points_X), len(selected_points_Y)))
    for i, x_idx in enumerate(selected_points_X):
        for j, y_idx in enumerate(selected_points_Y):
            M[i, j] = np.linalg.norm(X[x_idx] - Y[y_idx])

    return M, selected_points_X, selected_points_Y


def calc_gromov_wasserstein_dist_alignment(X, Y, loss_fun="square_loss"):
    """
    Find the optimal transport plan using Gromov-Wasserstein distance.

    Parameters:
    - X: numpy array of shape (n_samples_X, n_features)
    - Y: numpy array of shape (n_samples_Y, n_features)
    - loss_fun: loss function to use, default is "square_loss"

    Returns:
    - gw_dist: Gromov-Wasserstein distance
    - T: Optimal transport plan
    - X_selected: Filtered and selected samples from X
    - Y_selected: Filtered and selected samples from Y
    - selected_points_X: Original indices of selected points in X
    - selected_points_Y: Original indices of selected points in Y
    """
    # Filter out samples with all zero values and track original indices
    if X.shape[0] > 1:
        X_filtered, original_indices_X = filter_non_zero_samples(X)
    else:
        X_filtered = X
        original_indices_X = np.array([0])
    if Y.shape[0] > 1:
        Y_filtered, original_indices_Y = filter_non_zero_samples(Y)
    else:
        Y_filtered = Y
        original_indices_Y = np.array([0])

    # Compute the pairwise distance matrices
    D_X = compute_pairwise_distance_matrix(X_filtered)
    D_Y = compute_pairwise_distance_matrix(Y_filtered)

    # Compute GW distance without matcher constraints
    p = np.ones(X_filtered.shape[0]) / X_filtered.shape[0]
    q = np.ones(Y_filtered.shape[0]) / Y_filtered.shape[0]
    gw_dist, log = ot.gromov.gromov_wasserstein2(D_X, D_Y, p, q, loss_fun, log=True)

    # Return all the selected points as original indices
    selected_points_X = original_indices_X
    selected_points_Y = original_indices_Y
    X_selected = X_filtered
    Y_selected = Y_filtered

    return (
        gw_dist,
        log["T"],
        X_selected,
        Y_selected,
        selected_points_X,
        selected_points_Y,
    )


def calc_gromov_wasserstein_dist_graph(G1, G2, loss_fun="square_loss"):
    # Filter out samples with all zero values and track original indices

    # Compute the pairwise distance matrices
    D_X = extract_weighted_path_distances(G1)
    D_Y = extract_weighted_path_distances(G2)

    # Compute GW distance without matcher constraints
    p = np.ones(G1.number_of_nodes()) / G1.number_of_nodes()
    q = np.ones(G2.number_of_nodes()) / G2.number_of_nodes()
    gw_dist, log = ot.gromov.gromov_wasserstein2(D_X, D_Y, p, q, loss_fun, log=True)

    return (
        gw_dist,
        log["T"],
    )


def calc_transport_dist_alignment(X, Y, matcher):
    """
    Find the optimal transport plan using a predefined cost matrix when matcher is provided.

    Parameters:
    - X: numpy array of shape (n_samples_X, n_features)
    - Y: numpy array of shape (n_samples_Y, n_features)
    - matcher: Dictionary mapping state indices in X to state indices in Y

    Returns:
    - ot_dist: The optimal transport cost
    - T: Optimal transport plan
    - X_selected: Filtered and selected samples from X
    - Y_selected: Filtered and selected samples from Y
    - selected_points_X: Original indices of selected points in X
    - selected_points_Y: Original indices of selected points in Y
    """
    # Filter out samples with all zero values and track original indices
    """X_filtered, original_indices_X = filter_non_zero_samples(X)
    Y_filtered, original_indices_Y = filter_non_zero_samples(Y)

    # Adjust matcher indices according to filtered data
    adjusted_matcher = {
        tuple(np.where(original_indices_X == i)[0][0] for i in x_indices): tuple(
            np.where(original_indices_Y == j)[0][0] for j in y_indices
        )
        for x_indices, y_indices in matcher.items()
    }"""

    # Construct the cost matrix and get selected states
    M, selected_points_X, selected_points_Y = construct_cost_matrix(X, Y, matcher)

    # Map selected indices back to original indices
    """selected_points_X = original_indices_X[selected_points_X_filtered]
    selected_points_Y = original_indices_Y[selected_points_Y_filtered]"""

    # Retrieve the selected samples from the original data
    X_selected = X[selected_points_X]
    Y_selected = Y[selected_points_Y]

    # Create the corresponding distributions for the selected states
    p = np.ones(M.shape[0]) / M.shape[0]
    q = np.ones(M.shape[1]) / M.shape[1]

    # Compute the optimal transport plan using the cost matrix M
    T = ot.emd(p, q, M)
    ot_dist = np.sum(T * M)

    return ot_dist, T, X_selected, Y_selected, selected_points_X, selected_points_Y


def interpolate_transport_adjacent(
    points_unlabeled,
    points1_labeled,
    points2_labeled,
    T,
    closest_nodes_dict,
    node_to_points_map,
    points1_index,
):
    """
    Interpolates the transport plan for unlabeled points using the transport plan matrix T.

    Parameters:
    - points_unlabeled: numpy array of shape (n_unlabeled, n_features)
    - points1_labeled: numpy array of shape (n_labeled, n_features)
    - points2_labeled: numpy array of shape (n_labeled, n_features)
    - T: numpy array of shape (n_labeled_X, n_labeled_Y), the transport plan matrix from GW
    - closest_nodes_dict: dictionary where keys are indices of points_unlabeled
      and values are tuples of the two closest node indices.
    - node_to_points_map: dictionary where keys are node indices and values are lists of indices
      of points associated with each node in points1_labeled.

    Returns:
    - interpolated_plan: numpy array of shape (n_unlabeled, n_features), interpolated transport plan
    """
    interpolated_plan = []

    for point in points_unlabeled:
        # Retrieve the indices of the two closest nodes from the dictionary
        node_idx1, node_idx2 = closest_nodes_dict[point]

        # Get the points associated with each node
        points_node1 = points1_labeled[
            np.where(np.isin(points1_index, node_to_points_map[node_idx1]))[0]
        ]
        points_node2 = points1_labeled[
            np.where(np.isin(points1_index, node_to_points_map[node_idx2]))[0]
        ]

        # Initialize the final transported point and total importance weight
        final_transport = np.zeros(points2_labeled.shape[1])
        total_importance_weight = 0.0

        # Loop over all combinations of points between the two nodes
        for idx1, point1 in enumerate(points_node1):
            for idx2, point2 in enumerate(points_node2):
                # Calculate distances from the unlabeled point to the current pair of points
                dist1 = np.linalg.norm(point - point1)
                dist2 = np.linalg.norm(point - point2)

                # Interpolation weights (based on inverse distance)
                weight1 = 1 / dist1 if dist1 != 0 else 1.0
                weight2 = 1 / dist2 if dist2 != 0 else 1.0
                total_weight = weight1 + weight2
                weight1 /= total_weight
                weight2 /= total_weight

                # Define an importance weight for this combination (e.g., product of weights)
                importance_weight = weight1 * weight2

                # Retrieve the corresponding transport plan for this pair using T
                transport_plan1 = T[
                    np.where(
                        np.isin(points1_index, node_to_points_map[node_idx1][idx1])
                    )[0],
                    :,
                ]
                transport_plan2 = T[
                    np.where(
                        np.isin(points1_index, node_to_points_map[node_idx2][idx2])
                    )[0],
                    :,
                ]

                # Calculate the weighted sum for the current combination
                combination_transport = importance_weight * (
                    weight1 * np.dot(transport_plan1, points2_labeled)
                    + weight2 * np.dot(transport_plan2, points2_labeled)
                )

                # Accumulate the result
                final_transport += np.squeeze(combination_transport)
                total_importance_weight += importance_weight
        assert total_importance_weight > 0

        # Normalize by the total importance weight to get the final interpolated transport
        final_transport /= total_importance_weight

        interpolated_plan.append(final_transport)

    return np.array(interpolated_plan)


if __name__ == "__main__":
    from modules.base import SRMB

    sr = SRMB(10, 10)
    sr.set_index("9_2")
    sr.load_map_structure("data/env/train/")
    eigenmap = sr.return_successormap()
    eigenmap_filtered = filter_non_zero_samples(eigenmap)
    pass
