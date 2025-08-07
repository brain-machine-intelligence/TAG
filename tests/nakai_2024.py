import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import warnings
from tqdm import tqdm
from scipy import stats
from scipy.sparse import SparseEfficiencyWarning
from modules.base import SRMB
from collections import deque
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import argparse
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'

# Suppress specific warnings that don't affect computation results
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.manifold._isomap")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.sparse._index")
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
warnings.filterwarnings("ignore", message=".*The number of connected components.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def gamma_to_scale(gamma):
    """
    Convert gamma value to equivalent scale value using scale = -1/ln(gamma)
    
    Parameters:
    - gamma: float, gamma value (0 < gamma < 1)
    
    Returns:
    - scale: float, equivalent scale value
    """
    if gamma <= 0 or gamma >= 1:
        raise ValueError("Gamma must be between 0 and 1 (exclusive)")
    return -1 / np.log(gamma)


def scale_to_gamma(scale):
    """
    Convert scale value to equivalent gamma value using gamma = exp(-1/scale)
    
    Parameters:
    - scale: float, scale value (scale > 0)
    
    Returns:
    - gamma: float, equivalent gamma value
    """
    if scale <= 0:
        raise ValueError("Scale must be positive")
    return np.exp(-1 / scale)


def create_gamma_equivalent_scales(gamma_values):
    """
    Create scale values that are equivalent to given gamma values
    
    Parameters:
    - gamma_values: list or array of gamma values
    
    Returns:
    - equivalent_scales: array of scale values equivalent to gamma_values
    """
    return np.array([gamma_to_scale(gamma) for gamma in gamma_values])


def shift_matrix(matrix, direction, n):
    """
    Shift all elements of a matrix by n positions in a specific direction.
    
    Parameters:
    matrix: numpy array - input matrix
    direction: str - direction ('right', 'down', 'left', 'up')
    n: int - number of positions to shift
    
    Returns:
    numpy array - transformed matrix
    """
    if n <= 0:
        return matrix.copy()
    
    rows, cols = matrix.shape
    result = np.zeros_like(matrix)
    
    if direction == 'right':
        # Shift right by n positions
        if n < cols:
            result[:, n:] = matrix[:, :-n]
        # If n >= cols, all elements shift out, so result remains filled with zeros
    
    elif direction == 'down':
        # Shift down by n positions
        if n < rows:
            result[n:, :] = matrix[:-n, :]
        # If n >= rows, all elements shift out, so result remains filled with zeros
    
    elif direction == 'left':
        # Shift left by n positions (from right to left)
        if n < cols:
            result[:, :-n] = matrix[:, n:]
        # If n >= cols, all elements shift out, so result remains filled with zeros
    
    elif direction == 'up':
        # Shift up by n positions (from bottom to top)
        if n < rows:
            result[:-n, :] = matrix[n:, :]
        # If n >= rows, all elements shift out, so result remains filled with zeros
    
    else:
        raise ValueError("direction must be one of: 'right', 'down', 'left', 'up'")
    
    return result


def extend_border(border, blocks, max_dist, num_row, num_column):
    """
    Extend border matrix to include n-step borders from 1 to max_dist.
    
    Args:
        border: num_states x 4 matrix indicating 1-step borders in each direction
        transition_matrix: num_states x num_states transition matrix
        max_dist: Maximum number of steps to consider
        
    Returns:
        extended_border: num_states x (4 * max_dist) matrix containing borders for all steps
    """
    num_states = border.shape[0]
    assert num_states == num_row * num_column
    extended_border = np.zeros((num_states, 4 * max_dist), dtype=bool)
    
    # Copy 1-step borders to the first 4 columns
    extended_border[:, :4] = border
    
    for i in range(1, max_dist - 1):
        for dir_idx, dir_str in enumerate(['right', 'left', 'down', 'up']):
            border_vector = border[:, dir_idx]
            border_matrix = border_vector.reshape(num_row, num_column)
            shifted_border_matrix = shift_matrix(border_matrix, dir_str, i)
            extended_border[:, 4 * i + dir_idx] = shifted_border_matrix.flatten()


    extended_border[blocks] = 0
    return extended_border


def extend_corner(corner, max_dim, seed=42):
    """
    Extend corner features from 8 dimensions to max_dim dimensions using random weighted sums.
    
    Parameters:
    - corner: array of shape (num_states, 8) - transformed corner features
    - max_dim: int - target dimension (must be >= 8)
    - seed: int - random seed for reproducibility
    
    Returns:
    - extended_corner: array of shape (num_states, max_dim)
    """
    if max_dim < 8:
        raise ValueError("max_dim must be >= 8")
    
    num_states, _ = corner.shape
    np.random.seed(seed)
    
    # Split max_dim into two halves (integer division)
    half_dim = max_dim // 2
    
    # Initialize extended corner
    extended_corner = np.zeros((num_states, max_dim))
    extended_corner[:, :4] = corner[:, :4]
    extended_corner[:, half_dim:half_dim + 4] = corner[:, 4:]
    
    # Process first half: use first 4 features with random weighted sum
    if half_dim > 0:
        # Generate random weights for first 4 features
        weights_first = np.random.random(4)
        weights_first = weights_first / np.sum(weights_first)  # Normalize to sum to 1
        
        # Create weighted combinations for first half
        for i in range(4, half_dim):
            # Generate random weights for this combination
            combo_weights = np.random.random(4)
            combo_weights = combo_weights / np.sum(combo_weights)
            
            # Apply weighted sum to first 4 features
            extended_corner[:, i] = np.dot(corner[:, :4], combo_weights)
    
    # Process second half: use last 4 features with random weighted sum
    if max_dim - half_dim > 0:
        # Generate random weights for last 4 features
        weights_second = np.random.random(4)
        weights_second = weights_second / np.sum(weights_second)  # Normalize to sum to 1
        
        # Create weighted combinations for second half
        for i in range(half_dim + 4, max_dim):
            # Generate random weights for this combination
            combo_weights = np.random.random(4)
            combo_weights = combo_weights / np.sum(combo_weights)
            
            # Apply weighted sum to last 4 features
            extended_corner[:, i] = np.dot(corner[:, 4:], combo_weights)
    
    return extended_corner


def rotate_2d_manifold(manifold_coords, degree):
    """
    Rotate 2D manifold coordinates by a given degree.
    
    Parameters:
    - manifold_coords: array of shape (N, 2), 2D coordinates
    - degree: float, rotation angle in degrees (positive for counterclockwise)
    
    Returns:
    - rotated_coords: array of shape (N, 2), rotated coordinates
    """
    import numpy as np
    
    # Convert degree to radians
    angle_rad = np.radians(degree)
    
    # Create rotation matrix
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_angle, -sin_angle],
                               [sin_angle, cos_angle]])
    
    # Apply rotation
    rotated_coords = manifold_coords @ rotation_matrix.T
    
    return rotated_coords




def transform_corner(corner):
    num_states, _ = corner.shape

    # Create an empty tensor for the transformed corner (new dimension is 8)
    transformed_corner = np.zeros((num_states, 8))

    for i in range(4):
        # If the element is -1, set 1 at index 4 + element_index
        transformed_corner[:, 4 + i] = (corner[:, i] == -1)

        # If the element is 1, set 1 at index element_index
        transformed_corner[:, i] = (corner[:, i] == 1)

    return transformed_corner

def build_graph(height, width, blocks):
    graph = {}
    for s in range(height * width):
        if s in blocks:
            continue
        r, c = divmod(s, width)
        neighbors = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width:
                t = rr * width + cc
                if t not in blocks:
                    neighbors.append(t)
        graph[s] = neighbors
    return graph

def multi_source_bfs(sources, n_states, graph):
    dist = np.full(n_states, np.inf)
    q = deque()
    for s in sources:
        dist[s] = 0
        q.append(s)
    while q:
        u = q.popleft()
        for v in graph.get(u, []):
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def fuzz_features(feature_matrix, height, width, blocks, scale=1.0):
    n_states, D = feature_matrix.shape
    graph = build_graph(height, width, set(blocks))
    fuzzy = np.zeros((n_states, D), dtype=float)
    for d in range(D):
        srcs_pos = np.where(feature_matrix[:, d] > 0)[0]
        srcs_neg = np.where(feature_matrix[:, d] < 0)[0]
        if len(srcs_pos) > 0:
            dist_pos = multi_source_bfs(srcs_pos, n_states, graph)
            fuzzy[:, d] += np.exp(-dist_pos / scale)
        if len(srcs_neg) > 0:
            dist_neg = multi_source_bfs(srcs_neg, n_states, graph)
            fuzzy[:, d] -= np.exp(-dist_neg / scale)
    return fuzzy

def shortest_path_length(graph, start, goal):
    """Calculate shortest path length between two states using BFS"""
    if start == goal:
        return 0
    
    queue = deque([(start, 0)])
    visited = set([start])
    
    while queue:
        current, length = queue.popleft()
        if current == goal:
            return length
        
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, length + 1))
    
    return float('inf')  # No path found

def calculate_geodesic_distances(nonblocks, blocks, height, width):
    """Calculate geodesic distances between all pairs of states"""
    graph = build_graph(height, width, set(blocks))
    n_states = len(nonblocks)
    geodesic_dists = np.zeros((n_states, n_states))
    
    for i, s1 in enumerate(nonblocks):
        for j, s2 in enumerate(nonblocks):
            if i != j:
                dist = shortest_path_length(graph, s1, s2)
                geodesic_dists[i, j] = dist
            else:
                geodesic_dists[i, j] = 0
    
    return geodesic_dists

def calculate_topology_preservation_geodesic(geodesic_dists, manifold_coords):
    """Calculate topology preservation using geodesic distances"""
    manifold_dists = pdist(manifold_coords)
    geodesic_dists_flat = geodesic_dists[np.triu_indices_from(geodesic_dists, k=1)]
    
    # Remove infinite distances (unreachable pairs)
    valid_mask = np.isfinite(geodesic_dists_flat)
    if np.sum(valid_mask) < 10:  # Need at least 10 valid pairs
        return np.nan
    
    valid_geodesic = geodesic_dists_flat[valid_mask]
    valid_manifold = manifold_dists[valid_mask]
    
    correlation, _ = spearmanr(valid_geodesic, valid_manifold)
    return correlation

def run_env_sr_topo_by_gamma_combined(args):
    """Run SR topology preservation analysis for a given gamma value"""
    gamma, ratio_index, index = args
    # Load SR data
    sr = SRMB(10, 10, gamma=gamma)
    sr.set_index(f"{ratio_index}_{index}")
    loaded = sr.load_map_structure("data/env/train/")
    if not loaded:
        return None
    
    # Load environment data
    blocks = np.load(f"data/env/train/blocks_{ratio_index}_{index}.npy")
    nonblocks = np.setdiff1d(np.arange(100), blocks)
    
    # Calculate geodesic distances
    geodesic_dists = calculate_geodesic_distances(nonblocks, blocks, 10, 10)
    
    topology_scores = calculate_topology_preservation_geodesic(
        geodesic_dists, sr.sr[nonblocks][:, nonblocks]
    )
    
    return topology_scores
        

def run_env_bc_topo_by_scale_combined(args):
    """Run BC topology preservation analysis for a given scale value"""
    ratio_index, index, scale, border_max_shift = args
    try:
        # Load environment data
        blocks = np.load(f"data/env/train/blocks_{ratio_index}_{index}.npy")
        nonblocks = np.setdiff1d(np.arange(100), blocks)
        
        # Load and process border+corner features
        border = np.load(f"data/border/train/{ratio_index}_{index}.npy")
        border = extend_border(border, blocks, border_max_shift, 10, 10)
        border = fuzz_features(border, 10, 10, blocks, scale=scale)
        
        corner = np.load(f"data/corner/train/{ratio_index}_{index}.npy")
        corner = transform_corner(corner)
        corner = extend_corner(corner, 100 - 4 * border_max_shift)
        corner = fuzz_features(corner, 10, 10, blocks, scale=scale)
        
        border_corner = np.concatenate((border, corner), axis=1)
        
        # Calculate geodesic distances
        geodesic_dists = calculate_geodesic_distances(nonblocks, blocks, 10, 10)
    
        topology_scores = calculate_topology_preservation_geodesic(
            geodesic_dists, border_corner[nonblocks]
        )
        
        return topology_scores
        
    except Exception as e:
        print(f"Error in BC topology calculation: {e}")
        return None

def run_env_sr_dim_by_gamma(args):
    """Run SR dimensionality analysis for a given gamma value"""
    gamma, ratio_index, index = args
    try:
        sr = SRMB(10, 10, gamma=gamma)
        sr.set_index(f"{ratio_index}_{index}")
        loaded = sr.load_map_structure("data/env/train/")
        if not loaded:
            return None
        
        blocks = np.load(f"data/env/train/blocks_{ratio_index}_{index}.npy")
        nonblocks = np.setdiff1d(np.arange(100), blocks)
        
        # Calculate dimensionality of SR using original manifold
        dim, _, _ = estimate_dimensionality(sr.sr[nonblocks][:, nonblocks])
        return dim
        
    except Exception as e:
        print(e)
        return None

def run_env_bc_dim_by_scale(args):
    ratio_index, index, scale, border_max_shift = args
    sr = SRMB(10, 10, gamma=0.995)
    sr.set_index(f"{ratio_index}_{index}")
    loaded = sr.load_map_structure("data/env/train/")
    if not loaded:
        return None
    blocks = np.load(f"data/env/train/blocks_{ratio_index}_{index}.npy")
    nonblocks = np.setdiff1d(np.arange(100), blocks)
    
    # Load and process border+corner features
    border = np.load(f"data/border/train/{ratio_index}_{index}.npy")
    border = extend_border(border, blocks, border_max_shift, 10, 10)
    border = fuzz_features(border, 10, 10, blocks, scale=scale)
    corner = np.load(f"data/corner/train/{ratio_index}_{index}.npy")
    corner = transform_corner(corner)
    corner = extend_corner(corner, 100 - 4 * border_max_shift)
    corner = fuzz_features(corner, 10, 10, blocks, scale=scale)
    border_corner = np.concatenate((border, corner), axis=1)
    
    # Calculate dimensionality of border+corner using original manifold
    dim, _, _ = estimate_dimensionality(border_corner[nonblocks])
    return dim


def estimate_dimensionality(X, n_radii=50, percentile_range=(20, 80)):
    """
    Estimate latent dimensionality using the Grassberger-Procaccia algorithm.

    Parameters:
    - X: array of shape (N, D), point cloud (e.g., SR vectors or feature representations)
    - n_radii: number of radius values to use
    - percentile_range: tuple (low_pct, high_pct) for selecting the range of C(r) values to fit

    Returns:
    - alpha: estimated dimensionality (slope)
    - r_vals: array of radii used
    - C_vals: correlation integral values for each radius
    """
    
    # Compute pairwise distances
    dists = pdist(X, "euclidean")
    N = X.shape[0]
    # Define radii from min to max pairwise distances
    r_min, r_max = np.min(dists), np.max(dists)
    r_vals = np.linspace(r_min, r_max, n_radii)

    # Compute correlation integral C(r) for each radius
    C_vals = []
    for r in r_vals:
        # Count pairs with distance < r
        count = np.sum(dists < r)
        # Normalize: number of pairs = N*(N-1)/2
        C_vals.append((2.0 * count) / (N * (N - 1)))
    C_vals = np.array(C_vals)

    # Select radii corresponding to percentile range of C(r)
    low_pct, high_pct = np.percentile(C_vals, percentile_range)
    mask = (C_vals >= low_pct) & (C_vals <= high_pct)

    # Fit line to log-log plot
    log_r = np.log(r_vals[mask]).reshape(-1, 1)
    log_C = np.log(C_vals[mask])
    reg = LinearRegression().fit(log_r, log_C)
    alpha = reg.coef_[0]

    return alpha, r_vals, C_vals


def run_unified_computation(start_index, end_index, gamma_values, scales):
    """Run unified computation"""
    print(f"Starting unified computation from index {start_index} to {end_index}")
    print(f"Gamma values: {gamma_values}")
    print(f"Scales: {scales}")
    
    # Create data directory
    data_dir = "data/neural_preds/nakai_2024/raw_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save parameter values
    np.save(os.path.join(data_dir, "gamma_values.npy"), np.array(gamma_values))
    np.save(os.path.join(data_dir, "scales.npy"), np.array(scales))
    

    
    # Process each environment
    for overall_index in tqdm(range(start_index, end_index)):
        ratio_index = (overall_index - 1) // 100 + 1
        index = (overall_index - 1) % 100 + 1
        
        print(f"Processing environment {ratio_index}_{index} (overall index {overall_index})")
        # SR dimensionality analysis
        sr_dim_results = []
        for i, gamma in enumerate(gamma_values):
            result = run_env_sr_dim_by_gamma((gamma, ratio_index, index))
            if result is not None:
                sr_dim_results.append(result)
        
        # SR topology analysis
        sr_topo_results = []
        for i, gamma in enumerate(gamma_values):
            result = run_env_sr_topo_by_gamma_combined((gamma, ratio_index, index))
            if result is not None:
                sr_topo_results.append(result)

        np.save(os.path.join(data_dir, f"sr_dim_{overall_index}.npy"), np.array(sr_dim_results))
        np.save(os.path.join(data_dir, f"sr_raw_{overall_index}.npy"), np.array(sr_topo_results))

        for border_max_shift in range(1, 24):
            # BC dimensionality analysis
            bc_dim_results = []
            for i, scale in enumerate(scales):
                result = run_env_bc_dim_by_scale((ratio_index, index, scale, border_max_shift))
                if result is not None:
                    bc_dim_results.append(result)
            
            # BC topology analysis
            bc_topo_results = []
            for i, scale in enumerate(scales):
                result = run_env_bc_topo_by_scale_combined((ratio_index, index, scale, border_max_shift))
                if result is not None:
                    bc_topo_results.append(result)

            
            np.save(os.path.join(data_dir, f"bc_dim_{overall_index}_{border_max_shift}.npy"), np.array(bc_dim_results))
            np.save(os.path.join(data_dir, f"bc_raw_{overall_index}_{border_max_shift}.npy"), np.array(bc_topo_results))




def run():
    gamma_values = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        0.95, 0.99, 0.995, 0.999
    ]
    
    # Create gamma-equivalent scales
    scales = create_gamma_equivalent_scales(gamma_values)

    run_unified_computation(1, 1001, gamma_values, scales)


def postprocess_raw_data():
    data_dir = "data/neural_preds/nakai_2024/raw_data"
    processed_dir = "data/neural_preds/nakai_2024"
    
    sr_dim_list = []
    bc_dim_list = []
    missing_sr_dim = 0
    missing_bc_dim = 0


    sr_raw_list = []
    bc_raw_list = []
    missing_sr_raw = 0
    missing_bc_raw = 0
    
    for overall_index in range(1, 1001):
        try:
            sr_dim = np.load(os.path.join(data_dir, f"sr_dim_{overall_index}.npy"))
            sr_dim_list.append(sr_dim)
        except FileNotFoundError:
            missing_sr_dim += 1
            continue

    if len(sr_dim_list) > 0:
        sr_dim_list = np.array(sr_dim_list)
        sr_dim_means = np.mean(sr_dim_list, axis=0)
        sr_dim_stds = np.std(sr_dim_list, axis=0)
        np.save(os.path.join(data_dir, "sr_dim_means_processed.npy"), np.array(sr_dim_means))
        np.save(os.path.join(data_dir, "sr_dim_stds_processed.npy"), np.array(sr_dim_stds))
    else:
        pass


    for overall_index in range(1, 1001):
        try:
            sr_raw = np.load(os.path.join(data_dir, f"sr_raw_{overall_index}.npy"))
            sr_raw_list.append(sr_raw)
        except FileNotFoundError:
            missing_sr_raw += 1
            continue

    # Process SR data
    if len(sr_raw_list) > 0:
        sr_value_list = []
        for i, sr_raw in enumerate(sr_raw_list):
            sr_value_list.append([])
            for j in range(len(sr_raw)):
                sr_value_list[i].append(sr_raw[j])
        sr_value_list = np.array(sr_value_list)
        sr_means = np.mean(sr_value_list, axis=0)
        sr_stds = np.std(sr_value_list, axis=0)
        
        np.save(os.path.join(processed_dir, f"sr_means_processed.npy"), np.array(sr_means))
        np.save(os.path.join(processed_dir, f"sr_stds_processed.npy"), np.array(sr_stds))
    else:
        pass

    bc_dim_means_per_border_max_shift = []
    bc_dim_stds_per_border_max_shift = []
    for border_max_shift in range(1, 24):
        for overall_index in range(1, 1001):
            try:
                bc_dim = np.load(os.path.join(data_dir, f"bc_dim_{overall_index}_{border_max_shift}.npy"))
                bc_dim_list.append(bc_dim)
            except FileNotFoundError:
                missing_bc_dim += 1
                continue
        
        if len(bc_dim_list) > 0:
            bc_dim_means_per_border_max_shift.append(np.mean(bc_dim_list, axis=0))
            bc_dim_stds_per_border_max_shift.append(np.std(bc_dim_list, axis=0))
        else:
            pass
        
    np.save(os.path.join(processed_dir, "bc_dim_means_processed.npy"), np.array(bc_dim_means_per_border_max_shift))
    np.save(os.path.join(processed_dir, "bc_dim_stds_processed.npy"), np.array(bc_dim_stds_per_border_max_shift))


def calculate_dimensionality_of_border_corner():
    sr = SRMB(10, 10, gamma=0.990)
    dim_border_list_per_scale = []
    dim_corner_list_per_scale = []
    dim_border_corner_list_per_scale = []
    scale = 999.5
    for num_border in range(1, 24):
        print(f'num_border: {num_border}')
        dim_sr_list = []
        dim_border_list = []
        dim_corner_list = []
        dim_border_corner_list = []
        for ratio_index in range(1, 11):
            for index in range(1, 101):
                sr.set_index(f'{ratio_index}_{index}')
                sr.load_map_structure('data/env/train/')
                blocks = np.load(f'data/env/train/blocks_{ratio_index}_{index}.npy')
                nonblocks = np.setdiff1d(np.arange(100), blocks)
                border = np.load(f' data/border/train/{ratio_index}_{index}.npy')
                border = extend_border(border, blocks, num_border, 10, 10)
                border = fuzz_features(border, 10, 10, blocks, scale=scale)
                # border = fuzz_features_sr(border, sr.sr)
                corner = np.load(f'data/corner/train/{ratio_index}_{index}.npy')
                corner = transform_corner(corner)
                corner = extend_corner(corner, 100 - num_border * 4)
                corner = fuzz_features(corner, 10, 10, blocks, scale=scale)
                # corner = fuzz_features_sr(corner, sr.sr)
                border_corner = np.concatenate((border, corner), axis=1)
                if border_corner.shape[1] != 100:
                    print(f"Error at {ratio_index}_{index}_{num_border}")
                dim_sr, r_sr, C_sr = estimate_dimensionality(sr.sr[nonblocks][:, nonblocks])
                dim_border, r_border, C_border = estimate_dimensionality(border[nonblocks])
                dim_corner, r_corner, C_corner = estimate_dimensionality(corner[nonblocks])
                dim_border_corner, r_border_corner, C_border_corner = estimate_dimensionality(border_corner[nonblocks])
                dim_sr_list.append(dim_sr)
                dim_border_list.append(dim_border)
                dim_corner_list.append(dim_corner)
                dim_border_corner_list.append(dim_border_corner)
        dim_border_list_per_scale.append(dim_border_list)
        dim_corner_list_per_scale.append(dim_corner_list)
        dim_border_corner_list_per_scale.append(dim_border_corner_list)
        np.save('data/neural_preds/nakai_2024/dim_sr.npy', np.array(dim_sr_list))
        np.save('data/neural_preds/nakai_2024/dim_border_corner.npy', np.array(dim_border_corner_list))




def check_data_availability():
    import os
    
    data_dir = "data/neural_preds/nakai_2024"
    
    if not os.path.exists(data_dir):
        return False
    
    # Check if processed data files exist
    required_files = [
        "sr_dim_means_processed.npy",
        "sr_dim_stds_processed.npy",
        "sr_means_processed.npy",
        "sr_stds_processed.npy",
        "bc_dim_means_processed.npy",
        "bc_dim_stds_processed.npy",
        "dim_sr.npy",
        "dim_border_corner.npy"
    ]
    
    for filename in required_files:
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            return False
    
    return True


def main():
    if not check_data_availability():
        run()
        postprocess_raw_data()
        calculate_dimensionality_of_border_corner()

if __name__ == "__main__":
    main()

