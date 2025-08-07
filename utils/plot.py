import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import more_itertools
import networkx as nx
import matplotlib.colors as mcolors

def plot_transition_graph(
    transition_matrix: np.ndarray,
    junction_states: list,
    deadend_states: list,
    edge_states: list,
    junction_color: str,
    deadend_color: str,
    edge_color: str,
    num_columns: int = 5,
    node_size: float = 300,
    edge_width: float = 1.0,
    ax=None,
    seed_to_color: dict = None
):
    """
    Plot a NetworkX graph of the state‐transition matrix with node positions matching matshow layout.
    By default (seed_to_color=None), nodes are colored by (junction_states, deadend_states, edge_states).
    If seed_to_color is provided as a dict {node_index: color_str}, then:
      - Those “seed” nodes are colored exactly as given.
      - All other nodes get colors by interpolating between the two nearest seeds (in hop‐distance).
    
    Parameters
    ----------
    transition_matrix : np.ndarray
        Square matrix (n_states × n_states) where entry (i, j) > 0 indicates a transition (i ↔ j).
    junction_states : list of int
        List of state indices to be colored with `junction_color` (only used if seed_to_color is None).
    deadend_states : list of int
        List of state indices to be colored with `deadend_color` (only used if seed_to_color is None).
    edge_states : list of int
        List of state indices to be colored with `edge_color` (only used if seed_to_color is None).
    junction_color : str
        Color (HTML name or hex) for nodes in `junction_states`.
    deadend_color : str
        Color for nodes in `deadend_states`.
    edge_color : str
        Color for nodes in `edge_states`.
    num_columns : int, optional (default=5)
        Number of columns in the grid layout (like matshow).
    node_size : float, optional (default=300)
        Marker size for nodes.
    edge_width : float, optional (default=1.0)
        Line width for edges.
    ax : matplotlib.axes.Axes, optional (default=None)
        Axes on which to draw. If None, uses plt.gca().
    seed_to_color : dict, optional (default=None)
        Mapping {node_index: color_str}. If provided, overrides the
        (junction_, deadend_, edge_) coloring. Colors all seed nodes
        exactly as given, then interpolates colors for other nodes based
        on shortest‐path hop‐distance to the two nearest seeds.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes with the plotted graph.
    """

    # 1) Verify square matrix
    n_states = transition_matrix.shape[0]
    if transition_matrix.shape[0] != transition_matrix.shape[1]:
        raise ValueError("transition_matrix must be square (n_states x n_states).")

    # 2) Compute number of rows for grid
    n_rows = int(np.ceil(n_states / num_columns))

    # 3) Build pos dict to mimic matshow, but invert x so that col=0 appears at rightmost if desired:
    #    row = state // num_columns
    #    col = state % num_columns
    #    x = num_columns - 1 - col  (so that state 0 is at x=num_columns-1)
    #    y = n_rows - 1 - row       (so that row 0 is at top)
    pos = {}
    for state in range(n_states):
        row = state // num_columns
        col = state % num_columns
        x = col
        y = row
        pos[state] = (x, y)

    # 4) Build undirected Graph G (since transitions are symmetric for connectivity)
    G = nx.Graph()
    G.add_nodes_from(range(n_states))

    it = np.nditer(transition_matrix, flags=['multi_index'])
    while not it.finished:
        i, j = it.multi_index
        if it[0] > 0:
            G.add_edge(i, j)
        it.iternext()

    # 5) Determine node colors
    if seed_to_color:
        # ----- 5a) Distance‐based seed interpolation -----
        # Convert seed colors to RGB arrays
        seed_rgb = {}
        for node_idx, colstr in seed_to_color.items():
            seed_rgb[node_idx] = np.array(mcolors.to_rgb(colstr), dtype=float)

        seed_nodes = list(seed_rgb.keys())
        k = len(seed_nodes)
        # Precompute shortest‐path lengths from each seed to every node (hop‐distance)
        # We'll treat unreachable nodes as distance = large (e.g. n_states+1)
        LARGE = n_states + 1
        dist_to_seeds = np.full((k, n_states), LARGE, dtype=float)

        # Use undirected G for hop‐distance
        for idx, s in enumerate(seed_nodes):
            # single‐source shortest path (hop‐count)
            dist_dict = dict(nx.shortest_path_length(G, source=s))
            for node, d in dist_dict.items():
                dist_to_seeds[idx, node] = d

        # Assign colors: if node is a seed, use its exact color; otherwise find two nearest seeds
        node_colors = []
        for v in range(n_states):
            if v in seed_rgb:
                node_colors.append(seed_rgb[v])
            else:
                # distances from v to each seed:
                dists = dist_to_seeds[:, v]  # shape (k,)
                # find two smallest distances
                if k == 1:
                    i1 = 0
                    c_color = seed_rgb[seed_nodes[0]]
                    node_colors.append(c_color)
                    continue
                # sort indices
                idxs = np.argsort(dists)  # ascending
                i1, i2 = idxs[0], idxs[1]
                d1, d2 = dists[i1], dists[i2]
                c1 = seed_rgb[seed_nodes[i1]]
                c2 = seed_rgb[seed_nodes[i2]]
                # if both distances are LARGE (unreachable), fallback to gray
                if d1 >= LARGE and d2 >= LARGE:
                    node_colors.append(np.array(mcolors.to_rgb("gray"), dtype=float))
                else:
                    # if one seed is unreachable, treat that as d2 LARGE → t = 0 → color = c1
                    if d1 + d2 > 0 and d2 < LARGE:
                        t = d1 / (d1 + d2)
                    else:
                        t = 0.0
                    interp_col = (1 - t) * c1 + t * c2
                    node_colors.append(interp_col)
        # Convert node_colors list of RGB‐arrays to list of tuples
        node_colors = [tuple(c) for c in node_colors]
    else:
        # ----- 5b) Original junction/deadend/edge coloring -----
        node_colors = []
        for state in G.nodes():
            if state in junction_states:
                node_colors.append(junction_color)
            elif state in deadend_states:
                node_colors.append(deadend_color)
            elif state in edge_states:
                node_colors.append(edge_color)
            else:
                node_colors.append('gray')

    # 6) If no Axes given, get current Axes
    if ax is None:
        ax = plt.gca()

    # 7) Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        node_color=node_colors,
        ax=ax
    )

    # 8) Draw edges (always unweighted, uniform color)
    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_width,
        ax=ax
    )

    # 9) Cleanup axes for clean display
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, num_columns - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()  # match matshow orientation

    return ax


def plot_matrix(matrix, save=False, directory=None):
    plt.matshow(matrix)
    plt.colorbar()
    if save:
        plt.savefig(directory + '.png')
        plt.close()
    else:
        plt.show()


def smooth_mean_std(mean_series, std_series, window_size):
    """
    Smooths the mean and std time series data using a moving average filter with padding.
    
    Parameters:
    mean_series (numpy array): The mean time series data.
    std_series (numpy array): The std time series data.
    window_size (int): The size of the moving window.
    
    Returns:
    tuple: The smoothed mean and std time series data.
    """
    # Create a moving average filter
    window = np.ones(window_size) / window_size
    
    # Pad the series to handle the edges
    pad_width = window_size // 2
    padded_mean = np.pad(mean_series, (pad_width, pad_width), mode='edge')
    padded_std = np.pad(std_series, (pad_width, pad_width), mode='edge')
    
    # Apply the filter to the padded series using np.convolve with 'same' mode
    smoothed_mean = np.convolve(padded_mean, window, mode='same')[pad_width:-pad_width]
    smoothed_std = np.convolve(padded_std, window, mode='same')[pad_width:-pad_width]
    
    return smoothed_mean, smoothed_std


def hex_to_rgb_normalized(hex_color):
    """
    Convert a hex color code to a normalized RGB tuple (values between 0 and 1).
    :param hex_color: str, the hex color code (e.g., "#FF69B4")
    :return: tuple, the normalized RGB value (e.g., (1.0, 0.411, 0.705))
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def plot_multiple(row, column, index_row, index_column, outputs, save=True, directory='display/result'):
    plt.figure(1, figsize=(index_column * 3, index_row * 3))
    plt.subplots_adjust(wspace=0.1, hspace=0.0001)
    for k in range(index_row * index_column):
        ax = plt.subplot(index_row, index_column, k + 1)
        ax.imshow(outputs[k].reshape(row, column))
        ax.set_xticks([])
        ax.set_yticks([])
    if save:
        plt.savefig(directory + '.png')
        plt.close()
    else:
        plt.show()


def scatter(datapoints, save=True, directory='display/result', axis_off=True):
    # Extract x, y, and z coordinates
    # Colors for the scatter plot points
    colors = np.arange(datapoints.shape[0]).astype(float) * -1

    # Creating 3D scatter plot
    fig = plt.figure()

    x = datapoints[:, 0]
    y = datapoints[:, 1]
    if len(datapoints[0]) >= 3:
        z = datapoints[:, 2]
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x, y, z, c=colors, s=10, cmap='rainbow')
    else:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(x, y, c=colors, s=10, cmap='rainbow')
    ax.grid(False)
    if axis_off:
        ax.set_axis_off()
    plt.tight_layout()

    # Save or show the plot
    if save:
        plt.savefig(directory + '.png')
        plt.close()
    else:
        plt.show()

    '''plt.scatter(
        datapoints[:, 0], datapoints[:, 1],
        c=np.arange(np.shape(datapoints)[0]).astype(float)*-1, cmap='rainbow'
    )
    if save:
        plt.savefig(directory + '.png')
        plt.close()
    else:
        plt.show()'''


def scatter_multiple(index_row, index_column, datapoints, save=False, directory=None, axis_off=True):
    dim = np.shape(datapoints[0])[1]
    if dim in [2, 3]:
        f, axes = plt.subplots(index_row, index_column, subplot_kw={'projection': '3d'} if dim == 3 else None, sharex='all', sharey='all')
        f.set_size_inches((index_column * 3, index_row * 3))
        for r in range(index_row):
            for c in range(index_column):
                k = r * index_column + c
                x = datapoints[k][:, 0]
                y = datapoints[k][:, 1]
                colors = np.arange(datapoints[k].shape[0]).astype(float) * -1
                if dim == 3:
                    z = datapoints[k][:, 2]
                    if index_row > 1 and index_column > 1:
                        axes[r, c].scatter(x, y, z, c=colors, s=10, cmap="rainbow")
                        axes[r, c].grid(False)
                        if axis_off:
                            axes[r, c].set_axis_off()
                    elif index_row == 1 and index_column > 1:
                        axes[c].scatter(x, y, z, c=colors, s=10, cmap="rainbow")
                        axes[c].grid(False)
                        if axis_off:
                            axes[c].set_axis_off()
                    elif index_row > 1 and index_column == 1:
                        axes[r].scatter(x, y, z, c=colors, s=10, cmap="rainbow")
                        axes[r].grid(False)
                        if axis_off:
                            axes[r].set_axis_off()
                    else:
                        raise ValueError('index_row and index_column values are not appropriate')
                else:
                    if index_row > 1 and index_column > 1:
                        axes[r, c].scatter(x, y, c=colors, s=10, cmap="rainbow")
                        axes[r, c].grid(False)
                        if axis_off:
                            axes[r, c].set_axis_off()
                    elif index_row == 1 and index_column > 1:
                        axes[c].scatter(x, y, c=colors, s=10, cmap="rainbow")
                        axes[c].grid(False)
                        if axis_off:
                            axes[c].set_axis_off()
                    elif index_row > 1 and index_column == 1:
                        axes[r].scatter(x, y, c=colors, s=10, cmap="rainbow")
                        axes[r].grid(False)
                        if axis_off:
                            axes[r].set_axis_off()
                    else:
                        raise ValueError('index_row and index_column values are not appropriate')

        if save:
            plt.savefig(directory + '.png')
            plt.close()
        else:
            plt.show()
    else:
        raise ValueError('The dimension of the data points is not 2 or 3')
        '''if simple_output:
            dim_list = list(more_itertools.pairwise(np.arange(dim)))
        else:
            dim_list = list(itertools.combinations(np.arange(dim), 2))
        for dimensions in dim_list:
            f, axes = plt.subplots(index_row, index_column, sharex='all', sharey='all')
            f.set_size_inches((index_column * 3, index_row * 3))
            for r in range(index_row):
                for c in range(index_column):
                    k = r * index_column + c
                    if index_row > 1 and index_column > 1:
                        axes[r, c].scatter(datapoints[k][:, dimensions[0]], datapoints[k][:, dimensions[1]],
                                       c=np.arange(np.shape(datapoints[k])[0]).astype(float)*-1, cmap='rainbow')
                    elif index_row == 1 and index_column > 1:
                        axes[c].scatter(datapoints[k][:, dimensions[0]], datapoints[k][:, dimensions[1]],
                                           c=np.arange(np.shape(datapoints[k])[0]).astype(float)*-1, cmap='rainbow')
                    elif index_row > 1 and index_column == 1:
                        axes[r].scatter(datapoints[k][:, dimensions[0]], datapoints[k][:, dimensions[1]],
                                           c=np.arange(np.shape(datapoints[k])[0]).astype(float)*-1, cmap='rainbow')
                    else:
                        plt.scatter(datapoints[k][:, dimensions[0]], datapoints[k][:, dimensions[1]],
                                 c=np.arange(np.shape(datapoints[k])[0]).astype(float) * -1,
                                 cmap='rainbow')
                    # axes[r, c].scatter(datapoints[:, 0], datapoints[:, 1])
            if save:
                plt.savefig(directory + '_({}, {})'.format(dimensions[0], dimensions[1]) + '.png')
                plt.close()
            else:
                plt.show()'''


def unique_name(dir, name, ext):
    actual_name = "%s/%s_1.%s" % (dir, name, ext)
    c = itertools.count(start=2, step=1)
    while os.path.exists(actual_name):
        actual_name = "%s/%s_%d.%s" % (dir, name, next(c), ext)
    return actual_name


if __name__ == '__main__':
    pass
