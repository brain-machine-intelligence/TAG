from modules.base import SRMB
from modules.model import TAGUpdate
from modules.comparison import TAGUpdateBiased
from modules.env import MazeEnvironment
import numpy as np
from skimage.transform import rescale
from scipy.signal import find_peaks
from tests.inner_loop import random_walk_policy
from tqdm import tqdm

def calculate_autocorrelogram(rate_matrix: np.ndarray, target_shape) -> np.ndarray:
    """
    Compute the 2D spatial autocorrelogram of a firing rate map.

    Parameters:
        rate_map (np.ndarray): 2D array of firing rates.

    Returns:
        np.ndarray: 2D autocorrelogram, centered.
    """
    ac_list = []
    r, c = target_shape
    # Subtract mean to remove DC component
    for index in range(rate_matrix.shape[1]):
        rate_map = rate_matrix[:, index].reshape(r, c)
        mean_subtracted = rate_map - np.nanmean(rate_map)
        # Fourier transform
        ft = np.fft.fft2(np.nan_to_num(mean_subtracted))
        # Compute autocorrelation via inverse FFT of power spectrum
        ac = np.fft.ifft2(ft * np.conj(ft)).real
        # Center the zero-lag peak
        ac_list.append(np.fft.fftshift(ac).flatten())
    return np.array(ac_list).T


def _rescale_and_resize(
    probe_ac: np.ndarray,
    current_shape: tuple,
    target_shape: tuple,
    sx: float,
    sy: float,
) -> np.ndarray:
    rescaled = []
    r, c = current_shape
    for ac_map_index in range(probe_ac.shape[1]):
        ac_map = probe_ac[:, ac_map_index].reshape(r, c)
        scaled = rescale(ac_map, (sy, sx), mode="reflect", anti_aliasing=False)
        # 2) Crop center if too large, or pad with zeros if too small
        h, w = target_shape
        sh, sw = scaled.shape

        # 2) Crop or pad from top-left
        h, w = target_shape
        sh, sw = scaled.shape

        # Crop: take at most target size from top-left
        end_h = min(h, sh)
        end_w = min(w, sw)
        cropped = scaled[0:end_h, 0:end_w]

        # Padding sizes: only on bottom and right
        pad_h_after = h - cropped.shape[0]
        pad_w_after = w - cropped.shape[1]

        out = np.pad(
            cropped,
            ((0, pad_h_after), (0, pad_w_after)),
            mode="constant",
            constant_values=0,
        )
        rescaled.append(out.flatten())
    return np.array(rescaled).T


def find_rescaling(
    baseline_ac: np.ndarray,
    probe_ac: np.ndarray,
    current_shape: tuple = (10, 10),
    target_shape: tuple = (10, 10),
    scale_range: np.ndarray = np.linspace(0.5, 1.5, 101),
) -> dict:
    """
    Brute‑force search for the pair of (scale_x, scale_y) that maximizes
    the Pearson correlation between baseline and scaled probe autocorrelograms.

    Returns a dict with:
        - scale_x: best horizontal scaling
        - scale_y: best vertical scaling
        - correlation: correlation coefficient
    """
    best_score = -np.inf
    best_sx, best_sy = 1.0, 1.0

    for sx in scale_range:
        for sy in scale_range:
            candidate = _rescale_and_resize(
                probe_ac, current_shape, target_shape, sx, sy
            )
            # Flatten and mask NaNs
            b = baseline_ac.flatten()
            c = candidate.flatten()
            mask = ~np.isnan(b)
            score = np.corrcoef(b[mask], c[mask])[0, 1]
            if score > best_score:
                best_score = score
                best_sx, best_sy = sx, sy

    return {"scale_x": best_sx, "scale_y": best_sy, "correlation": best_score}


def normalize_rescaling(scale_factor: float, env_change_factor: float) -> float:
    """
    Normalize raw scale_factor by the percentage change in enclosure size.

    e.g., if walls doubled (env_change_factor=2.0), and grid stretched
    by scale_factor=1.5, normalized = (1.5 / 2.0) * 100 = 75%.
    """
    return (scale_factor / env_change_factor) * 100


def grid_spacing(
    ac: np.ndarray, axis: str = "x", num_peaks: int = 3, min_dist: int = 5
) -> float:
    """
    Estimate grid spacing by finding the average location of the first few
    peaks in the autocorrelogram profile along one axis.

    Parameters:
        ac (np.ndarray): Autocorrelogram.
        axis (str): 'x' or 'y'.
    """
    center = np.array(ac.shape) // 2
    if axis == "x":
        profile = ac[center[0], center[1] :]
    else:
        profile = ac[center[0] :, center[1]]

    peaks, _ = find_peaks(profile, distance=min_dist)
    if len(peaks) < num_peaks:
        return np.nan
    return np.mean(peaks[:num_peaks])


def asymmetry_index(ac: np.ndarray, **kwargs) -> float:
    """
    Compute the asymmetry index as the ratio of horizontal vs. vertical spacing.
    """
    sx = grid_spacing(ac, axis="x", **kwargs)
    sy = grid_spacing(ac, axis="y", **kwargs)
    return sx / sy

def run():
    env_ll = MazeEnvironment(10, 10, auto_reset=False)
    env_ll.update_map(blocks=[])

    env_ss = MazeEnvironment(10, 10, auto_reset=False)
    blocks_lowerright = []
    for r in range(7, 10):
        for c in range(7, 10):
            blocks_lowerright.append(env_ss.index_to_state(r, c))
    blocks_lowerleft = []
    for r in range(7, 10):
        for c in range(0, 7):
            blocks_lowerleft.append(env_ss.index_to_state(r, c))
    blocks_upperright = []
    for r in range(0, 7):
        for c in range(7, 10):
            blocks_upperright.append(env_ss.index_to_state(r, c))

    blocks_ss = blocks_lowerright + blocks_lowerleft + blocks_upperright
    env_ss.update_map(blocks=blocks_ss)
    env_ss.set_start_goal_states(0, 66)

    env_ls = MazeEnvironment(10, 10, auto_reset=False)
    blocks_ls = blocks_lowerright + blocks_upperright
    env_ls.update_map(blocks=blocks_ls)
    env_ls.set_start_goal_states(0, 96)

    env_sl = MazeEnvironment(10, 10, auto_reset=False)
    blocks_sl = blocks_lowerleft + blocks_lowerright
    env_sl.update_map(blocks=blocks_sl)
    env_sl.set_start_goal_states(0, 69)

    tag_with_border_result = np.zeros(6)
    tag_without_border_result = np.zeros(6)

    np.random.seed(42)
    rw_ll = []
    rw_ss = []
    rw_ls = []
    rw_sl = []

    rw_list = [rw_ll, rw_ss, rw_ls, rw_sl]
    env_list = [env_ll, env_ss, env_ls, env_sl]

    for j in range(4):
        env = env_list[j]
        rw = rw_list[j]
        state = env.reset()
        # Run the random walk for 20,000 steps
        for i in tqdm(range(20000)):
            action = random_walk_policy(0.5)
            next_state, _, _, _ = env.step(action)
            rw.append([state, action, next_state])
            state = next_state

    baseline = SRMB(10, 10, 0.995)
    baseline.update_map_structure(blocks=[])
    tag_with_border = TAGUpdate(10, 10, 0.995)
    tag_with_border.set_map(baseline.sr)
    tag_without_border = TAGUpdateBiased(10, 10)
    tag_without_border.set_map(baseline.sr)
    tag_without_border.transition_count = np.ones((100, 100))

    for i in tqdm(range(20000)):
        state, action, next_state = rw_ls[i]
        tag_with_border.update_map(state, action, next_state)
        tag_without_border.update_map(state, action, next_state)

    baseline_map = baseline.return_successormap()
    probe_map = tag_with_border.return_successormap()[env_ls.nonblocks]

    # Compute autocorrelograms
    base_ac = calculate_autocorrelogram(baseline_map, (10, 10))
    probe_ac = calculate_autocorrelogram(probe_map, (10, 7))

    # Find rescaling factors
    res = find_rescaling(base_ac, probe_ac, (10, 7), (10, 10))

    tag_with_border_result[0] = 1 / res["scale_x"]
    tag_with_border_result[1] = 1 / res["scale_y"]

    probe_map = tag_without_border.return_successormap()[env_ls.nonblocks]
    probe_ac = calculate_autocorrelogram(probe_map, (10, 7))
    res = find_rescaling(base_ac, probe_ac, (10, 7), (10, 10))
    tag_without_border_result[0] = 1 / res["scale_x"]
    tag_without_border_result[1] = 1 / res["scale_y"]

    for i in tqdm(range(20000)):
        state, action, next_state = rw_sl[i]
        tag_with_border.update_map(state, action, next_state)
        tag_without_border.update_map(state, action, next_state)

    baseline_map = baseline.return_successormap()
    probe_map = tag_with_border.return_successormap()[env_sl.nonblocks]
    # Compute autocorrelograms
    base_ac = calculate_autocorrelogram(baseline_map, (10, 10))
    probe_ac = calculate_autocorrelogram(probe_map, (7, 10))

    # Find rescaling factors
    res = find_rescaling(base_ac, probe_ac, (7, 10), (10, 10))
    tag_with_border_result[2] = 1 / res["scale_x"]
    tag_with_border_result[3] = 1 / res["scale_y"]

    probe_map = tag_without_border.return_successormap()[env_sl.nonblocks]
    probe_ac = calculate_autocorrelogram(probe_map, (7, 10))
    res = find_rescaling(base_ac, probe_ac, (7, 10), (10, 10))
    tag_without_border_result[2] = 1 / res["scale_x"]
    tag_without_border_result[3] = 1 / res["scale_y"]

    for i in tqdm(range(20000)):
        state, action, next_state = rw_ss[i]
        tag_with_border.update_map(state, action, next_state)
        tag_without_border.update_map(state, action, next_state)

    baseline_map = baseline.return_successormap()
    probe_map = tag_with_border.return_successormap()[env_ss.nonblocks]
    # Compute autocorrelograms
    base_ac = calculate_autocorrelogram(baseline_map, (10, 10))
    probe_ac = calculate_autocorrelogram(probe_map, (7, 7))
    # Find rescaling factors
    res = find_rescaling(base_ac, probe_ac, (7, 7), (10, 10))
    tag_with_border_result[4] = 1 / res["scale_x"]
    tag_with_border_result[5] = 1 / res["scale_y"]
    probe_map = tag_without_border.return_successormap()[env_ss.nonblocks]
    probe_ac = calculate_autocorrelogram(probe_map, (7, 7))
    res = find_rescaling(base_ac, probe_ac, (7, 7), (10, 10))
    tag_without_border_result[4] = 1 / res["scale_x"]
    tag_without_border_result[5] = 1 / res["scale_y"]
    print("TAG with border rescaling factors:")
    print(tag_with_border_result)
    print("TAG without border rescaling factors:")
    print(tag_without_border_result)

    np.save("data/neural_preds/barry_2007/tag_with_border_result.npy", tag_with_border_result)
    np.save("data/neural_preds/barry_2007/tag_without_border_result.npy", tag_without_border_result)


def check_data_availability():
    import os
    
    base_dir = "data/neural_preds/barry_2007/"
    
    if not os.path.exists(base_dir):
        return False
    
    required_files = [
        "tag_with_border_result.npy",
        "tag_without_border_result.npy"
    ]
    
    for filename in required_files:
        file_path = os.path.join(base_dir, filename)
        if not os.path.exists(file_path):
            return False
    
    return True


def main():
    if not check_data_availability():
        run()