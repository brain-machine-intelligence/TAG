'''
guo 2024 test 위항 SR decomposition을 cluster 이용해서 parallelization
'''
from ast import Not
from modules.base import *
from utils.plot import plot_multiple
import networkx as nx
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from modules.base import *
from modules.env import *
import numpy as np
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import gc
import itertools
from tqdm import tqdm
from itertools import combinations
from scipy import stats
import multiprocessing as mp
import os


num_row, num_column = 9, 9
# for scale_factor in range(1, 6):
scale_factor = 5
gamma = 0.995
n_neighbors = 13
env = MazeEnvironment(num_row, num_column)
blocks = []
for r in range(1, 8):
    for c in range(1, 8):
        blocks.append(env.index_to_state(r, c))
env.insert_block(blocks)
env.scale_map(scale_factor)
blocks = env.blocks
nonblocks = env.nonblocks
# env.visualize(display=True)

srmb = SRMB(env.num_row, env.num_column, gamma=0.995)
srmb.update_map_structure(blocks=blocks)
T = srmb.transition_matrix[nonblocks][:, nonblocks]

fraction_list = [0.1, 0.2]
seed_list = [42, 43, 44, 45, 46, 47, 48, 49]
thresholds = [230]

num_tasks = len(fraction_list) * len(seed_list) * len(thresholds)
combos = [list(itertools.product(fraction_list, seed_list, thresholds))]


def sparsify_transition(P, zero_fraction, seed=None):
    """
    P:        (n×n) transition matrix (numpy array)
    zero_fraction: 비제로 엘리먼트 중 0으로 만들 비율 (0.0 ~ 1.0)
    seed:     랜덤 시드 (int or None)

    returns:  희소화(sparsify) 후 정규화된 전이행렬
    """
    if seed is not None:
        np.random.seed(seed)

    # 1) 원본 복제
    P2 = P.copy()

    # 2) 비제로 엘리먼트의 (row, col) 인덱스 목록
    nz_rows, nz_cols = np.nonzero(P2)
    nz_pairs = list(zip(nz_rows, nz_cols))

    # 3) 0으로 만들 엘리먼트 수 계산
    total_nz = len(nz_pairs)
    n_zero = int(np.floor(zero_fraction * total_nz))
    
    # 4) 랜덤 선택
    chosen_idx = np.random.choice(total_nz, size=n_zero, replace=False)
    for idx in chosen_idx:
        i, j = nz_pairs[idx]
        P2[i, j] = 0.0

    # 5) 각 행별 합이 0이 아닐 때만 합=1 되도록 정규화
    row_sums = P2.sum(axis=1)
    nonzero_rows = row_sums > 0
    P2[nonzero_rows] = P2[nonzero_rows] / row_sums[nonzero_rows, None]

    return P2


def bias_transition(P, bias_factor, seed=None):
    """
    P:            (n×n) transition matrix (각 행 합=1, non-zero entries 모두 동일)
    bias_factor:  0.0이면 완전 uniform, 1.0에 가까워질수록 한 방향으로 강하게 편향 (0 ≤ bias_factor < 1)
    seed:         랜덤 시드 (int 또는 None)

    returns:      bias 적용 후 정규화된 전이행렬 (support는 그대로 유지)
    """
    if not (0.0 <= bias_factor < 1.0):
        raise ValueError("bias_factor must satisfy 0.0 ≤ bias_factor < 1.0")
    if seed is not None:
        np.random.seed(seed)

    P2 = P.copy()
    n = P2.shape[0]

    for i in range(n):
        # non-zero 인덱스
        idxs = np.nonzero(P2[i])[0]
        k = len(idxs)
        # 이웃이 1개 이하라면 변화 없음
        if k <= 1:
            continue

        # 1) 편향 방향 선택 (이웃 중 하나를 랜덤하게)
        b = np.random.choice(idxs)

        # 2) 기본 uniform 확률과 편향 가중치 합성
        #    원래 uniform prob = 1/k
        #    편향 확률 = (1 - bias_factor)/k  (모두에게) + bias_factor at b
        base = (1.0 - bias_factor) / k
        new_probs = np.full(k, base)
        # b에 편향
        b_idx = np.where(idxs == b)[0][0]
        new_probs[b_idx] += bias_factor

        # 3) 할당
        P2[i, idxs] = new_probs

    return P2


def geodesic_spearman_corr(embeddings, T):
    """
    Compute Spearman correlation between each state's embedding-distance profile
    and the negative geodesic distances over the graph defined by T.

    Parameters:
    - embeddings: (N, D) array of state embeddings (in your case D=2)
    - T:          (N, N) adjacency/transition matrix (nonzero entries indicate edges)

    Returns:
    - corrs:      1D array of length N, Spearman rho for each state
    """
    # 1) pairwise Euclidean distances between embeddings → M (N×N, symmetric)
    M = squareform(pdist(embeddings, metric='euclidean'))
    N = M.shape[0]
    corrs = np.zeros(N)
    # Build directed graph from adjacency
    G = nx.from_numpy_array((T > 0).astype(int), create_using=nx.DiGraph)

    for j in range(N):
        # 1) Compute shortest path lengths from node j
        lengths = nx.single_source_shortest_path_length(G, j)
        # 2) Build distance array, unreachable -> inf
        d = np.array([lengths.get(i, np.inf) for i in range(N)], dtype=float)
        # 3) Replace inf with max finite + 1 for ranking
        if np.isinf(d).any():
            max_finite = np.max(d[np.isfinite(d)])
            d[np.isinf(d)] = max_finite + 1

        # 4) Compute Spearman correlation between M[:, j] and -d
        col = M[:, j]
        rho, _ = spearmanr(col, -d)
        corrs[j] = rho

    return corrs

def split_reconstruct_SR(Q, lam, threshold):
    try:
        Q_inv = np.linalg.pinv(Q)
    except:
        try:
            Q_inv = np.linalg.inv(Q)
        except:
            return None, None

    Q_b = Q[:, :threshold]  # n x threshold
    lam_b = lam[:threshold]  # threshold
    Q_a = Q[:, threshold:]  # n x (n-threshold)
    lam_a = lam[threshold:]  # n-threshold

    SR_before = Q_b @ np.diag(lam_b) @ Q_inv[:threshold, :]
    SR_after = Q_a @ np.diag(lam_a) @ Q_inv[threshold:, :]

    return SR_before, SR_after


def compute_embedding(fraction, seed, threshold, task_type):
    sr = SRMB(num_row * scale_factor, num_column * scale_factor, gamma=gamma)
    sr.update_map_structure(blocks)
    if fraction > 0:
        if task_type == 0:
            sr.transition_matrix = sparsify_transition(sr.transition_matrix, zero_fraction=fraction, seed=seed)
        elif task_type == 1:
            sr.transition_matrix = bias_transition(sr.transition_matrix, bias_factor=fraction, seed=seed)
        else:
            raise NotImplementedError("Task type not implemented")
        sr.compute_sr()

    eigvec, eigval = sr.decompose_map_eigen()
    SR_before, SR_after = split_reconstruct_SR(eigvec.T, eigval, threshold=threshold)

    if SR_before is None or SR_after is None:
        emb_before = np.zeros((len(nonblocks), 2))
        emb_after = np.zeros((len(nonblocks), 2))
    else:
        emb_before = Isomap(
            n_components=2, n_neighbors=n_neighbors, eigen_solver="dense"
        ).fit_transform(SR_before[nonblocks][:, nonblocks])
        emb_after = Isomap(
            n_components=2, n_neighbors=n_neighbors, eigen_solver="dense"
        ).fit_transform(SR_after[nonblocks][:, nonblocks])

    emb_whole = Isomap(
        n_components=2, n_neighbors=n_neighbors, eigen_solver="dense"
    ).fit_transform(sr.sr[nonblocks][:, nonblocks])

    np.save(f'data/neural_preds/guo_2024/raw_data/emb_before_{task_type}_{fraction}_{seed}_{threshold}.npy', emb_before)
    np.save(f'data/neural_preds/guo_2024/raw_data/emb_after_{task_type}_{fraction}_{seed}_{threshold}.npy', emb_after)
    np.save(f'data/neural_preds/guo_2024/raw_data/emb_whole_{task_type}_{fraction}_{seed}_{threshold}.npy', emb_whole)



def compute_embedding_parallel():
    all_params = []
    param_list = fraction_list
    for fraction in param_list:
        for seed in seed_list:
            for threshold in thresholds:
                all_params.append((fraction, seed, threshold, 0))

    # Use joblib's Parallel with all available cores
    _ = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_embedding)(*params) for params in all_params
    )


def postprocess(task_type):
    # 전역으로 정의된 리스트들
    # fraction_list, thresholds, seed_list, T
    out_dir = 'data/neural_preds/guo_2024'
    
    # 1) (fraction, threshold) 조합 평탄화
    combos = [(f, t) for f in fraction_list for t in thresholds]
    
    # 2) 한 조합을 처리하는 함수 (클로저)
    def _process_one(fraction, threshold):
        corrs_before_list = []
        corrs_after_list = []
        corrs_whole_list = []
        
        for seed in seed_list:
            emb_before = np.load(
                f'{out_dir}/raw_data/emb_before_{task_type}_{fraction}_{seed}_{threshold}.npy'
            )
            emb_after  = np.load(
                f'{out_dir}/raw_data/emb_after_{task_type}_{fraction}_{seed}_{threshold}.npy'
            )
            emb_whole  = np.load(
                f'{out_dir}/raw_data/emb_whole_{task_type}_{fraction}_{seed}_{threshold}.npy'
            )

            # Euclidean 거리 행렬 생성
            M_before = squareform(pdist(emb_before, metric='euclidean'))
            M_after  = squareform(pdist(emb_after,  metric='euclidean'))
            M_whole  = squareform(pdist(emb_whole,  metric='euclidean'))

            corrs_before_list.append(geodesic_spearman_corr(M_before, T))
            corrs_after_list.append(geodesic_spearman_corr(M_after,  T))
            corrs_whole_list.append(geodesic_spearman_corr(M_whole,  T))

        # 결과 저장
        np.save(f'{out_dir}/corrs_before_{task_type}_{fraction}_{threshold}.npy',
                np.array(corrs_before_list))
        np.save(f'{out_dir}/corrs_after_{task_type}_{fraction}_{threshold}.npy',
                np.array(corrs_after_list))
        np.save(f'{out_dir}/corrs_whole_{task_type}_{fraction}_{threshold}.npy',
                np.array(corrs_whole_list))

    # 3) 병렬 실행: 사용 가능한 모든 코어 활용
    Parallel(n_jobs=-1)(
        delayed(_process_one)(fraction, threshold)
        for fraction, threshold in tqdm(combos, desc="Processing combos")
    )


def check_data_availability():
    import os
    
    out_dir = 'data/neural_preds/guo_2024'
    
    if not os.path.exists(out_dir):
        return False
    
    # Check if processed data files exist for task_type 0
    task_type = 0
    combos = [(f, t) for f in fraction_list for t in thresholds]
    
    for fraction, threshold in combos:
        required_files = [
            f'corrs_before_{task_type}_{fraction}_{threshold}.npy',
            f'corrs_after_{task_type}_{fraction}_{threshold}.npy',
            f'corrs_whole_{task_type}_{fraction}_{threshold}.npy'
        ]
        
        for filename in required_files:
            file_path = os.path.join(out_dir, filename)
            if not os.path.exists(file_path):
                return False
    
    return True


def main():
    if not check_data_availability():
        compute_embedding_parallel()
        postprocess(0)

