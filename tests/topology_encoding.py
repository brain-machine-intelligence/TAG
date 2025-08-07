from utils.skeletonize import skeletonize_env_parametric, plot_skeleton, draw_graph_with_curved_edges, save_adj_matrix, plot_state_segmentation, draw_scatter_plot_skeleton
import networkx as nx
import pickle
import os

from modules.env import MazeEnvironment
from modules.model import SRMB
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm



def compute_relative_phase_idxs(i_stable, N_global):
    return np.round(np.linspace(1, i_stable, N_global)).astype(int)


def find_stabilization_iter(ratio, index, max_iter=20):
    edge_dict = {}
    for i in range(1, max_iter + 1):
        G = nx.MultiGraph(nx.read_graphml(
            f"data/skeletonize/skeleton_graph/train_parametric/{ratio}_{index}_{i}.graphml"
        ))
        G = convert_to_integer_graph(G)
        edge_list = sorted(G.edges())
        edge_dict[i] = edge_list

    for i in range(1, max_iter):
        base = edge_dict[i]
        stable = True
        for j in range(i + 1, max_iter + 1):
            if edge_dict[j] != base:
                stable = False
                break
        if stable:
            return i
    return max_iter


def convert_to_integer_graph(G):
    # Step 1: Create a mapping from original nodes to integers
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}

    # Step 2: Create a new MultiGraph to maintain structure
    new_G = nx.MultiGraph()

    # Step 3: Add nodes with new integer labels
    new_G.add_nodes_from(node_mapping.values())

    # Step 4: Add edges with the new integer nodes and maintain original edge attributes
    for u, v, key, data in G.edges(data=True, keys=True):
        new_G.add_edge(node_mapping[u], node_mapping[v], key=key, **data)

    return new_G


def load_graph(ratio, index, t):
    path = f"data/skeletonize/skeleton_graph/train_parametric/{ratio}_{index}_{t}.graphml"
    G = nx.MultiGraph(nx.read_graphml(path))
    return convert_to_integer_graph(G)


def calc_parametric_basis_10_train_cluster(start_index, end_index):
    base_graph_dir = "data/skeletonize/skeleton_graph/train_parametric"
    os.makedirs(base_graph_dir, exist_ok=True)

    sizes_per_type = [500, 400, 300]
    for flat in range(start_index, end_index + 1):
        # flat 1->(1,1), 2->(1,2)...100->(1,100), 101->(2,1) ...
        ratio_index = (flat - 1) // 100 + 1
        index = (flat - 1) % 100 + 1

        env = MazeEnvironment(10, 10)
        env.set_index(f"{ratio_index}_{index}")
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

            if simplified_G is not None:
                nx.write_graphml(
                    simplified_G,
                    f"data/skeletonize/skeleton_graph/"
                    f"train_parametric/{ratio_index}_{index}_{max_iter}.graphml",
                )
          

def calc_graph_edit_dist_graph_basis_10_parametric(overall_index_1, STABLE_ITERS, N_GLOBAL):

    # (1) overall_index_1 → ratio_index_1, index_1
    ratio_index_1 = (overall_index_1 - 1) // 100 + 1
    index_1       = (overall_index_1 - 1) % 100 + 1

    # (2) env1의 안정화 iteration
    i_stable_1 = STABLE_ITERS[overall_index_1 - 1]

    # (3) env1의 상대적 위상 보간 인덱스
    idxs_1 = compute_relative_phase_idxs(i_stable_1, N_GLOBAL)

    # (4) 결과 저장용 배열: shape = (1000, N_GLOBAL), 초기값 = -1
    ged_vec = np.ones((1000, N_GLOBAL)) * (-1.0)

    # (5) env1 그래프 캐시
    G1_cache = {}
    for local_iter in np.unique(idxs_1):
        G1_cache[local_iter] = load_graph(ratio_index_1, index_1, int(local_iter))

    # (6) 전체 1000개 env에 대해 루프
    for overall_index_2 in range(1, 1001):
        # 동일 env → 행 전체 0
        if overall_index_1 == overall_index_2:
            ged_vec[overall_index_2 - 1, :] = 0.0
            continue

        # env2 ratio/index, 안정화 iteration
        ratio_index_2 = (overall_index_2 - 1) // 100 + 1
        index_2       = (overall_index_2 - 1) % 100 + 1
        i_stable_2    = STABLE_ITERS[overall_index_2 - 1]

        # env2 상대적 위상 보간 인덱스
        idxs_2 = compute_relative_phase_idxs(i_stable_2, N_GLOBAL)

        # env2 그래프 캐시
        G2_cache = {}
        for local_iter in np.unique(idxs_2):
            G2_cache[local_iter] = load_graph(ratio_index_2, index_2, int(local_iter))

        # 상대적 phase별로 Graph Edit Distance 계산
        for g in range(N_GLOBAL):
            local_iter_1 = int(idxs_1[g])
            local_iter_2 = int(idxs_2[g])

            G1 = G1_cache[local_iter_1]
            G2 = G2_cache[local_iter_2]

            try:
                # networkx.optimize_graph_edit_distance는 생성기(generator)를 반환
                # next(...)를 통해 최솟값(첫 번째 candidate)만 추출
                ged = next(nx.optimize_graph_edit_distance(G1, G2))
            except (nx.NetworkXError, StopIteration):
                ged = np.nan

            ged_vec[overall_index_2 - 1, g] = ged

    # (7) 파일 저장
    out_path = f"data/topology_transfer/raw_data/ge_dist_graph_relvec_{overall_index_1}.npy"
    np.save(out_path, ged_vec)
    print(f"Saved graph edit distances to: {out_path}")


def postprocess_ge_dist_graph_relvec():
    # 1) 첫 번째 파일을 로드하여 global_max(열 개수) 파악
    example_path = "data/topology_encoding/raw_data/ge_dist_graph_relvec_1.npy"
    try:
        example_arr = np.load(example_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"예시 파일을 찾을 수 없습니다: {example_path}")

    # example_arr의 형상 = (1000, global_max)
    if example_arr.ndim != 2 or example_arr.shape[0] != 1000:
        raise ValueError(
            f"예시 파일의 형상이 (1000, global_max) 이어야 합니다. "
            f"현재 형상: {example_arr.shape}"
        )
    global_max = example_arr.shape[1]

    # 2) 최종 3D 텐서 생성: (1000, 1000, global_max)
    tensor = np.zeros((1000, 1000, global_max), dtype=example_arr.dtype)

    # 3) 전체 인덱스를 순회하며 각 파일을 로드해 텐서에 삽입
    for overall_index_1 in tqdm(range(1, 1001)):
        relvec_path = f"data/topology_encoding/raw_data/ge_dist_graph_relvec_{overall_index_1}.npy"
        try:
            arr = np.load(relvec_path)
        except FileNotFoundError:
            print(f"[경고] 파일을 찾을 수 없습니다: {relvec_path} — 해당 인덱스 스킵")
            continue

        # 배열 형상 검증
        if arr.ndim != 2 or arr.shape != (1000, global_max):
            print(
                f"[오류] '{relvec_path}' 의 형상이 예상과 다릅니다. "
                f"예상: (1000, {global_max}), 현재: {arr.shape} — 스킵"
            )
            continue

        # 텐서의 첫 번째 축(=overall_index_1-1)에 해당하는 2D 슬라이스에 대입
        tensor[overall_index_1 - 1, :, :] = arr

    # 4) 결과를 단일 .npy 파일로 저장
    out_path = "data/topology_encoding/ge_dist_graph_relparam_tensor.npy"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, tensor)
    print(f"Processed and saved tensor to: {out_path}")


def check_data_availability():
    import os
    
    target_file = "data/topology_encoding/ge_dist_graph_relparam_tensor.npy"
    
    if os.path.exists(target_file):
        return True
    else:
        return False


def main():
    if not check_data_availability():
        calc_parametric_basis_10_train_cluster(1, 1001)
        tmp_list = []
        for overall_idx in range(1, 1001):
            ratio_idx = (overall_idx - 1) // 100 + 1
            idx       = (overall_idx - 1) % 100 + 1
            i_stable  = find_stabilization_iter(ratio_idx, idx, max_iter=20)
            tmp_list.append(i_stable)
        STABLE_ITERS = tmp_list
        N_GLOBAL = max(STABLE_ITERS)
        for overall_index_1 in range(1, 1001):
            calc_graph_edit_dist_graph_basis_10_parametric(overall_index_1, STABLE_ITERS, N_GLOBAL)
        postprocess_ge_dist_graph_relvec()


if __name__ == "__main__":
    main()
