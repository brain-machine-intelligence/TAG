import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from itertools import product, combinations
from sklearn.metrics import accuracy_score
import glob
from tensorboard.backend.event_processing import event_accumulator
from modules.model import TAGTransfer
from modules.env import MazeEnvironment
import pickle
from tqdm import tqdm
from modules.outer_loop import TransformerClassifier, transform_corner
import random
import os
import multiprocessing as mp


# Constants
num_envs = 1000  # Number of environments for training
test_envs = 200  # Number of environments for the test set
grid_size = 10
input_dim = 32  # Final total feature size after transforming vertex and adding block feature
test_interval = 5  # Test every 5 epochs

# Feature Sizes
corner_dim = 4  # Original size of corner feature (will be transformed to 8)
border_dim = 4  # Binary feature (either 0 or 1)
place_dim = 100  # Positive float values
grid_dim = 9  # Continuous values between -1 and 1
projected_place_dim = 10  # We project place feature to 10 dimensions to balance with other features
block_dim = 1  # 1 if block, 0 otherwise (binary feature)


def normalize_place(place):
    min_vals = place.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    max_vals = place.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    return (place - min_vals) / (max_vals - min_vals + 1e-8)


def normalize_grid(grid):
    min_vals = grid.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    max_vals = grid.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    return (grid - min_vals) / (max_vals - min_vals + 1e-8)


def train_and_test_model(model, train_loader, test_features, test_labels, num_epochs, test_interval, optimizer, writer):
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            corner, border, place, grid, block, labels = batch
            optimizer.zero_grad()
            outputs = model(corner, border, place, grid, block)
            loss = criterion(outputs.view(-1, model.output_dim), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            writer.add_scalar(
                "Training Loss (Batch)",
                loss.item(),
                epoch * len(train_loader) + batch_idx,
            )
        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar("Training Loss (Epoch)", avg_train_loss, epoch)
        if (epoch + 1) % test_interval == 0:
            test_acc = test_model(model, test_features, test_labels)
            writer.add_scalar("Test Accuracy", test_acc, epoch)


def test_model(model, test_features, test_labels):
    model.eval()
    with torch.no_grad():
        test_outputs = model(
            test_features[0],
            test_features[1],
            test_features[2],
            test_features[3],
            test_features[4],
        )
        _, test_preds = torch.max(test_outputs, dim=-1)
        test_acc = accuracy_score(test_labels.cpu().view(-1), test_preds.cpu().view(-1))
    return test_acc


def save_data():
    device = torch.device("cuda:0")
    pbc = TAGTransfer(10, 10)
    env = MazeEnvironment(10, 10)
    all_corners_train, all_borders_train, all_places_train = [], [], []
    all_grids_train, all_blocks_train = [], []
    for ratio_index in tqdm(range(1, 11)):
        for index in range(1, 101):
            pbc.set_map(
                np.load(f"data/sr/train/sr_{ratio_index}_{index}.npy")
            )
            env.set_index(f"{ratio_index}_{index}")
            env.load_map("data/env/train/")
            grid = pbc.return_successormap()[:, :-1]
            place = np.copy(pbc.return_map())
            corner, border = pbc.structure_head_corner_stack()
            blocks = np.array([1 if s in env.blocks else 0 for s in range(100)])
            all_grids_train.append(grid.reshape(10, 10, -1))
            all_places_train.append(place.reshape(10, 10, -1))
            all_corners_train.append(corner.reshape(10, 10, -1))
            all_borders_train.append(border.reshape(10, 10, -1))
            all_blocks_train.append(blocks.reshape(10, 10, -1))
    all_grids_train = torch.tensor(np.array(all_grids_train)).float().to(device)
    all_places_train = torch.tensor(np.array(all_places_train)).float().to(device)
    all_corners_train = torch.tensor(np.array(all_corners_train)).float().to(device)
    all_borders_train = torch.tensor(np.array(all_borders_train)).float().to(device)
    all_blocks_train = torch.tensor(np.array(all_blocks_train)).float().to(device)
    all_grids_train = normalize_grid(all_grids_train)
    all_places_train = normalize_place(all_places_train)
    all_corners_transformed_train = transform_corner(all_corners_train)
    train_features = {
        "corner": all_corners_transformed_train,
        "border": all_borders_train,
        "place": all_places_train,
        "grid": all_grids_train,
        "block": all_blocks_train,
    }
    train_labels = torch.full((num_envs, grid_size, grid_size), 0).long().to(device)
    for ratio_index in tqdm(range(1, 11)):
        for index in range(1, 101):
            env_idx = (ratio_index - 1) * 100 + (index - 1)
            with open(
                f"data/skeletonize/nodes/train/v_{ratio_index}_{index}.pkl",
                "rb",
            ) as f:
                junction_nodes = pickle.load(f)
            with open(
                f"data/skeletonize/nodes/train/e_{ratio_index}_{index}.pkl",
                "rb",
            ) as f:
                edge_nodes = pickle.load(f)
            with open(
                f"data/skeletonize/nodes/train/d_{ratio_index}_{index}.pkl",
                "rb",
            ) as f:
                deadend_nodes = pickle.load(f)
            for node_list in deadend_nodes:
                for state in node_list:
                    row, col = pbc.state_to_index(state)
                    train_labels[env_idx, row, col] = 1.0
            for node_list in edge_nodes:
                for state in node_list:
                    row, col = pbc.state_to_index(state)
                    train_labels[env_idx, row, col] = 2.0
            for node_list in junction_nodes:
                for state in node_list:
                    row, col = pbc.state_to_index(state)
                    train_labels[env_idx, row, col] = 3.0

    all_corners_test, all_borders_test, all_places_test = [], [], []
    all_grids_test, all_blocks_test = [], []
    for index in tqdm(range(1, 1001)):
        pbc.set_map(np.load(f"data/sr/test/sr_{index}.npy"))
        env.set_index(f"{index}")
        env.load_map("data/env/test/")
        grid = pbc.return_successormap()[:, :-1]
        place = np.copy(pbc.return_map())
        corner, border = pbc.structure_head_corner_stack()
        blocks = np.array([1 if s in env.blocks else 0 for s in range(100)])
        all_grids_test.append(grid.reshape(10, 10, -1))
        all_places_test.append(place.reshape(10, 10, -1))
        all_corners_test.append(corner.reshape(10, 10, -1))
        all_borders_test.append(border.reshape(10, 10, -1))
        all_blocks_test.append(blocks.reshape(10, 10, -1))
    all_grids_test = torch.tensor(np.array(all_grids_test)).float().to(device)
    all_places_test = torch.tensor(np.array(all_places_test)).float().to(device)
    all_corners_test = torch.tensor(np.array(all_corners_test)).float().to(device)
    all_borders_test = torch.tensor(np.array(all_borders_test)).float().to(device)
    all_blocks_test = torch.tensor(np.array(all_blocks_test)).float().to(device)
    all_grids_test = normalize_grid(all_grids_test)
    all_places_test = normalize_place(all_places_test)
    all_corners_transformed_test = transform_corner(all_corners_test)
    test_features = {
        "corner": all_corners_transformed_test,
        "border": all_borders_test,
        "place": all_places_test,
        "grid": all_grids_test,
        "block": all_blocks_test,
    }
    test_labels = torch.full((num_envs, grid_size, grid_size), 0).long().to(device)
    for index in tqdm(range(1, 1001)):
        env_idx = index - 1
        with open(f"data/skeletonize/nodes/test/v_{index}.pkl", "rb") as f:
            junction_nodes = pickle.load(f)
        with open(f"data/skeletonize/nodes/test/e_{index}.pkl", "rb") as f:
            edge_nodes = pickle.load(f)
        with open(f"data/skeletonize/nodes/test/d_{index}.pkl", "rb") as f:
            deadend_nodes = pickle.load(f)
        for node_list in deadend_nodes:
            for state in node_list:
                row, col = pbc.state_to_index(state)
                test_labels[env_idx, row, col] = 1.0
        for node_list in edge_nodes:
            for state in node_list:
                row, col = pbc.state_to_index(state)
                test_labels[env_idx, row, col] = 2.0
        for node_list in junction_nodes:
            for state in node_list:
                row, col = pbc.state_to_index(state)
                test_labels[env_idx, row, col] = 3.0

    os.makedirs("data/outer_loop", exist_ok=True)

    torch.save(train_features, "data/outer_loop/train_features.pth")
    torch.save(train_labels, "data/outer_loop/train_labels.pth")
    torch.save(test_features, "data/outer_loop/test_features.pth")
    torch.save(test_labels, "data/outer_loop/test_labels.pth")
    print("Training and test data have been saved to disk.")


def load_data(device_local=None):
    if device_local is not None:
        device = device_local
    train_features = torch.load("data/outer_loop/train_features.pth", map_location=device)
    train_labels = torch.load("data/outer_loop/train_labels.pth", map_location=device)
    test_features = torch.load("data/outer_loop/test_features.pth", map_location=device)
    test_labels = torch.load("data/outer_loop/test_labels.pth", map_location=device)
    print("Training and test data have been loaded successfully.")
    return train_features, train_labels, test_features, test_labels


def init_worker(gpu):
    global worker_gpu_id, device
    worker_gpu_id = gpu
    torch.cuda.set_device(worker_gpu_id)
    device = torch.device(f"cuda:{worker_gpu_id}")
    print(f"Worker initialized on GPU {worker_gpu_id}")


def run_training_task(args):
    seed, num_layers, nhead, lr, batch_size, combo = args
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_features, train_labels, test_features, test_labels = load_data()

    feature_names = ["corner", "border", "place", "grid"]
    feature_abbreviations = {"corner": "c", "border": "b", "place": "p", "grid": "g"}
    if combo:
        combo_abbr = "".join([feature_abbreviations[feat] for feat in combo])
    else:
        combo_abbr = "n"

    selected_train_features = []
    selected_test_features = []
    for feat in feature_names:
        if feat in combo:
            selected_train_features.append(train_features[feat])
            selected_test_features.append(test_features[feat])
        else:
            selected_train_features.append(torch.zeros_like(train_features[feat]))
            selected_test_features.append(torch.zeros_like(test_features[feat]))
    selected_train_features.append(train_features["block"])
    selected_test_features.append(test_features["block"])

    train_dataset = TensorDataset(*selected_train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    log_dir = f"data/outer_loop/runs/features_{combo_abbr}_layers_{num_layers}_heads_{nhead}_lr_{lr}_bs_{batch_size}_seed_{seed}"
    writer = SummaryWriter(log_dir=log_dir)

    print(
        f"Training: Seed {seed}, GPU {worker_gpu_id}, layers {num_layers}, heads {nhead}, lr {lr}, bs {batch_size}, features: {combo_abbr}"
    )

    model = TransformerClassifier(
        input_dim=input_dim, output_dim=4, num_layers=num_layers, nhead=nhead
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_epochs = 500
    train_and_test_model(
        model,
        train_loader,
        selected_test_features,
        test_labels,
        num_epochs,
        test_interval,
        optimizer,
        writer,
    )

    save_path = f"data/outer_loop/model_weights_dict_{combo_abbr}_{num_layers}_{nhead}_{lr}_{batch_size}_{seed}.pth"
    torch.save(model.state_dict(), save_path)
    writer.close()
    print(
        f"Completed: Seed {seed}, GPU {worker_gpu_id}, params: layers:{num_layers}, heads:{nhead}, lr:{lr}, bs:{batch_size}, combo:{combo_abbr} -> saved to {save_path}"
    )


def postprocess_outer_loop():
    # Path to the TensorBoard log directory
    num_layers = 6 # Number of Transformer encoder layers
    nhead = 8   # Number of attention heads
    lr = 0.001  # Learning rates
    batch_size = 10  # Batch sizes

    # Available features and dimensions
    feature_names = ['corner', 'border', 'place', 'grid']
    feature_dims = {
        "corner": 8,  # After transformation
        "border": 4,
        "place": 10,  # Projected place feature
        "grid": 9,
    }
    block_dim = 1  # Block feature is always included

    # Generate all combinations of vertex, border, place, and grid (excluding the block feature which is always included)
    feature_combinations = []
    for r in range(0, len(feature_names) + 1):
        feature_combinations.extend(combinations(feature_names, r))  # Generate 2^4 combinations of features

    # Loop over each feature combination, train and evaluate the model
    # Perform the hyperparameter search
    for combo in feature_combinations:
        # Define a mapping of features to abbreviations
        feature_abbreviations = {
            'corner': 'c',
            'border': 'b',
            'place': 'p',
            'grid': 'g'
        }
        # Generate abbreviated name from the combo
        if len(combo):
            combo_abbr = ''.join([feature_abbreviations[feat] for feat in combo])
        else:
            combo_abbr = 'n'
        base_log_dir = f"data/outer_loop/runs/features_{combo_abbr}_layers_{num_layers}_heads_{nhead}_lr_{lr}_bs_{batch_size}"
        pattern = base_log_dir + "*"

        log_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]

        all_batchloss = []
        all_epochloss = []
        all_testacc = []

        for log_dir in log_dirs:
            event_acc = event_accumulator.EventAccumulator(log_dir)
            event_acc.Reload()

            scalars_batchloss = event_acc.Scalars('Training Loss (Batch)')
            scalars_epochloss = event_acc.Scalars('Training Loss (Epoch)')
            scalars_testacc = event_acc.Scalars('Test Accuracy')

            values_batchloss = np.array([scalar.value for scalar in scalars_batchloss])
            values_epochloss = np.array([scalar.value for scalar in scalars_epochloss])
            values_testacc = np.array([scalar.value for scalar in scalars_testacc])

            all_batchloss.append(values_batchloss[-10000:])
            all_epochloss.append(values_epochloss[-500:])
            all_testacc.append(values_testacc[-100:])

        stacked_batchloss = np.stack(all_batchloss, axis=0)
        stacked_epochloss = np.stack(all_epochloss, axis=0)
        stacked_testacc = np.stack(all_testacc, axis=0)

        np.save('data/outer_loop/logs/' + combo_abbr + '_stacked_batchloss.npy', stacked_batchloss)
        np.save('data/outer_loop/logs/' + combo_abbr + '_stacked_epochloss.npy', stacked_epochloss)
        np.save('data/outer_loop/logs/' + combo_abbr + '_stacked_testacc.npy', stacked_testacc)



def check_data_availability():
    logs_dir = "data/outer_loop/logs/"
    
    if not os.path.exists(logs_dir):
        return False
    
    feature_names = ["corner", "border", "place", "grid"]
    feature_combinations = []
    for r in range(0, len(feature_names) + 1):
        feature_combinations.extend(combinations(feature_names, r))
    
    for combo in feature_combinations:
        feature_abbreviations = {"corner": "c", "border": "b", "place": "p", "grid": "g"}
        if combo:
            combo_abbr = "".join([feature_abbreviations[feat] for feat in combo])
        else:
            combo_abbr = "n"
        
        required_files = [
            f"{combo_abbr}_stacked_testacc.npy"
        ]
        
        for filename in required_files:
            file_path = os.path.join(logs_dir, filename)
            if not os.path.exists(file_path):
                return False
    
    return True


def process_outer_loop():    
    num_layers = 6
    nhead = 8
    lr = 0.001
    batch_size = 10
    seeds = [42, 7, 123, 2021]

    feature_names = ["corner", "border", "place", "grid"]
    feature_combinations = []
    for r in range(0, len(feature_names) + 1):
        feature_combinations.extend(combinations(feature_names, r))

    tasks = []
    for seed in seeds:
        for combo in feature_combinations:
            tasks.append((seed, num_layers, nhead, lr, batch_size, combo))

    for task in tasks:
        run_training_task(task)


def main():
    if not check_data_availability():
        save_data()
        process_outer_loop()
        postprocess_outer_loop()
