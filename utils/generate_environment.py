from modules.env import MazeEnvironment
import numpy as np
from tqdm import tqdm
import os


def generate_env(test, num):
    test_str = 'test' if test else 'train'
    row = 10
    column = 10
    env = MazeEnvironment(row, column)
    if test:
        ratio_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        for index in tqdm(range(1, num + 1)):
            env.set_index('{}'.format(index))
            ratio = np.random.choice(ratio_list)
            env.generate_random_blocks(ratio, ratio + 0.05)
            env.visualize(
                display=False, directory='data/env/{}/'.format(test_str)
            )
            env.save_map('data/env/{}/'.format(test_str))
    else:
        ratio = 0
        for ratio_index in tqdm(range(1, 11)):
            for index in tqdm(range(1, num + 1)):
                env.set_index('{}_{}'.format(ratio_index, index))
                env.generate_random_blocks(ratio, ratio + 0.05)
                env.visualize(
                    display=False, directory='data/env/{}/'.format(test_str)
                )
                env.save_map('data/env/{}/'.format(test_str))
            ratio += 0.05


def check_files_exist(test, num):
    """
    Check if the target files already exist
    """
    test_str = 'test' if test else 'train'
    base_dir = f'data/env/{test_str}/'
    
    if not os.path.exists(base_dir):
        return False
    
    if test:
        # Check if test files exist (1 to num)
        for index in range(1, num + 1):
            map_file = os.path.join(base_dir, f'blocks_{index}.npy')
            if not os.path.exists(map_file):
                return False
    else:
        # Check if train files exist (ratio_index_index format)
        for ratio_index in range(1, 11):
            for index in range(1, num + 1):
                map_file = os.path.join(base_dir, f'blocks_{ratio_index}_{index}.npy')
                if not os.path.exists(map_file):
                    return False
    
    return True


def main():
    test_num = 1000
    train_num = 100
    
    # Check if test files exist
    if not check_files_exist(True, test_num):
        if not os.path.exists('data/env/test'):
            os.makedirs('data/env/test')
        generate_env(True, test_num)
    
    # Check if train files exist
    if not check_files_exist(False, train_num):
        if not os.path.exists('data/env/train'):
            os.makedirs('data/env/train')
        generate_env(False, train_num)
