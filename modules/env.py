import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import copy
import seaborn as sns
from matplotlib.ticker import MultipleLocator


class Environment:
    def __init__(self, num_row, num_column):
        self.num_row = num_row
        self.num_column = num_column
        self.num_states = num_row * num_column
        self.map = np.zeros((num_row, num_column), dtype=bool)
        self.blocks = np.array([])
        self.nonblocks = np.arange(self.num_states)
        self.walls = []
        self.current_state = 0
        self.episode_step_count = 0
        self.index = None

    def set_index(self, index):
        self.index = str(index)

    def generate_random_blocks(self, min_ratio, max_ratio):
        try:
            while True:
                while True:
                    blocks = np.random.choice(
                        self.num_states,
                        np.random.randint(
                            np.floor(self.num_states * min_ratio),
                            np.ceil(self.num_states * max_ratio),
                        ),
                        replace=False,
                    )
                    self.update_map(blocks=blocks)
                    if self.is_connected():
                        break
                blocks_to_delete = []
                if self.goal in self.blocks:
                    blocks_to_delete.append(self.goal)
                if self.start in self.blocks:
                    blocks_to_delete.append(self.start)
                # rotation 고려해서 네 귀퉁이는 전부 제거
                if self.index_to_state(0, 0) in self.blocks:
                    blocks_to_delete.append(self.index_to_state(self.num_row - 1, 0))
                if self.index_to_state(0, self.num_column - 1) in self.blocks:
                    blocks_to_delete.append(self.index_to_state(0, self.num_column - 1))
                if (
                    self.index_to_state(self.num_row - 1, self.num_column - 1)
                    in self.blocks
                ):
                    blocks_to_delete.append(self.index_to_state(self.num_row - 1, 0))
                if self.index_to_state(0, self.num_column - 1) in self.blocks:
                    blocks_to_delete.append(self.index_to_state(0, self.num_column - 1))
                self.delete_block(blocks_to_delete)
                if self.is_connected():
                    break
        except:
            raise ValueError("min_ratio and max_ratio should be properly set")

    def return_adjacency_matrix(
        self, min_row=None, max_row=None, min_column=None, max_column=None
    ):
        if min_row is None:
            min_row = 0
        if max_row is None:
            max_row = self.num_row - 1
        if min_column is None:
            min_column = 0
        if max_column is None:
            max_column = self.num_column - 1
        nonblocks_cropped = []
        cropped_row = max_row + 1 - min_row
        cropped_column = max_column + 1 - min_column
        cropped_states = cropped_row * cropped_column
        for row in range(min_row, max_row + 1):
            for column in range(min_column, max_column + 1):
                state = self.index_to_state(row, column)
                if state in self.nonblocks:
                    nb_row, nb_col = self.state_to_index(state)
                    nonblocks_cropped.append(
                        self.index_to_state(
                            nb_row - min_row, nb_col - min_column, cropped_column
                        )
                    )
        nonblocks_cropped = np.array(nonblocks_cropped)

        adjacency_matrix = np.zeros(
            (np.shape(nonblocks_cropped)[0], np.shape(nonblocks_cropped)[0])
        )
        for index in range(np.shape(nonblocks_cropped)[0]):
            if (
                nonblocks_cropped[index] - cropped_column > -1
                and nonblocks_cropped[index] - cropped_column in nonblocks_cropped
            ):
                adjacency_matrix[
                    index,
                    np.where(
                        np.array(nonblocks_cropped)
                        == nonblocks_cropped[index] - cropped_column
                    ),
                ] = True
            if (
                nonblocks_cropped[index] + cropped_column < cropped_states
                and nonblocks_cropped[index] + cropped_column in nonblocks_cropped
            ):
                adjacency_matrix[
                    index,
                    np.where(
                        np.array(nonblocks_cropped)
                        == nonblocks_cropped[index] + cropped_column
                    ),
                ] = True
            if (
                nonblocks_cropped[index] % cropped_column > 0
                and nonblocks_cropped[index] - 1 in nonblocks_cropped
            ):
                adjacency_matrix[
                    index,
                    np.where(
                        np.array(nonblocks_cropped) == nonblocks_cropped[index] - 1
                    ),
                ] = True
            if (
                nonblocks_cropped[index] % cropped_column < cropped_column - 1
                and nonblocks_cropped[index] + 1 in nonblocks_cropped
            ):
                adjacency_matrix[
                    index,
                    np.where(
                        np.array(nonblocks_cropped) == nonblocks_cropped[index] + 1
                    ),
                ] = True
        return adjacency_matrix

    def index_to_state(self, row, column, num_column=None):
        if num_column is None:
            num_column = self.num_column
        if column >= num_column:
            raise ValueError("Column outside of range")
        return row * num_column + column

    def state_to_index(self, state, num_column=None):
        if num_column is None:
            num_column = self.num_column
        row, column = np.divmod(state, num_column)
        return int(row), int(column)

    def update_map(self, blocks=None, walls=None):
        if blocks is None:
            blocks = np.array([])
        if walls is None:
            walls = np.array([])
        self.blocks = blocks
        self.nonblocks = np.setdiff1d(np.arange(self.num_states), self.blocks)
        self.map = np.zeros((self.num_row, self.num_column), dtype=bool)
        for i in self.blocks:
            self.map[self.state_to_index(i)] = True
        for i in self.nonblocks:
            self.map[self.state_to_index(i)] = False
        self.walls = [(x[0], x[1]) for x in walls]

    def load_map(self, directory):

        if os.path.exists(directory + "blocks_" + self.index + ".npy"):
            blocks = np.load(directory + "blocks_" + self.index + ".npy")
        else:
            blocks = None
        if os.path.exists(directory + "walls_" + self.index + ".npy"):
            walls = np.load(directory + "walls_" + self.index + ".npy")
        else:
            walls = None
        self.update_map(blocks, walls)

    def save_map(self, directory):
        np.save(directory + "blocks_" + self.index + ".npy", self.blocks)
        np.save(directory + "walls_" + self.index + ".npy", self.walls)

    def concat_map(self, partial_env, shift_factor, scale_factor, rotation_factor):
        partial_env.scale_map(scale_factor)
        partial_env.rotate_map(rotation_factor)
        env = Environment(self.num_row, self.num_column)
        new_blocks = []
        for b in partial_env.return_blocks():
            brow, bcol = partial_env.state_to_index(b)
            new_blocks.append(env.index_to_state(brow, bcol))
        env.update_map(blocks=new_blocks)
        env.shift_map_wo_expansion(shift_factor)
        self.insert_block(env.return_blocks())

    def add_padding(self, padding_thickness):
        num_column_prev = self.num_column
        self.num_row += 2 * padding_thickness
        self.num_column += 2 * padding_thickness
        self.num_states = self.num_row * self.num_column
        new_blocks = []
        for block in self.blocks:
            block_row, block_column = self.state_to_index(block, num_column_prev)
            new_blocks.append(
                self.index_to_state(
                    block_row + padding_thickness, block_column + padding_thickness
                )
            )
        self.blocks = new_blocks
        new_walls = []
        for wall in self.walls:
            wall_start, wall_end = wall
            wall_start_row, wall_start_column = self.state_to_index(
                wall_start, num_column_prev
            )
            wall_end_row, wall_end_column = self.state_to_index(
                wall_end, num_column_prev
            )
            new_wall_start = self.index_to_state(
                wall_start_row + padding_thickness,
                wall_start_column + padding_thickness,
            )
            new_wall_end = self.index_to_state(
                wall_end_row + padding_thickness, wall_end_column + padding_thickness
            )
            new_walls.append((new_wall_start, new_wall_end))
        self.walls = new_walls

    def shift_blocks(self, shift_factor, expansion=False):
        row_shift, column_shift = shift_factor
        new_blocks = []
        new_num_row = self.num_row + np.abs(row_shift) if expansion else self.num_row
        new_num_column = (
            self.num_column + np.abs(column_shift) if expansion else self.num_column
        )
        for block in self.blocks:
            block_row, block_column = self.state_to_index(block)
            if expansion:
                block_row = block_row + row_shift if row_shift > 0 else block_row
                block_column = (
                    block_column + column_shift if column_shift > 0 else block_column
                )
            else:
                block_row += row_shift
                block_column += column_shift
            if 0 <= block_row < new_num_row and 0 <= block_column < new_num_column:
                new_blocks.append(
                    self.index_to_state(block_row, block_column, new_num_column)
                )
        return new_blocks

    def shift_walls(self, shift_factor, expansion=False):
        row_shift, column_shift = shift_factor
        new_walls = []
        new_num_row = self.num_row + np.abs(row_shift) if expansion else self.num_row
        new_num_column = (
            self.num_column + np.abs(column_shift) if expansion else self.num_column
        )
        for wall in self.walls:
            wall_start, wall_end = wall
            wall_start_row, wall_start_column = self.state_to_index(wall_start)
            wall_end_row, wall_end_column = self.state_to_index(wall_end)
            if expansion:
                wall_start_row = (
                    wall_start_row + row_shift if row_shift > 0 else wall_start_row
                )
                wall_start_column = (
                    wall_start_column + column_shift
                    if column_shift > 0
                    else wall_start_column
                )
                wall_end_row = (
                    wall_end_row + row_shift if row_shift > 0 else wall_end_row
                )
                wall_end_column = (
                    wall_end_column + column_shift
                    if column_shift > 0
                    else wall_end_column
                )
            else:
                wall_start_row += row_shift
                wall_start_column += column_shift
                wall_end_row += row_shift
                wall_end_column += column_shift
            if (
                0 <= wall_start_row < new_num_row
                and 0 <= wall_start_column < new_num_column
                and 0 <= wall_end_row < new_num_row
                and 0 <= wall_end_column < new_num_column
            ):
                new_walls.append(
                    (
                        self.index_to_state(wall_start_row, wall_start_column),
                        self.index_to_state(wall_end_row, wall_end_column),
                    )
                )
        return new_walls

    def shift_map_wo_expansion(self, shift_factor):
        new_blocks = self.shift_blocks(shift_factor)
        new_walls = self.shift_walls(shift_factor)
        self.update_map(blocks=new_blocks, walls=new_walls)

    def shift_map_w_expansion(self, shift_factor):
        for i in range(np.abs(shift_factor[0]) + np.abs(shift_factor[1])):
            row_shift = np.sign(shift_factor[0]) if i < np.abs(shift_factor[0]) else 0
            column_shift = (
                np.sign(shift_factor[1]) if i >= np.abs(shift_factor[0]) else 0
            )
            """if i == 0 and row_shift == 0:
                continue
            if i == 1 and column_shift == 0:
                continue"""
            new_num_row = self.num_row + np.abs(row_shift)
            new_num_column = self.num_column + np.abs(column_shift)
            new_blocks = self.shift_blocks((row_shift, column_shift), True)
            new_walls = self.shift_walls((row_shift, column_shift), True)
            border_blocks = [[] for _ in range(4)]
            for b in self.blocks:
                br, bc = self.state_to_index(b)
                if bc == 0:  # left
                    border_blocks[0].append(br)
                if bc == self.num_column - 1:  # right
                    border_blocks[1].append(br)
                if br == 0:  # up
                    border_blocks[2].append(bc)
                if br == self.num_row - 1:  # down
                    border_blocks[3].append(bc)

            if row_shift < 0:  # moving up
                for bc in border_blocks[3]:
                    new_blocks = np.append(
                        new_blocks,
                        self.index_to_state(new_num_row - 1, bc, new_num_column),
                    )
            elif row_shift > 0:  # moving down
                for bc in border_blocks[2]:
                    new_blocks = np.append(
                        new_blocks, self.index_to_state(0, bc, new_num_column)
                    )
            else:
                pass
            if column_shift < 0:  # moving left
                for br in border_blocks[1]:
                    new_blocks = np.append(
                        new_blocks,
                        self.index_to_state(br, new_num_column - 1, new_num_column),
                    )
            elif column_shift > 0:  # moving right
                for br in border_blocks[0]:
                    new_blocks = np.append(
                        new_blocks, self.index_to_state(br, 0, new_num_column)
                    )
            else:
                pass
            self.num_row = new_num_row
            self.num_column = new_num_column
            self.num_states = self.num_row * self.num_column
            self.update_map(blocks=new_blocks, walls=new_walls)

    def scale_map(self, scale_factor):
        scaled_row = int(self.num_row * scale_factor)
        scaled_column = int(self.num_column * scale_factor)
        new_blocks = []
        new_walls = []
        if scale_factor >= 1:
            for block in self.blocks:
                block_row, block_col = self.state_to_index(int(block))
                for new_block_row in range(
                    block_row * scale_factor, (block_row + 1) * scale_factor
                ):
                    for new_block_col in range(
                        block_col * scale_factor, (block_col + 1) * scale_factor
                    ):
                        new_blocks.append(
                            self.index_to_state(
                                new_block_row,
                                new_block_col,
                                self.num_column * scale_factor,
                            )
                        )
            for wall in self.walls:
                wall_start, wall_end = wall
                wall_start_row, wall_start_column = self.state_to_index(wall_start)
                wall_end_row, wall_end_column = self.state_to_index(wall_end)
                if wall_start_row == wall_end_row:
                    new_wall_start_column = wall_start_column * scale_factor + (scale_factor - 1)
                    new_wall_end_column = wall_end_column * scale_factor
                    for scale_index in range(scale_factor):
                        new_wall_start_row = wall_start_row * scale_factor + scale_index
                        new_wall_end_row = wall_end_row * scale_factor + scale_index
                        new_walls.append((
                            self.index_to_state(
                                new_wall_start_row,
                                new_wall_start_column,
                                self.num_column * scale_factor,
                            ), self.index_to_state(
                                new_wall_end_row,
                                new_wall_end_column,
                                self.num_column * scale_factor,
                            )
                        ))
                elif wall_start_column == wall_end_column:
                    new_wall_start_row = wall_start_row * scale_factor + (scale_factor - 1)
                    new_wall_end_row = wall_end_row * scale_factor
                    for scale_index in range(scale_factor):
                        new_wall_start_column = wall_start_column * scale_factor + scale_index
                        new_wall_end_column = wall_end_column * scale_factor + scale_index
                        new_walls.append((
                            self.index_to_state(
                                new_wall_start_row,
                                new_wall_start_column,
                                self.num_column * scale_factor,
                            ), self.index_to_state(
                                new_wall_end_row,
                                new_wall_end_column,
                                self.num_column * scale_factor,
                            )
                        ))
                else:
                    raise ValueError()
        else:
            raise ValueError("Not yet implemented")
        self.num_row = scaled_row
        self.num_column = scaled_column
        self.num_states = self.num_row * self.num_column
        self.update_map(blocks=new_blocks, walls=new_walls)

    def rotate_map(self, rotation_factor):
        for _ in range(rotation_factor):
            new_blocks = []
            new_walls = []
            for block in self.blocks:
                block_row, block_column = self.state_to_index(block)
                new_blocks.append(
                    self.index_to_state(
                        self.num_column - 1 - block_column, block_row, self.num_row
                    )
                )
            for wall in self.walls:
                wall_start, wall_end = wall
                wall_start_row, wall_start_column = self.state_to_index(wall_start)
                wall_end_row, wall_end_column = self.state_to_index(wall_end)
                if wall_start_row == wall_end_row:
                    new_walls.append(
                        (
                            self.index_to_state(
                                self.num_column - 1 - wall_end_column,
                                wall_end_row,
                                self.num_row,
                            ),
                            self.index_to_state(
                                self.num_column - 1 - wall_start_column,
                                wall_start_row,
                                self.num_row,
                            ),
                        )
                    )
                elif wall_start_column == wall_end_column:
                    new_walls.append(
                        (
                            self.index_to_state(
                                self.num_column - 1 - wall_start_column,
                                wall_start_row,
                                self.num_row,
                            ),
                            self.index_to_state(
                                self.num_column - 1 - wall_end_column,
                                wall_end_row,
                                self.num_row,
                            ),
                        )
                    )
                else:
                    raise ValueError()
            column = self.num_column
            self.num_column = self.num_row
            self.num_row = column
            self.update_map(blocks=new_blocks, walls=new_walls)

    def insert_block(self, blocks):
        self.blocks = np.insert(self.blocks, len(self.blocks), blocks)
        self.nonblocks = np.setdiff1d(np.arange(self.num_states), self.blocks)

    def delete_block(self, blocks):
        if isinstance(blocks, int):
            self.blocks = np.delete(self.blocks, np.where(self.blocks == blocks))
        else:
            for block in blocks:
                self.blocks = np.delete(self.blocks, np.where(self.blocks == block))
        self.nonblocks = np.setdiff1d(np.arange(self.num_states), self.blocks)

    def reset_structure(self):
        """
        아직 wall 지우는건 미포함임.
        """
        self.delete_block(copy.deepcopy(self.blocks))

    def return_blocks(self):
        return self.blocks

    def return_nonblocks(self):
        return self.nonblocks

    def return_current_state(self):
        return self.current_state

    def is_connected(
        self, min_row=None, max_row=None, min_column=None, max_column=None
    ):
        if (
            min_row is None
            or max_row is None
            or min_column is None
            or max_column is None
        ):
            min_row = 0
            max_row = self.num_row - 1
            min_column = 0
            max_column = self.num_column - 1
        adjacency_matrix = self.return_adjacency_matrix(
            min_row, max_row, min_column, max_column
        )
        visited = [False] * np.shape(adjacency_matrix)[0]
        dfs(visited, 0, adjacency_matrix)
        return bool(np.prod(visited))

class MazeEnvironment(Environment):
    def __init__(
        self,
        num_row,
        num_column,
        auto_reset=True,
    ):
        super(MazeEnvironment, self).__init__(num_row, num_column)
        self.action_list = ["left", "right", "up", "down"]
        self.start = 0
        self.goal = self.num_states - 1
        self.episode_step_max = 1000
        self.auto_reset = auto_reset
        self.goal_reward = 5
        self.puddle_reward = -1
        self.puddle_states = []
        self.trajectory = [self.start]

    def step(self, action):
        row, column = self.state_to_index(self.current_state)

        if self.action_list[action] == "left":
            if column >= 1 and self.current_state - 1 in self.nonblocks:
                if (self.index_to_state(row, column - 1), self.current_state) not in self.walls:
                    column -= 1
        if self.action_list[action] == "right":
            if (
                column <= self.num_column - 2
                and self.current_state + 1 in self.nonblocks
            ):
                if (self.current_state, self.index_to_state(row, column + 1)) not in self.walls:
                    column += 1
        if self.action_list[action] == "up":
            if row >= 1 and self.current_state - self.num_column in self.nonblocks:
                if (self.index_to_state(row - 1, column), self.current_state) not in self.walls:
                    row -= 1
        if self.action_list[action] == "down":
            if (
                row <= self.num_row - 2
                and self.current_state + self.num_column in self.nonblocks
            ):
                if (self.current_state, self.index_to_state(row + 1, column)) not in self.walls:
                    row += 1
        self.current_state = row * self.num_column + column

        next_state = self.current_state
        self.trajectory.append(next_state)

        self.episode_step_count += 1

        if self.current_state == self.goal:
            reward = self.goal_reward
            done = True
            if self.auto_reset:
                self.reset()
        elif self.episode_step_count >= self.episode_step_max:
            reward = 0
            done = True
            if self.auto_reset:
                self.reset()
        else:
            if self.current_state in self.puddle_states:
                reward = self.puddle_reward
            else:
                reward = 0
            done = False

        '''if self.auto_reset:
            done = True
            self.reset()'''

        return next_state, reward, done, self.episode_step_count

    def reset(self):
        self.episode_step_count = 0
        self.current_state = self.start
        self.trajectory = [self.start]
        return self.current_state

    def visualize(
        self,
        display=False,
        directory="./display/",
        puddle=False,
        trajectory=False,
        no_startgoal=False,
    ):
        vis = np.zeros((self.num_row, self.num_column))
        if puddle:
            for s in self.puddle_states:
                row, column = self.state_to_index(s)
                vis[row, column] = 2
        if not no_startgoal:
            srow, scol = self.state_to_index(self.start)
            vis[srow, scol] = 1
            if self.goal is not None:
                grow, gcol = self.state_to_index(self.goal)
                vis[grow, gcol] = 0.5
        for b in self.blocks:
            brow, bcol = self.state_to_index(b)
            vis[brow, bcol] = 1.5
        plt.matshow(
            vis,
            cmap=colors.ListedColormap(
                ["white", "red", "blue", "black", "lightskyblue"]
            ),
            vmin=0,
            vmax=2,
        )
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
        for w in self.walls:
            w_1, w_2 = w
            wrow_1, wcol_1 = self.state_to_index(w_1)
            wrow_2, wcol_2 = self.state_to_index(w_2)
            if wrow_1 == wrow_2:
                plt.vlines(
                    (wcol_1 + wcol_2) / 2,
                    wrow_1 - 1 / 2,
                    wrow_1 + 1 / 2,
                    colors="black",
                    linewidth=3

                )
            elif wcol_1 == wcol_2:
                plt.hlines(
                    (wrow_1 + wrow_2) / 2,
                    wcol_1 - 1 / 2,
                    wcol_1 + 1 / 2,
                    colors="black",
                    linewidth=3
                )
            else:
                raise ValueError()

        if trajectory:
            arrow_set = []
            if len(np.array(self.trajectory).shape) == 1:
                for s_index in range(len(self.trajectory) - 1):
                    s = self.trajectory[s_index]
                    y, x = self.state_to_index(s)
                    s_prime = self.trajectory[s_index + 1]
                    y_prime, x_prime = self.state_to_index(s_prime)
                    dx = x_prime - x
                    dy = y_prime - y
                    arrow_set.append((x, y, dx, dy))
            else:
                for t_index in range(len(self.trajectory)):
                    for s_index in range(len(self.trajectory[t_index]) - 1):
                        s = self.trajectory[t_index][s_index]
                        y, x = self.state_to_index(s)
                        s_prime = self.trajectory[t_index][s_index + 1]
                        y_prime, x_prime = self.state_to_index(s_prime)
                        dx = x_prime - x
                        dy = y_prime - y
                        arrow_set.append((x, y, dx, dy))
            arrow_set = list(set(arrow_set))
            for x, y, dx, dy in arrow_set:
                plt.arrow(x, y, dx * 0.8, dy * 0.8, head_width=0.1, color="black")

        if not display:
            if directory is None:
                directory = "./"
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(directory + "plot_" + self.index + ".png")
            plt.close()
        else:
            plt.show()

    def get_available_actions(self):
        actions = [False, False, False, False]
        row, column = self.state_to_index(self.current_state, self.num_column)
        if column >= 1 and self.current_state - 1 in self.nonblocks:
            actions[0] = True
        if column <= self.num_column - 2 and self.current_state + 1 in self.nonblocks:
            actions[1] = True
        if row >= 1 and self.current_state - self.num_column in self.nonblocks:
            actions[2] = True
        if (
            row <= self.num_row - 2
            and self.current_state + self.num_column in self.nonblocks
        ):
            actions[3] = True
        return np.array(actions, dtype=bool)

    def set_random_start_goal_states(self):
        while True:
            self.start, self.goal = np.random.choice(self.nonblocks, 2, replace=False)
            if (
                abs(self.start - self.goal) != 1
                and abs(self.start - self.goal) != self.num_column
            ):
                break

    def set_start_goal_states(self, start, goal):
        self.start = start
        self.goal = goal
        self.trajectory = [self.start]

    def set_goal_reward(self, goal_reward):
        self.goal_reward = goal_reward

    def set_puddle_reward(self, puddle_reward):
        self.puddle_reward = puddle_reward

    def set_puddle_states(self, puddle_states):
        self.puddle_states = list(set(puddle_states))

    def set_puddle_states_quantile(self, puddle_size, puddle_loc):
        self.puddle_cases = []
        h_half = self.num_row // 2
        w_half = self.num_column // 2

        if puddle_loc == 1:  # 왼쪽 위 사분면
            x_bounds = (0, h_half)
            y_bounds = (0, w_half)
        elif puddle_loc == 2:  # 오른쪽 위 사분면
            x_bounds = (0, h_half)
            y_bounds = (w_half, self.num_column)
        elif puddle_loc == 3:  # 왼쪽 아래 사분면
            x_bounds = (h_half, self.num_row)
            y_bounds = (0, w_half)
        elif puddle_loc == 4:  # 오른쪽 아래 사분면
            x_bounds = (h_half, self.num_row)
            y_bounds = (w_half, self.num_column)

        for x in range(x_bounds[0], x_bounds[1] - puddle_size + 1):  # 0, 1, 2
            for y in range(y_bounds[0], y_bounds[1] - puddle_size + 1):  # 0, 1, 2
                puddle_states = [
                    self.index_to_state(x + i, y + j)
                    for i in range(puddle_size)
                    for j in range(puddle_size)
                ]

                if all(
                    state not in self.blocks
                    and state != self.start
                    and state != self.goal
                    for state in puddle_states
                ):
                    self.puddle_cases.append(puddle_states)

    def return_start(self):
        return self.start

    def return_goal(self):
        return self.goal

    def random_walk_policy(self, rd_prob=0.5):
        lu_prob = 1 - rd_prob
        probs = [lu_prob / 2, rd_prob / 2, lu_prob / 2, rd_prob / 2]
        av_actions = self.get_available_actions()
        unav_actions = 1 - av_actions
        unav_prob = np.sum(probs * unav_actions)
        av_probs = probs * av_actions
        av_props = av_probs / np.sum(av_probs)
        av_probs += unav_prob * av_props
        return np.random.choice(4, p=av_probs)

    def shift_map_w_expansion(self, shift_factor):
        super().shift_map_w_expansion(shift_factor)
        self.goal = self.num_states - 1

    def scale_map(self, scale_factor):
        super().scale_map(scale_factor)
        self.goal = self.num_states - 1


class MazeEnvironmentMultisubgoal(MazeEnvironment):

    def __init__(
        self,
        num_row=10,
        num_column=10,
        num_features=3,
        num_states_per_feature=4,
        auto_reset=False,
        seed=0,
    ):
        assert num_row == num_column
        super(MazeEnvironmentMultisubgoal, self).__init__(
            num_row, num_column, auto_reset
        )
        np.random.seed(seed)
        self.num_features = num_features
        self.num_states_per_feature = num_states_per_feature
        self.rewards_per_features = np.ones(self.num_features)
        self.features_states = None
        self.features_states_copy = None
        self.state_features = None
        self.set_random_feature_states()

    def load_map(self, directory):
        super().load_map(directory)
        self.set_random_feature_states()

    def set_random_feature_states(self):
        self.features_states = np.random.choice(
            np.setdiff1d(self.nonblocks, [self.start, self.goal]),
            (self.num_features, self.num_states_per_feature),
            replace=False,
        )
        self.features_states_copy = np.copy(self.features_states)
        self.state_features = self.construct_state_features()

    def set_feature_reward(self, rewards):
        assert len(rewards) == self.num_features
        self.rewards_per_features = rewards

    def return_state_reward(self):
        reward = np.zeros(self.num_states)
        reward[self.goal] = self.goal_reward
        for feature in range(self.num_features):
            for state in self.features_states[feature]:
                reward[state] = self.rewards_per_features[feature]
        return reward

    def visualize(
            self, display=False, directory=None, include_agent=True, trajectory=False, no_startgoal=False
    ):
        vis = np.zeros((self.num_row, self.num_column))
        srow, scol = self.state_to_index(self.start)
        vis[srow, scol] = 1
        if not no_startgoal:
            srow, scol = self.state_to_index(self.start)
            vis[srow, scol] = 1
            if self.goal is not None:
                grow, gcol = self.state_to_index(self.goal)
                vis[grow, gcol] = 0.5

        for b in self.blocks:
            brow, bcol = self.state_to_index(b)
            vis[brow, bcol] = 1.5
        val = 2
        for feature_index in range(self.num_features):
            for state in self.features_states[feature_index]:
                if state >= 0:
                    frow, fcol = self.state_to_index(state)
                    vis[frow, fcol] = val
            val += 0.5
        if include_agent:
            arow, acol = self.state_to_index(self.current_state)
            vis[arow, acol] = -0.5
            plt.matshow(
                vis,
                cmap=colors.ListedColormap(
                    [
                        "yellow",
                        "white",
                        "red",
                        "blue",
                        "black",
                        "orange",
                        "green",
                        "pink",
                        "olive",
                        "cyan",
                    ][: 4 + self.num_features + 1]
                ),
            )
        else:
            plt.matshow(
                vis,
                cmap=colors.ListedColormap(
                    [
                        "white",
                        "red",
                        "blue",
                        "black",
                        "orange",
                        "green",
                        "pink",
                        "olive",
                        "cyan",
                    ][: 4 + self.num_features]
                ),
            )
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
        for w in self.walls:
            w_1, w_2 = w
            wrow_1, wcol_1 = self.state_to_index(w_1)
            wrow_2, wcol_2 = self.state_to_index(w_2)
            if wrow_1 == wrow_2:
                plt.vlines(
                    (wcol_1 + wcol_2) / 2,
                    wrow_1 - 1 / 2,
                    wrow_1 + 1 / 2,
                    colors="black",
                    linewidth=3
                )
            elif wcol_1 == wcol_2:
                plt.hlines(
                    (wrow_1 + wrow_2) / 2,
                    wcol_1 - 1 / 2,
                    wcol_1 + 1 / 2,
                    colors="black",
                    linewidth=3
                )
            else:
                raise ValueError()
        if trajectory:
            arrow_set = []
            if len(np.array(self.trajectory).shape) == 1:
                for s_index in range(len(self.trajectory) - 1):
                    s = self.trajectory[s_index]
                    y, x = self.state_to_index(s)
                    s_prime = self.trajectory[s_index + 1]
                    y_prime, x_prime = self.state_to_index(s_prime)
                    dx = x_prime - x
                    dy = y_prime - y
                    arrow_set.append((x, y, dx, dy))
            else:
                for t_index in range(len(self.trajectory)):
                    for s_index in range(len(self.trajectory[t_index]) - 1):
                        s = self.trajectory[t_index][s_index]
                        y, x = self.state_to_index(s)
                        s_prime = self.trajectory[t_index][s_index + 1]
                        y_prime, x_prime = self.state_to_index(s_prime)
                        dx = x_prime - x
                        dy = y_prime - y
                        arrow_set.append((x, y, dx, dy))
            arrow_set = list(set(arrow_set))
            for x, y, dx, dy in arrow_set:
                plt.arrow(x, y, dx * 0.8, dy * 0.8, head_width=0.1, color="black")
        if not display:
            if directory is None:
                directory = "./"
            plt.savefig(directory + "plot_" + self.index + ".png")
            plt.close()
        else:
            plt.show()

    def return_state_features(self):
        return copy.deepcopy(self.state_features)

    def return_features_weight(self):
        return np.hstack([self.rewards_per_features, self.goal_reward])

    def construct_state_features(self):
        state_features = np.zeros((self.num_states, self.num_features + 1), dtype=int)
        for feature in range(self.num_features):
            for state in self.features_states[feature]:
                state_features[int(state)][int(feature)] = 1
        state_features[self.goal][-1] = 1
        return state_features

    def reconstruct_features(self):
        self.features_states = copy.deepcopy(self.features_states_copy)
        self.state_features = self.construct_state_features()

    def step(self, action):
        state, reward, done, self.step_count = super(
            MazeEnvironmentMultisubgoal, self
        ).step(action)
        if done:
            self.reconstruct_features()
        elif self.current_state in self.features_states:
            feature = np.where(self.features_states == self.current_state)[0][0]
            state_index = np.where(self.features_states == self.current_state)[1][0]
            reward = self.rewards_per_features[feature]
            self.features_states[feature][state_index] = -1
            self.state_features[self.current_state][feature] = 0
            done = False

        return (
            state,
            reward,
            done,
            self.step_count,
        )


def dfs(visited, start_index, adjacency_matrix):
    visited[start_index] = True
    for child in np.where(adjacency_matrix[start_index] == True)[0]:
        if not visited[child]:
            dfs(visited, child, adjacency_matrix)


def is_connected(row, column, blocks):
    num_states = row * column
    nonblocks = np.setdiff1d(np.arange(row * column), blocks)
    adjacency_matrix = np.zeros((np.shape(nonblocks)[0], np.shape(nonblocks)[0]))
    for index in range(np.shape(nonblocks)[0]):
        if nonblocks[index] - column > -1 and nonblocks[index] - column in nonblocks:
            adjacency_matrix[
                index, np.where(np.array(nonblocks) == nonblocks[index] - column)
            ] = True
        if (
            nonblocks[index] + column < num_states
            and nonblocks[index] + column in nonblocks
        ):
            adjacency_matrix[
                index, np.where(np.array(nonblocks) == nonblocks[index] + column)
            ] = True
        if nonblocks[index] % column > 0 and nonblocks[index] - 1 in nonblocks:
            adjacency_matrix[
                index, np.where(np.array(nonblocks) == nonblocks[index] - 1)
            ] = True
        if nonblocks[index] % column < column - 1 and nonblocks[index] + 1 in nonblocks:
            adjacency_matrix[
                index, np.where(np.array(nonblocks) == nonblocks[index] + 1)
            ] = True
    visited = [False] * np.shape(adjacency_matrix)[0]
    dfs(visited, 0, adjacency_matrix)
    return bool(np.prod(visited))


class MazeVisualization(MazeEnvironment):
    def __init__(self, row, column):
        super(MazeVisualization, self).__init__(row, column)

    def visualize(
            self,
            display=False,
            ax=None,
            directory="display/",
            puddle=False,
            trajectory=False,
            no_startgoal=False,
            vertex_states = [],
            edge_states = [],
            deadend_states = [],
            subgoals=[], 
            unexplored_states=[],
            set_grid=False,
            junction_color=sns.color_palette("pastel", 8)[3],
            deadend_color=sns.color_palette("pastel", 8)[4],
            edge_color=sns.color_palette("pastel", 8)[2],
    ):
        vis = np.zeros((self.num_row, self.num_column))
        if puddle:
            for s in self.puddle_states:
                row, column = self.state_to_index(s)
                vis[row, column] = 2
        if not no_startgoal:
            srow, scol = self.state_to_index(self.start)
            vis[srow, scol] = 1
            if self.goal is not None:
                grow, gcol = self.state_to_index(self.goal)
                vis[grow, gcol] = 0.5
        for b in self.blocks:
            brow, bcol = self.state_to_index(b)
            vis[brow, bcol] = 1.5
        for s in vertex_states:
            row, col = self.state_to_index(s)
            vis[row, col] = 2.5
        for s in edge_states:
            row, col = self.state_to_index(s)
            vis[row, col] = 3
        for s in deadend_states:
            row, col = self.state_to_index(s)
            vis[row, col] = 3.5
        for s in subgoals:
            row, col = self.state_to_index(s)
            vis[row, col] = 4
        for s in unexplored_states:
            row, col = self.state_to_index(s)
            vis[row, col] = 4.5
        if ax is None:
            raise ValueError("Please provide an axis")
        if set_grid:
            ax.matshow(
                vis,
                cmap=colors.ListedColormap(
                    ["white", "red", "blue", "black", "lightskyblue", 
                        junction_color, edge_color, 
                        deadend_color, "orange", "gray"]
                ),
                vmin=0,
                vmax=4.5,
                extent=[0, self.num_row, 0, self.num_column]
            )
        else:
            ax.matshow(
                vis,
                cmap=colors.ListedColormap(
                    ["white", "red", "blue", "black", "lightskyblue", 
                        junction_color, edge_color, 
                        deadend_color, "orange", "gray"]
                ),
                vmin=0,
                vmax=4.5,
            )
        ax.tick_params(
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
            bottom=False,
            top=False,
            labeltop=False,
        )
        # ax.grid(False)
        for w in self.walls:
            w_1, w_2 = w
            wrow_1, wcol_1 = self.state_to_index(w_1)
            wrow_2, wcol_2 = self.state_to_index(w_2)
            if wrow_1 == wrow_2:
                ax.vlines(
                    (wcol_1 + wcol_2) / 2,
                    wrow_1 - 1 / 2,
                    wrow_1 + 1 / 2,
                    colors="black",
                    linewidth=5

                )
            elif wcol_1 == wcol_2:
                ax.hlines(
                    (wrow_1 + wrow_2) / 2,
                    wcol_1 - 1 / 2,
                    wcol_1 + 1 / 2,
                    colors="black",
                    linewidth=5
                )
            else:
                raise ValueError()

        if trajectory:
            arrow_set = []
            if len(np.array(self.trajectory).shape) == 1:
                for s_index in range(len(self.trajectory) - 1):
                    s = self.trajectory[s_index]
                    y, x = self.state_to_index(s)
                    s_prime = self.trajectory[s_index + 1]
                    y_prime, x_prime = self.state_to_index(s_prime)
                    dx = x_prime - x
                    dy = y_prime - y
                    arrow_set.append((x, y, dx, dy))
            else:
                for t_index in range(len(self.trajectory)):
                    for s_index in range(len(self.trajectory[t_index]) - 1):
                        s = self.trajectory[t_index][s_index]
                        y, x = self.state_to_index(s)
                        s_prime = self.trajectory[t_index][s_index + 1]
                        y_prime, x_prime = self.state_to_index(s_prime)
                        dx = x_prime - x
                        dy = y_prime - y
                        arrow_set.append((x, y, dx, dy))
            arrow_set = list(set(arrow_set))
            for x, y, dx, dy in arrow_set:
                ax.arrow(x, y, dx * 0.8, dy * 0.8, head_width=0.1, color="black")
        if set_grid:
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.grid(True)
        else:
            ax.grid(False)
        return ax


if __name__ == "__main__":
    import copy

    env = MazeEnvironment(10, 10)
    env.step(1)
