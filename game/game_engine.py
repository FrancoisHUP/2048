import random
import json

GRID_SIZE = 4

class GameEngine:
    def __init__(self):
        self.grid = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
        self.score = 0
        self.done = False  # Will be True when no more moves are possible
        self.reset()

    def reset(self):
        """Reset the board to start a new game."""
        self.grid = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
        self.score = 0
        self.done = False
        self._add_random_tile()
        self._add_random_tile()

    def step(self, action):
        """
        Perform a move in the environment.
        action is one of ['up','down','left','right'] or an integer representing that move.
        Returns: (new_grid, reward, done, info)
        - new_grid: the state after the move
        - reward: how much score was gained from merges this move
        - done: True if no moves left (game over)
        - info: optional debug info
        """
        if self.done:
            return self.grid, 0, True, {}

        old_score = self.score
        old_grid = [row[:] for row in self.grid]

        # 1. Apply the move
        self._move(action)

        # If the board didn’t change, no new tile and no reward
        if self.grid == old_grid:
            return self.grid, 0, self.done, {}

        # 2. Add a random tile
        self._add_random_tile()

        # 3. Check game over
        self.done = self._check_game_over()

        score_diff = self.score - old_score
        return self.grid, score_diff, self.done, {}

    def _add_random_tile(self):
        """Add a random tile of value 2 or 4 to an empty cell."""
        empty_cells = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if self.grid[i][j] == 0]
        if not empty_cells:
            return
        i, j = random.choice(empty_cells)
        self.grid[i][j] = 4 if random.random() < 0.1 else 2

    def _move(self, direction):
        """Apply move logic: up/down/left/right."""
        if direction == 'left':
            self.grid = self._slide_left(self.grid)
        elif direction == 'right':
            reversed_grid = [row[::-1] for row in self.grid]
            new_reversed = self._slide_left(reversed_grid)
            self.grid = [row[::-1] for row in new_reversed]
        elif direction == 'up':
            transposed = self._transpose(self.grid)
            moved = self._slide_left(transposed)
            self.grid = self._transpose(moved)
        elif direction == 'down':
            transposed = self._transpose(self.grid)
            reversed_grid = [row[::-1] for row in transposed]
            new_reversed = self._slide_left(reversed_grid)
            unreversed = [row[::-1] for row in new_reversed]
            self.grid = self._transpose(unreversed)

    def _slide_left(self, grid):
        """Slide everything left in a 2D grid and return the new grid."""
        new_grid = []
        for row in grid:
            compressed = self._compress(row)
            merged = self._merge(compressed)
            final = self._compress(merged)
            new_grid.append(final)
        return new_grid

    def _compress(self, row):
        """Push non-zero values to the front (left) of the row."""
        new_row = [num for num in row if num != 0]
        new_row += [0] * (GRID_SIZE - len(new_row))
        return new_row

    def _merge(self, row):
        """Merge adjacent equal tiles from left to right."""
        for i in range(GRID_SIZE - 1):
            if row[i] != 0 and row[i] == row[i+1]:
                row[i] *= 2
                self.score += row[i]  # add to the total score
                row[i+1] = 0
        return row

    def _transpose(self, grid):
        return [list(r) for r in zip(*grid)]

    def _check_game_over(self):
        """Check if there are no moves left."""
        # If there's an empty cell, not over
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] == 0:
                    return False
                # check horizontal neighbor
                if j < GRID_SIZE - 1 and self.grid[i][j] == self.grid[i][j+1]:
                    return False
                # check vertical neighbor
                if i < GRID_SIZE - 1 and self.grid[i][j] == self.grid[i+1][j]:
                    return False
        return True
    
    def is_valid_action(self, action):
        """Check if an action is valid (leads to a state change)."""
        original_grid = [row[:] for row in self.grid]

        # Simulate the move
        self._move(action)

        # Compare grids
        is_valid = self.grid != original_grid

        # Restore the original state
        self.grid = original_grid

        return is_valid

    def get_state(self):
        """Return a copy of the current grid (useful for AI)."""
        return [row[:] for row in self.grid]
    
    def print_grid(self):
        for row in self.grid:
            print(row) 

    def get_score(self):
        return self.score

    def is_done(self):
        return self.done

class GameRecorder:
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Clear the current recording buffer.
        """
        self.recording = []
        self.active = False

    def start(self):
        """
        Start recording moves.
        """
        self.reset()
        self.active = True

    def record_step(self, state, action, next_state, reward, done, score):
        """
        Record a single step: the old state, the action,
        the resulting next_state, the reward, whether done, and score.
        """
        if not self.active:
            return
        self.recording.append({
            "state": state,         # 4x4 grid before the move
            "action": action,       # e.g., 'left', 'right', etc.
            "next_state": next_state,
            "reward": reward,
            "done": done,
            "score": score
        })

    def stop(self):
        """
        Stop recording.
        """
        self.active = False

    def save_to_json(self, filename="game_record.json"):
        """
        Save the current recording as JSON.
        """
        with open(filename, "w") as f:
            json.dump(self.recording, f)
        # print(f"Game recording saved to {filename}")

    def load_from_json(self, filename="game_record.json"):
        """
        Load a recorded game from JSON file.
        Returns the loaded recording (list of steps).
        """
        with open(filename, "r") as f:
            data = json.load(f)
        return data