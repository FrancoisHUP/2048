import tkinter as tk
import random
from tkinter import messagebox

# Constants for the game
GRID_SIZE = 4
CELL_SIZE = 100
CELL_PADDING = 10
BACKGROUND_COLOR = "#92877d"
EMPTY_CELL_COLOR = "#9e948a"

CELL_COLORS = {
    2: "#eee4da",
    4: "#ede0c8",
    8: "#f2b179",
    16: "#f59563",
    32: "#f67c5f",
    64: "#f65e3b",
    128: "#edcf72",
    256: "#edcc61",
    512: "#edc850",
    1024: "#edc53f",
    2048: "#edc22e",
}

FONT = ("Verdana", 24, "bold")

# Adjust these for faster animations
ANIMATION_STEPS = 3     # Smaller means fewer frames => faster
ANIMATION_DELAY = 20    # Milliseconds between each animation step => faster

class Game2048:
    def __init__(self, master):
        self.master = master
        master.title("2048 Game")
        master.resizable(False, False)

        # Game state
        self.grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.score = 0
        self.best_score = 0  # Keep track of best (high) score

        # If True, we do not accept new moves
        self.animation_in_progress = False

        # Top frame for Score, Best, and Restart button
        top_frame = tk.Frame(master)
        top_frame.pack(pady=10)

        self.score_label = tk.Label(top_frame, text=f"Score: {self.score}", font=("Verdana", 16))
        self.score_label.pack(side=tk.LEFT, padx=10)

        self.best_label = tk.Label(top_frame, text=f"Best: {self.best_score}", font=("Verdana", 16))
        self.best_label.pack(side=tk.LEFT, padx=10)

        self.restart_button = tk.Button(top_frame, text="Restart", font=("Verdana", 12), command=self.restart_game)
        self.restart_button.pack(side=tk.LEFT, padx=10)

        canvas_width = GRID_SIZE * (CELL_SIZE + CELL_PADDING) + CELL_PADDING
        canvas_height = GRID_SIZE * (CELL_SIZE + CELL_PADDING) + CELL_PADDING
        self.canvas = tk.Canvas(master, width=canvas_width, height=canvas_height, bg=BACKGROUND_COLOR)
        self.canvas.pack()

        self.draw_grid()
        self.add_random_tile()
        self.add_random_tile()
        self.draw_tiles()

        # Bind arrow keys
        master.bind("<Up>", self.move_up)
        master.bind("<Down>", self.move_down)
        master.bind("<Left>", self.move_left)
        master.bind("<Right>", self.move_right)

    # -------------------------------------------------------------------------
    #                   RESTART / NEW GAME BUTTON
    # -------------------------------------------------------------------------
    def restart_game(self):
        """Reset the game to an empty grid and reset score."""
        self.grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.score = 0
        self.animation_in_progress = False
        self.score_label.config(text=f"Score: {self.score}")
        self.draw_grid()
        self.add_random_tile()
        self.add_random_tile()
        self.draw_tiles()

    # -------------------------------------------------------------------------
    #                   DRAWING THE BOARD AND TILES
    # -------------------------------------------------------------------------
    def draw_grid(self):
        self.canvas.delete("all")
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x0 = CELL_PADDING + j * (CELL_SIZE + CELL_PADDING)
                y0 = CELL_PADDING + i * (CELL_SIZE + CELL_PADDING)
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=EMPTY_CELL_COLOR, outline="")

    def draw_tiles(self):
        self.draw_grid()
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                value = self.grid[i][j]
                if value != 0:
                    self.draw_single_tile(i, j, value)

        # Update score labels
        self.score_label.config(text=f"Score: {self.score}")
        if self.score > self.best_score:
            self.best_score = self.score
        self.best_label.config(text=f"Best: {self.best_score}")

        self.master.update_idletasks()

    def draw_single_tile(self, i, j, value, offset_x=0, offset_y=0):
        """
        Draws a single tile at (i,j). 
        offset_x/offset_y are used for animation (pixel offsets).
        """
        x0 = CELL_PADDING + j * (CELL_SIZE + CELL_PADDING) + offset_x
        y0 = CELL_PADDING + i * (CELL_SIZE + CELL_PADDING) + offset_y
        x1 = x0 + CELL_SIZE
        y1 = y0 + CELL_SIZE

        color = CELL_COLORS.get(value, "#3c3a32")
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
        x_center = x0 + CELL_SIZE / 2
        y_center = y0 + CELL_SIZE / 2
        self.canvas.create_text(x_center, y_center, text=str(value), font=FONT, fill="#776e65")

    # -------------------------------------------------------------------------
    #                   ADD A RANDOM TILE
    # -------------------------------------------------------------------------
    def add_random_tile(self):
        empty_cells = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if self.grid[i][j] == 0]
        if not empty_cells:
            return
        i, j = random.choice(empty_cells)
        self.grid[i][j] = 4 if random.random() < 0.1 else 2

    # -------------------------------------------------------------------------
    #       MOVE LOGIC (LEFT / RIGHT / UP / DOWN) + TRACKING TILE MOVEMENTS
    # -------------------------------------------------------------------------
    def move_left(self, event):
        if self.animation_in_progress:
            return
        self.animation_in_progress = True

        old_grid = [row[:] for row in self.grid]
        new_grid, moves = self.slide_left(old_grid)

        if new_grid == old_grid:  # No movement or merge
            self.animation_in_progress = False
            return

        self.animate_tiles(old_grid, new_grid, moves)
        self.grid = new_grid
        self.add_random_tile()
        self.draw_tiles()
        self.check_2048_tile()
        self.check_game_over()
        self.animation_in_progress = False

    def move_right(self, event):
        if self.animation_in_progress:
            return
        self.animation_in_progress = True

        old_grid = [row[:] for row in self.grid]
        reversed_grid = [row[::-1] for row in old_grid]
        slid_grid, moves = self.slide_left(reversed_grid)
        final_grid = [row[::-1] for row in slid_grid]

        fixed_moves = []
        for (r0, c0, r1, c1, val) in moves:
            old_c_fixed = GRID_SIZE - 1 - c0
            new_c_fixed = GRID_SIZE - 1 - c1
            fixed_moves.append((r0, old_c_fixed, r1, new_c_fixed, val))

        if final_grid == old_grid:
            self.animation_in_progress = False
            return

        self.animate_tiles(old_grid, final_grid, fixed_moves)
        self.grid = final_grid
        self.add_random_tile()
        self.draw_tiles()
        self.check_2048_tile()
        self.check_game_over()
        self.animation_in_progress = False

    def move_up(self, event):
        if self.animation_in_progress:
            return
        self.animation_in_progress = True

        old_grid = [row[:] for row in self.grid]
        transposed = self.transpose(old_grid)
        new_transposed, moves = self.slide_left(transposed)
        final_grid = self.transpose(new_transposed)

        fixed_moves = []
        for (r0, c0, r1, c1, val) in moves:
            fixed_moves.append((c0, r0, c1, r1, val))

        if final_grid == old_grid:
            self.animation_in_progress = False
            return

        self.animate_tiles(old_grid, final_grid, fixed_moves)
        self.grid = final_grid
        self.add_random_tile()
        self.draw_tiles()
        self.check_2048_tile()
        self.check_game_over()
        self.animation_in_progress = False

    def move_down(self, event):
        if self.animation_in_progress:
            return
        self.animation_in_progress = True

        old_grid = [row[:] for row in self.grid]
        transposed = self.transpose(old_grid)
        reversed_grid = [row[::-1] for row in transposed]
        slid_grid, moves = self.slide_left(reversed_grid)
        unreversed = [row[::-1] for row in slid_grid]
        final_grid = self.transpose(unreversed)

        fixed_moves = []
        for (r0, c0, r1, c1, val) in moves:
            c0_fixed = GRID_SIZE - 1 - c0
            c1_fixed = GRID_SIZE - 1 - c1
            fixed_moves.append((c0_fixed, r0, c1_fixed, r1, val))

        if final_grid == old_grid:
            self.animation_in_progress = False
            return

        self.animate_tiles(old_grid, final_grid, fixed_moves)
        self.grid = final_grid
        self.add_random_tile()
        self.draw_tiles()
        self.check_2048_tile()
        self.check_game_over()
        self.animation_in_progress = False

    # -------------------------------------------------------------------------
    #     slide_left: returns (new_grid, moves_list) describing tile movements
    # -------------------------------------------------------------------------
    def slide_left(self, old_grid):
        new_grid = []
        moves = []

        for row_idx in range(GRID_SIZE):
            old_row = old_grid[row_idx]

            # Gather non-zero tiles => list of (orig_col, value)
            temp = []
            for col_idx, val in enumerate(old_row):
                if val != 0:
                    temp.append((col_idx, val))

            final_row = [0] * GRID_SIZE
            write_pos = 0
            skip_next = False
            for i in range(len(temp)):
                if skip_next:
                    skip_next = False
                    continue

                col_i, val_i = temp[i]
                # Check if next tile can merge
                if i < len(temp) - 1 and val_i == temp[i+1][1]:
                    col_j, val_j = temp[i+1]
                    merged_val = val_i * 2
                    self.score += merged_val

                    # Merge: tile col_j merges into position `write_pos`
                    moves.append((row_idx, col_j, row_idx, write_pos, merged_val))
                    # If the left tile col_i also physically moved, track that
                    if col_i != write_pos:
                        moves.append((row_idx, col_i, row_idx, write_pos, merged_val))

                    final_row[write_pos] = merged_val
                    skip_next = True
                else:
                    # No merge => tile moves from col_i to write_pos
                    if col_i != write_pos:
                        moves.append((row_idx, col_i, row_idx, write_pos, val_i))
                    final_row[write_pos] = val_i

                write_pos += 1

            new_grid.append(final_row)

        return new_grid, moves

    # -------------------------------------------------------------------------
    #               ANIMATION: ONLY FOR TILES IN 'moves'
    # -------------------------------------------------------------------------
    def animate_tiles(self, old_grid, new_grid, moves):
        for step in range(1, ANIMATION_STEPS + 1):
            t = step / ANIMATION_STEPS

            self.canvas.delete("all")
            self.draw_grid()

            # 1. Draw tiles that are NOT moving
            moving_positions = set((m[2], m[3]) for m in moves)  # (new_i, new_j)
            original_positions = set((m[0], m[1]) for m in moves)

            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    val_final = new_grid[i][j]
                    # Draw if not in motion
                    if val_final != 0 and (i, j) not in moving_positions:
                        if (i, j) not in original_positions:
                            self.draw_single_tile(i, j, val_final)

            # 2. Draw moving tiles with interpolation
            for (old_i, old_j, new_i, new_j, val) in moves:
                row_diff = new_i - old_i
                col_diff = new_j - old_j
                offset_y = row_diff * (CELL_SIZE + CELL_PADDING) * t
                offset_x = col_diff * (CELL_SIZE + CELL_PADDING) * t
                self.draw_single_tile(old_i, old_j, val, offset_x=offset_x, offset_y=offset_y)

            self.master.update_idletasks()
            self.master.after(ANIMATION_DELAY)

    # -------------------------------------------------------------------------
    #          CHECK FOR 2048 TILE / CHECK GAME OVER
    # -------------------------------------------------------------------------
    def check_2048_tile(self):
        for row in self.grid:
            if 2048 in row:
                messagebox.showinfo("2048", "Congratulations! You created a 2048 tile!")
                return

    def check_game_over(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] == 0:
                    return
                if j < GRID_SIZE - 1 and self.grid[i][j] == self.grid[i][j + 1]:
                    return
                if i < GRID_SIZE - 1 and self.grid[i][j] == self.grid[i + 1][j]:
                    return

        messagebox.showinfo("2048", f"Game Over!\nYour Score: {self.score}")
        self.master.quit()

    def transpose(self, grid):
        return [list(row) for row in zip(*grid)]


def main():
    root = tk.Tk()
    game = Game2048(root)
    root.mainloop()

if __name__ == "__main__":
    main()
