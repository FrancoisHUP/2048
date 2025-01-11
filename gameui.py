import tkinter as tk
from tkinter import messagebox
import random

# For the GUI constants:
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

ANIMATION_STEPS = 3
ANIMATION_DELAY = 20


class Game2048GUI:
    def __init__(self, master, engine, ai_model=None):
        """
        master: Tk root
        engine: instance of GameEngine
        ai_model: your trained AI object (optional), with a method `predict_move(state) -> action`
        """
        self.master = master
        self.engine = engine
        self.ai_model = ai_model  # If provided, we can let the AI play
        self.animation_in_progress = False

        # GUI Setup
        master.title("2048 Game")
        master.resizable(False, False)

        top_frame = tk.Frame(master)
        top_frame.pack(pady=10)

        self.score_label = tk.Label(top_frame, text=f"Score: 0", font=("Verdana", 16))
        self.score_label.pack(side=tk.LEFT, padx=10)

        self.ai_button = tk.Button(top_frame, text="AI Move", font=("Verdana", 12), command=self.ai_move)
        self.ai_button.pack(side=tk.LEFT, padx=10)

        canvas_width = GRID_SIZE * (CELL_SIZE + CELL_PADDING) + CELL_PADDING
        canvas_height = GRID_SIZE * (CELL_SIZE + CELL_PADDING) + CELL_PADDING
        self.canvas = tk.Canvas(master, width=canvas_width, height=canvas_height, bg=BACKGROUND_COLOR)
        self.canvas.pack()

        # Key bindings
        master.bind("<Up>", self.on_up)
        master.bind("<Down>", self.on_down)
        master.bind("<Left>", self.on_left)
        master.bind("<Right>", self.on_right)

        # Initial draw
        self.draw_tiles()

    def draw_tiles(self):
        self.canvas.delete("all")
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x0 = CELL_PADDING + j * (CELL_SIZE + CELL_PADDING)
                y0 = CELL_PADDING + i * (CELL_SIZE + CELL_PADDING)
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=EMPTY_CELL_COLOR, outline="")

        grid = self.engine.get_state()
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                value = grid[i][j]
                if value != 0:
                    self.draw_single_tile(i, j, value)

        current_score = self.engine.get_score()
        self.score_label.config(text=f"Score: {current_score}")
        self.master.update_idletasks()

    def draw_single_tile(self, i, j, value):
        x0 = CELL_PADDING + j * (CELL_SIZE + CELL_PADDING)
        y0 = CELL_PADDING + i * (CELL_SIZE + CELL_PADDING)
        x1 = x0 + CELL_SIZE
        y1 = y0 + CELL_SIZE

        color = CELL_COLORS.get(value, "#3c3a32")
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
        x_center = x0 + CELL_SIZE / 2
        y_center = y0 + CELL_SIZE / 2
        self.canvas.create_text(x_center, y_center, text=str(value), font=FONT, fill="#776e65")

    # -------------------------------------------------------------------------
    #                        KEYBOARD CONTROLS
    # -------------------------------------------------------------------------
    def on_left(self, event):
        self.human_move('left')

    def on_right(self, event):
        self.human_move('right')

    def on_up(self, event):
        self.human_move('up')

    def on_down(self, event):
        self.human_move('down')

    def human_move(self, action):
        """For manual play with arrow keys."""
        if self.engine.is_done():
            return
        self._do_move(action)

    def ai_move(self):
        """If we have an AI model, query it for an action and make that move."""
        if not self.ai_model:
            return  # No AI provided
        if self.engine.is_done():
            return

        state = self.engine.get_state()
        action = self.ai_model.predict_move(state)  # e.g., returns 'left','right','up','down'
        self._do_move(action)

    def _do_move(self, action):
        if self.animation_in_progress:
            return  # ignore moves if still animating (optional, see advanced version)

        # Step the engine
        old_score = self.engine.get_score()
        old_grid = self.engine.get_state()
        new_grid, reward, done, info = self.engine.step(action)

        # Just re-draw everything right away (no fancy tile animations here)
        self.draw_tiles()

        # Check if game over
        if done:
            messagebox.showinfo("2048", f"Game Over!\nYour Score: {self.engine.get_score()}")

        # Optionally check if we got 2048
        # (the engine does not forcibly stop, we can keep going if we want)
        if any(2048 in row for row in new_grid):
            messagebox.showinfo("2048", "You created a 2048 tile!")

