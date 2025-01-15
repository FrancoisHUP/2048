import tkinter as tk
from tkinter import messagebox, filedialog
from game.game_engine import GameEngine, GameRecorder
import math
    
GRID_SIZE = 4
CELL_SIZE = 100
CELL_PADDING = 10
BACKGROUND_COLOR = "#92877d"
EMPTY_CELL_COLOR = "#9e948a"
# The original dictionary for base tiles up to 2048
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

# A small list of extra colors we can cycle through for tiles above 2048.
# You can add as many as you want or define a color gradient.
EXTRA_COLORS = [
    "#edc22e",  # the same as 2048 for 4096
    "#bdc22e",
    "#9dc22e",
    "#7dc22e",
    "#5dc22e",
    "#3dc22e",
    "#1dc22e",
    "#1dc25e",
    "#1dc27e",
    "#1dc29e",
    "#1dc2be",
]
FONT = ("Verdana", 24, "bold")

# Dictionary to map the textual action to an arrow symbol
ACTION_SYMBOLS = {
    "left": "←",
    "right": "→",
    "up": "↑",
    "down": "↓"
}

def get_tile_color(value):
    """
    Return a color for any 2^N tile.
    - If it's <= 2048, use the base color from CELL_COLORS.
    - If it's > 2048, generate or pick an extended color.
    """
    if value in CELL_COLORS:
        return CELL_COLORS[value]
    else:
        # For anything above 2048, pick a color from EXTRA_COLORS
        # based on the tile's exponent
        # e.g., 4096 -> exponent = 12, 8192 -> 13, etc.
        exponent = int(math.log2(value))
        # We'll cycle through EXTRA_COLORS so we don't run out
        # shift by 11 because 2^11 = 2048
        index = (exponent - 11) % len(EXTRA_COLORS)
        return EXTRA_COLORS[index]
    
class Game2048GUI:
    def __init__(self, master, engine, ai_model=None):
        self.master = master
        self.engine = engine
        self.ai_model = ai_model
        self.animation_in_progress = False

        self.best_score = 0
        self.is_recording = False
        self.recorder = GameRecorder()

        self.loaded_data = []
        self.current_step = -1
        self.is_replaying = False
        self.is_playing = False

        master.title("2048 Game")
        master.resizable(False, False)

        # ------------------- TOP FRAMES LAYOUT -------------------
        top_frame = tk.Frame(master)
        top_frame.pack(pady=5)

        # 1) score_frame -> Score + Best
        self.score_frame = tk.Frame(top_frame)
        self.score_frame.pack(side=tk.LEFT, padx=10)

        # 2) button_frame -> New Game, Load Game, Record, AI Move
        self.button_frame = tk.Frame(top_frame)
        self.button_frame.pack(side=tk.LEFT, padx=10)

        # 3) replay_frame -> Back, Next, Play/Pause/Replay, Speed slider, Action label
        self.replay_frame = tk.Frame(top_frame)
        self.replay_frame.pack_forget()

        # ------------------- SCORE FRAME -------------------
        self.score_label = tk.Label(self.score_frame, text="Score: 0", font=("Verdana", 16), width=10, anchor="w")
        self.score_label.pack(anchor='w')

        self.best_score_label = tk.Label(self.score_frame, text=f"Best: {self.best_score}", font=("Verdana", 16), width=10, anchor="w")
        self.best_score_label.pack(anchor='w')

        # ------------------- BUTTON FRAME -------------------
        self.new_game_button = tk.Button(
            self.button_frame, text="New Game", font=("Verdana", 12),
            command=self.new_game
        )
        self.new_game_button.pack(side=tk.LEFT, padx=5)

        self.load_button = tk.Button(
            self.button_frame, text="Load Game", font=("Verdana", 12),
            command=self.load_game
        )
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.record_button = tk.Button(
            self.button_frame, text="Record", font=("Verdana", 12),
            command=self.toggle_recording
        )
        self.record_button.pack(side=tk.LEFT, padx=5)

        self.ai_button = tk.Button(
            self.button_frame, text="AI Move", font=("Verdana", 12),
            command=self.ai_move
        )
        self.ai_button.pack(side=tk.LEFT, padx=5)

        # ------------------- REPLAY FRAME -------------------
        self.back_button = tk.Button(
            self.replay_frame, text="Back", font=("Verdana", 12),
            command=self.back_step
        )
        self.back_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(
            self.replay_frame, text="Next", font=("Verdana", 12),
            command=self.next_step
        )
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.play_button = tk.Button(
            self.replay_frame, text="Play", font=("Verdana", 12),
            command=self.toggle_play_pause
        )
        self.play_button.pack(side=tk.LEFT, padx=5)

        self.play_speed_scale = tk.Scale(
            self.replay_frame,
            from_=1000,
            to=10,
            resolution=10,
            orient='horizontal',
            label='Speed (ms)'
        )
        self.play_speed_scale.set(500)
        self.play_speed_scale.pack(side=tk.LEFT, padx=5)

        # Label to display the current action as an arrow
        self.action_label = tk.Label(self.replay_frame, text="Action: N/A", font=("Verdana", 12), width=10, anchor="w")
        self.action_label.pack(side=tk.LEFT, padx=10)

        # ------------------- CANVAS -------------------
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

    # -------------------------------------------------------------------------
    #                             DRAWING
    # -------------------------------------------------------------------------
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
        if current_score > self.best_score:
            self.best_score = current_score
        self.best_score_label.config(text=f"Best: {self.best_score}")

        self.master.update_idletasks()

    def draw_single_tile(self, i, j, value):
        x0 = CELL_PADDING + j * (CELL_SIZE + CELL_PADDING)
        y0 = CELL_PADDING + i * (CELL_SIZE + CELL_PADDING)
        x1 = x0 + CELL_SIZE
        y1 = y0 + CELL_SIZE

        color = get_tile_color(value) 
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
        x_center = x0 + CELL_SIZE / 2
        y_center = y0 + CELL_SIZE / 2
        self.canvas.create_text(x_center, y_center, text=str(value), font=FONT, fill="#776e65")

    # -------------------------------------------------------------------------
    #                         RECORD / STOP
    # -------------------------------------------------------------------------
    def toggle_recording(self):
        if not self.is_recording:
            self.recorder.start()
            self.is_recording = True
            self.record_button.config(text="Stop")
            print("Recording started.")
        else:
            self.recorder.stop()
            self.is_recording = False
            print("Recording stopped.")

            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json")],
                initialfile="my_2048_game.json",
                title="Save 2048 Recording"
            )
            if file_path:
                self.recorder.save_to_json(file_path)
                print(f"Recording saved to: {file_path}")

            self.record_button.config(text="Record")

    # -------------------------------------------------------------------------
    #                   LOAD A PREVIOUS GAME (REPLAY MODE)
    # -------------------------------------------------------------------------
    def load_game(self):
        if not self.engine.is_done() and not self.is_replaying:
            answer = messagebox.askyesno(
                "Abandon game?",
                "A game is currently in progress. Are you sure you want to load another game and lose this progress?"
            )
            if not answer:
                return

        file_path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Open 2048 Recording"
        )
        if not file_path:
            return

        data = self.recorder.load_from_json(file_path)
        if not data:
            messagebox.showerror("Error", "No valid data found in file.")
            return

        self.loaded_data = data
        self.current_step = -1
        self.is_replaying = True
        self.is_playing = False

        # Set the board to the first step
        first_step = self.loaded_data[0]
        self.engine.grid = [row[:] for row in first_step["state"]]
        self.engine.score = first_step["score"]
        self.engine.done = False
        self.draw_tiles()

        # Hide AI and Record buttons
        self.ai_button.pack_forget()
        self.record_button.pack_forget()

        # Show replay controls
        self.replay_frame.pack(side=tk.LEFT, padx=10)

        # Reset the Play button
        self.play_button.config(text="Play")
        # Reset action label
        self.action_label.config(text="Action: N/A")

    # -------------------------------------------------------------------------
    #                  REPLAY CONTROLS: BACK, NEXT, PLAY/PAUSE, REPLAY
    # -------------------------------------------------------------------------
    def back_step(self):
        """Go one step back in self.loaded_data, if possible."""
        if not self.is_replaying or len(self.loaded_data) == 0:
            return

        if self.current_step <= 0:
            messagebox.showinfo("Replay", "Already at the earliest step.")
            return

        self.current_step -= 1
        step_data = self.loaded_data[self.current_step]
        # Move board to next_state
        next_grid = step_data["next_state"]
        self.engine.grid = [row[:] for row in next_grid]
        self.engine.score = step_data["score"]
        self.engine.done = step_data["done"]
        self.draw_tiles()

        # Convert the textual action to an arrow symbol
        action_text = step_data["action"]
        arrow = ACTION_SYMBOLS.get(action_text, action_text)
        self.action_label.config(text=f"Action: {arrow}")

    def next_step(self):
        """Go one step forward in self.loaded_data, showing 'next_state'."""
        if not self.is_replaying or len(self.loaded_data) == 0:
            return

        if self.current_step >= len(self.loaded_data) - 1:
            messagebox.showinfo("Replay finished", "Reached the end of the recorded game.")
            self.play_button.config(text="Replay")
            self.is_playing = False
            return

        self.current_step += 1
        step_data = self.loaded_data[self.current_step]
        next_grid = step_data["next_state"]
        self.engine.grid = [row[:] for row in next_grid]
        self.engine.score = step_data["score"]
        self.engine.done = step_data["done"]
        self.draw_tiles()

        # Convert textual action to arrow
        action_text = step_data["action"]
        arrow = ACTION_SYMBOLS.get(action_text, action_text)
        self.action_label.config(text=f"Action: {arrow}")

        if self.current_step >= len(self.loaded_data) - 1:
            messagebox.showinfo("Replay finished", "Reached the end of the recorded game.")
            self.play_button.config(text="Replay")
            self.is_playing = False

    def toggle_play_pause(self):
        """Button can say "Play", "Pause", or "Replay"."""
        if not self.is_replaying:
            return

        current_text = self.play_button.cget("text")

        if current_text == "Play":
            self.is_playing = True
            self.play_button.config(text="Pause")
            self.auto_play()

        elif current_text == "Pause":
            self.is_playing = False
            self.play_button.config(text="Play")

        elif current_text == "Replay":
            self.current_step = -1
            self.is_playing = False

            if self.loaded_data:
                first_step = self.loaded_data[0]
                self.engine.grid = [row[:] for row in first_step["state"]]
                self.engine.score = first_step["score"]
                self.engine.done = False
                self.draw_tiles()
            self.play_button.config(text="Play")
            self.action_label.config(text="Action: N/A")

    def auto_play(self):
        """Automatically move forward every X ms until the end or paused."""
        if not self.is_replaying or not self.is_playing:
            return

        self.next_step()  # step forward

        if self.current_step < len(self.loaded_data) - 1 and self.is_playing:
            delay = self.play_speed_scale.get()
            self.master.after(delay, self.auto_play)
        else:
            # Reached end or paused
            self.is_playing = False

    # -------------------------------------------------------------------------
    #                             NEW GAME
    # -------------------------------------------------------------------------
    def new_game(self):
        if self.is_replaying:
            answer = messagebox.askyesno(
                "Quit Replay?",
                "You are currently replaying a recorded game. Quit replay and start a new game?"
            )
            if not answer:
                return

            self.is_replaying = False
            self.is_playing = False
            self.loaded_data = []
            self.current_step = -1

            # Hide replay controls
            self.replay_frame.pack_forget()

            # Show AI and Record again
            self.record_button.pack(side=tk.LEFT, padx=5)
            self.ai_button.pack(side=tk.LEFT, padx=5)

        self.engine.reset()
        self.draw_tiles()

    # -------------------------------------------------------------------------
    #                        KEYBOARD EVENTS
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
        if self.is_replaying:
            messagebox.showinfo("Replay mode", "Currently in replay mode. Use 'Back', 'Next', or 'Play/Pause/Replay'.")
            return
        if self.engine.is_done():
            return
        self._do_move(action)

    def ai_move(self):
        if self.is_replaying:
            messagebox.showinfo("Replay mode", "Currently in replay mode. Cannot use AI here.")
            return
        if not self.ai_model:
            return
        if self.engine.is_done():
            return

        action = self.ai_model.predict_move(self.engine.get_state())
        self._do_move(action)

    def _do_move(self, action):
        if self.animation_in_progress:
            return

        old_state = self.engine.get_state()
        new_grid, reward, done, info = self.engine.step(action)
        new_score = self.engine.get_score()

        self.draw_tiles()

        # If we're recording, store this step
        if self.recorder.active:
            self.recorder.record_step(
                old_state,
                action,
                new_grid,
                reward,
                done,
                new_score
            )

        if done:
            if new_score > self.best_score:
                self.best_score = new_score
                self.best_score_label.config(text=f"Best: {self.best_score}")
            messagebox.showinfo("2048", f"Game Over!\nYour Score: {new_score}")
