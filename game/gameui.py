import tkinter as tk
from tkinter import messagebox, filedialog
from game.game_engine import GameEngine, GameRecorder
import math

import tkinter as tk
import threading
import queue
import time

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from agent.dqn.ai import DQN2048

    
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

        self.load_model_button = tk.Button(
            self.button_frame, text="Load Model", font=("Verdana", 12),
            command=self.load_model
        )
        self.load_model_button.pack(side=tk.LEFT, padx=5)

        self.ai_button = tk.Button(
            self.button_frame, text="AI Move", font=("Verdana", 12),
            command=self.ai_move,
            state=tk.DISABLED  # Initially disabled
        )
        self.ai_button.pack(side=tk.LEFT, padx=5) 

        self.train_button = tk.Button(
            self.button_frame, text="Train AI", font=("Verdana", 12),
            command=self.open_training_config
        )
        self.train_button.pack(side=tk.LEFT, padx=5)

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
    #                            LOAD MODEL
    # -------------------------------------------------------------------------
    def load_model(self):
        """
        Open a file dialog to select a model, load it, and enable the AI Move button.
        """
        file_path = filedialog.askopenfilename(
            title="Load AI Model",
            filetypes=[("Model Files", "*.pth"), ("All Files", "*.*")]
        )

        if not file_path:
            messagebox.showinfo("Load Model", "No model file selected.")
            return

        try:
            # Load the model
            self.ai_model = DQN2048(model_path=file_path)
            self.ai_button.config(state=tk.NORMAL)  # Enable the AI Move button
            messagebox.showinfo("Load Model", f"Model loaded successfully from:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Load Model", f"Failed to load model:\n{e}")

    # -------------------------------------------------------------------------
    #                             AI MOVE
    # -------------------------------------------------------------------------
    def ai_move(self):
        """
        Use the loaded AI model to make a move on behalf of the user.
        Ensures the AI doesn't get stuck by predicting invalid moves.
        """
        if not self.ai_model:
            messagebox.showerror("AI Move", "No AI model loaded.")
            return

        if self.engine.is_done():
            messagebox.showinfo("Game Over", "The game is over. Start a new game.")
            return

        # Get valid actions
        valid_actions = [
            action for action in ['up', 'down', 'left', 'right']
            if self.engine.is_valid_action(action)
        ]

        if not valid_actions:
            messagebox.showinfo("No Moves", "No valid moves left. Game over.")
            return

        # Use the AI model to predict the next move
        try:
            action = self.ai_model.predict_move(self.engine.get_state(), valid_actions)
            self._do_move(action)
        except ValueError as e:
            messagebox.showerror("AI Error", f"AI could not make a valid move: {e}")

    # -------------------------------------------------------------------------
    #                            DO MOVE
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    #                             TRAIN AI
    # -------------------------------------------------------------------------
    def open_training_config(self):
        """
        Open the TrainingConfigWindow to set training parameters.
        """
        TrainingConfigWindow(self.master)

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


class TrainingWindow(tk.Toplevel):
    """
    A window that runs training in a background thread and displays
    real-time metrics (Reward, Loss, Score, etc.) using matplotlib.
    This is similar to what we've built before, but it now accepts
    all hyperparameters as arguments.
    """
    def __init__(
        self,
        parent,
        ai,
        env,
        episodes=5000,
        gamma=0.99,
        lr_start=5e-4,
        lr_end=5e-5,
        batch_size=128,
        memory_size=100000,
        target_update=25,
        eps_start=1.0,
        eps_end=0.1,
        eps_decay=40000,
        temperature=1.0,
    ):
        super().__init__(parent)
        self.title("2048 Training")
        self.geometry("800x600")
        self.resizable(False, False)

        self.ai = ai
        self.env = env

        # Store the hyperparameters
        self.episodes = episodes
        self.gamma = gamma
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_update = target_update
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.temperature = temperature

        # We’ll create a queue for receiving training status
        self.result_queue = queue.Queue()

        # -- Top Frame with Info Labels --
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.label_episode = tk.Label(top_frame, text="Episode: 0 / 0", width=18, anchor="w")
        self.label_episode.pack(side=tk.LEFT, padx=10)

        self.label_steps = tk.Label(top_frame, text="Steps: 0", width=12, anchor="w")
        self.label_steps.pack(side=tk.LEFT, padx=10)

        self.label_eps = tk.Label(top_frame, text="Epsilon: N/A", width=16, anchor="w")
        self.label_eps.pack(side=tk.LEFT, padx=10)

        self.label_sps = tk.Label(top_frame, text="Steps/sec: 0", width=16, anchor="w")
        self.label_sps.pack(side=tk.LEFT, padx=10)

        self.label_htile = tk.Label(top_frame, text="Highest Tile: 0", width=18, anchor="w")
        self.label_htile.pack(side=tk.LEFT, padx=10)

        self.best_score = 0  # Initialize best score
        self.label_best_score = tk.Label(top_frame, text="Best Score: 0", width=18, anchor="w")
        self.label_best_score.pack(side=tk.LEFT, padx=10)

        # -- Matplotlib Figure with 3 Subplots: Reward, Loss, Score --
        self.fig = Figure(figsize=(7, 4), dpi=100)
        # We use a 3x1 grid for three subplots
        self.ax_reward = self.fig.add_subplot(311)
        self.ax_score = self.fig.add_subplot(312)
        self.ax_loss = self.fig.add_subplot(313)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Data for plots
        self.episodes_list = []
        self.rewards_list = []
        self.losses_list = []
        self.scores_list = [] 

        # Start the training in a background thread
        self.train_thread = threading.Thread(
            target=self.run_training_thread,
            daemon=True
        )
        self.train_thread.start()

        # Start polling the queue
        self.after(100, self.poll_queue)

    def run_training_thread(self):
        """
        This method runs in the background thread.
        We'll call a method from your AI (like "train_in_thread")
        that uses the hyperparams we set. We'll pass self.result_queue for updates.
        """
        # You might create a new method in DQN2048 that accepts all these parameters:
        self.ai.train_in_thread(
            env=self.env,
            episodes=self.episodes,
            gamma=self.gamma,
            lr_start=self.lr_start,
            lr_end=self.lr_end,
            batch_size=self.batch_size,
            memory_size=self.memory_size,
            target_update=self.target_update,
            eps_start=self.eps_start,
            eps_end=self.eps_end,
            eps_decay=self.eps_decay,
            temperature=self.temperature,
            result_queue=self.result_queue,
        )
    def poll_queue(self):
        """
        Check the result_queue for new training info.
        Update the UI. Schedule itself to run again after 100ms.
        """
        while True:
            try:
                msg = self.result_queue.get_nowait()
            except queue.Empty:
                break

            if msg is None:
                # Training is done
                tk.messagebox.showinfo("Training Complete", "All episodes finished!")
                return
            else:
                # Unpack data from the training thread
                episode = msg["episode"]
                episodes_total = msg["episodes_total"]
                steps_done = msg["steps_done"]
                steps_per_sec = msg["steps_per_sec"]
                highest_tile_overall = msg["highest_tile_overall"]
                epsilon = msg["epsilon"]
                reward = msg["reward"]
                loss = msg["loss"]
                score = msg["score"]  # Current game score

                # Update the best score if the current score is higher
                if score > self.best_score:
                    self.best_score = score
                    self.label_best_score.config(text=f"Best Score: {self.best_score}")

                # Update labels
                self.label_episode.config(text=f"Episode: {episode}/{episodes_total}")
                self.label_steps.config(text=f"Steps: {steps_done}")
                self.label_eps.config(text=f"Epsilon: {epsilon:.4f}")
                self.label_sps.config(text=f"Steps/sec: {steps_per_sec:.2f}")
                self.label_htile.config(text=f"Highest Tile: {highest_tile_overall}")

                # Update data lists
                self.episodes_list.append(episode)
                self.rewards_list.append(reward)
                self.losses_list.append(loss)
                self.scores_list.append(score)

                # Redraw all plots
                self.ax_reward.clear()
                self.ax_reward.set_title("Reward per Episode")
                self.ax_reward.set_xlabel("Episode")
                self.ax_reward.set_ylabel("Reward")
                self.ax_reward.plot(self.episodes_list, self.rewards_list, color="blue")

                self.ax_score.clear()
                self.ax_score.set_title("Score per Episode")
                self.ax_score.set_xlabel("Episode")
                self.ax_score.set_ylabel("Score")
                self.ax_score.plot(self.episodes_list, self.scores_list, color="green")

                self.ax_loss.clear()
                self.ax_loss.set_title("Loss per Episode")
                self.ax_loss.set_xlabel("Episode")
                self.ax_loss.set_ylabel("Loss")
                self.ax_loss.plot(self.episodes_list, self.losses_list, color="red")                

                self.fig.tight_layout()
                self.canvas.draw()

        # Continue polling
        self.after(100, self.poll_queue)

class TrainingConfigWindow(tk.Toplevel):
    """
    A window that lets the user set all the training parameters before starting training.
    Once the user clicks 'Start Training', we open the TrainingWindow with the specified hyperparams.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Training Configuration")
        self.geometry("400x500")
        self.transient(parent)  # Make it a modal dialog
        self.grab_set()  # Ensure it captures all events while open
        self.resizable(False, False)

        # We'll store entries in a dict for convenience
        self.entries = {}

        row = 0

        # MODEL PATH (LOAD EXISTING)
        tk.Label(self, text="Model Path").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        self.model_path_var = tk.StringVar()
        tk.Entry(self, textvariable=self.model_path_var, width=30).grid(row=row, column=1, padx=5, pady=5)
        tk.Button(self, text="Browse", command=self.browse_model).grid(row=row, column=2, padx=5, pady=5)
        row += 1

        # EPISODES
        row = self._add_label_entry("Episodes", default_val="5000", row=row)
        # GAMMA
        row = self._add_label_entry("Gamma", default_val="0.99", row=row)
        # LR_START
        row = self._add_label_entry("LR Start", default_val="0.0005", row=row)
        # LR_END
        row = self._add_label_entry("LR End", default_val="0.0005", row=row)
        # BATCH_SIZE
        row = self._add_label_entry("Batch Size", default_val="128", row=row)
        # MEMORY_SIZE
        row = self._add_label_entry("Memory Size", default_val="100000", row=row)
        # TARGET_UPDATE
        row = self._add_label_entry("Target Update (steps)", default_val="25", row=row)
        # EPS_START
        row = self._add_label_entry("Eps Start", default_val="1.0", row=row)
        # EPS_END
        row = self._add_label_entry("Eps End", default_val="0.1", row=row)
        # EPS_DECAY
        row = self._add_label_entry("Eps Decay", default_val="40000", row=row)
        # TEMPERATURE
        row = self._add_label_entry("Temperature", default_val="1.0", row=row)

        # START TRAINING BUTTON
        tk.Button(self, text="Start Training", command=self.start_training).grid(row=row, column=0, columnspan=3, pady=20)


    def _add_label_entry(self, label_text, default_val, row):
        """
        Utility to add a label + entry with default value.
        We'll store the StringVar in self.entries[label_text].
        """
        tk.Label(self, text=label_text).grid(row=row, column=0, sticky="e", padx=5, pady=5)
        var = tk.StringVar(value=default_val)
        entry = tk.Entry(self, textvariable=var, width=15)
        entry.grid(row=row, column=1, padx=5, pady=5)
        self.entries[label_text] = var
        return row + 1

    def browse_model(self):
        """
        Let user pick a model file. Store path in self.model_path_var.
        """
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Files", "*.pth"), ("All Files", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)

    def start_training(self):
        """
        Read all user entries, parse them, create a new AI (with model if provided),
        then open the TrainingWindow with the user-specified hyperparameters.
        """
        try:
            model_path = self.model_path_var.get().strip() or None

            episodes = int(self.entries["Episodes"].get().strip())
            gamma = float(self.entries["Gamma"].get().strip())
            lr_start = float(self.entries["LR Start"].get().strip())
            lr_end = float(self.entries["LR End"].get().strip())
            batch_size = int(self.entries["Batch Size"].get().strip())
            memory_size = int(self.entries["Memory Size"].get().strip())
            target_update = int(self.entries["Target Update (steps)"].get().strip())
            eps_start = float(self.entries["Eps Start"].get().strip())
            eps_end = float(self.entries["Eps End"].get().strip())
            eps_decay = float(self.entries["Eps Decay"].get().strip())
            temperature = float(self.entries["Temperature"].get().strip())
        except ValueError as e:
            messagebox.showerror("Invalid input", f"Please enter valid numeric values.\n\nError: {e}")
            return

        # Create environment
        env = GameEngine()

        # Create AI
        if model_path:
            ai = DQN2048(model_path=model_path)
        else:
            ai = DQN2048()

        # Create the TrainingWindow
        train_window = TrainingWindow(
            parent=self,
            ai=ai,
            env=env,
            episodes=episodes,
            gamma=gamma,
            lr_start=lr_start,
            lr_end=lr_end,
            batch_size=batch_size,
            memory_size=memory_size,
            target_update=target_update,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay=eps_decay,
            temperature=temperature,
        )
        train_window.grab_set()  # Make the training window modal if needed

