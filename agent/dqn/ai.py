import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import deque

# ======================
# 1. NETWORK DEFINITION
# ======================
class DQN(nn.Module):
    """
    A simpler MLP network for 2048. 
    The board is only 4x4, so we can flatten it into 16 inputs.
    """
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(16, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        # x will be of shape (batch_size, 1, 4, 4)
        # Flatten to shape (batch_size, 16)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ==========================================
# 2. UTILITY FUNCTIONS & REPLAY MEMORY
# ==========================================
def state_to_tensor(grid):
    """
    Convert the 4x4 grid to a log2 scale 
    (0 if cell is empty, else log2(cell_value)).
    Then shape = (1, 1, 4, 4) for batch usage in PyTorch.
    """
    mat = []
    for row in grid:
        row_vals = []
        for val in row:
            row_vals.append(math.log2(val) if val > 0 else 0)
        mat.append(row_vals)
    mat = np.array(mat, dtype=np.float32)   # shape (4, 4)
    mat = np.expand_dims(mat, axis=(0,1))   # shape (1, 1, 4, 4)
    return torch.from_numpy(mat)

def boltzmann_exploration(q_values, temperature=1.0):
        probabilities = torch.softmax(q_values / temperature, dim=0).cpu().numpy()
        return np.random.choice(len(q_values), p=probabilities)

class ReplayMemory:
    """
    A simple replay buffer to store (state, action, reward, next_state, done).
    """
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.memory)

# ===============================
# 3. MAIN DQN 2048 AGENT CLASS
# ===============================
class DQN2048:
    def __init__(self, model_path=None):
        """
        - self.net      -> policy network (used for action selection)
        - self.target_net -> target network (updated at intervals for stable training)
        """
        # Initialize the policy network
        self.net = DQN()

        # Load weights if given, else we can save an initial random policy
        if model_path:
            self.net.load_state_dict(torch.load(model_path))
        else:
            self.create_random_policy()
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        # Initialize target net as a copy of policy net
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()

        # Actions = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        self.action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

        print(f"Using device: {self.device}")

    def create_random_policy(self):
        """If you like, you can save the initial weights here."""
        torch.save(self.net.state_dict(), "agent/dqn/model_chkpt/dqn_2048_init.pth")

    # -----------------------------------------------
    # 3.1 Action Selection (Epsilon-Greedy Strategy)
    # -----------------------------------------------
    def select_action(self, state, env, steps_done,
                  eps_start=1.0, eps_end=0.1, eps_decay=40000,
                  temperature=1.0):
        """
        Epsilon-greedy with Boltzmann exploration.
        """
        # Compute decreasing epsilon
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)

        # Exploration
        if random.random() < eps_threshold:
            # Perform Boltzmann exploration
            with torch.no_grad():
                q_values = self.net(state).squeeze(0)  # Q-values for current state
                action_idx = boltzmann_exploration(q_values, temperature=temperature)
                action = self.action_map[action_idx]

            # Check if the chosen action is valid; fallback if invalid
            if env.is_valid_action(action):
                return action
            else:
                valid_actions = [a for a in self.action_map.values() if env.is_valid_action(a)]
                return random.choice(valid_actions) if valid_actions else random.choice(self.action_map.values())
        else:
            # Exploitation: choose the best Q-value
            with torch.no_grad():
                q_values = self.net(state).squeeze(0).cpu().numpy()
                action_idx = np.argmax(q_values)
                action = self.action_map[action_idx]

            # Check if the chosen action is valid; fallback if invalid
            if env.is_valid_action(action):
                return action
            else:
                valid_actions = [a for a in self.action_map.values() if env.is_valid_action(a)]
                return random.choice(valid_actions) if valid_actions else random.choice(self.action_map.values())
    # --------------------------
    # 3.2 Training Loop
    # --------------------------
    def train(self, env, episodes=5000, eval_interval=250, eval_episodes=50, recorder=None):
        """
        Main training loop. 
        - env: 2048 environment with reset(), step(), is_done(), etc.
        - episodes: number of training episodes
        - eval_interval: how often to run evaluation
        - eval_episodes: number of games to test performance
        """
        # Hyperparameters
        gamma = 0.99
        lr = 5e-4           # Slightly larger than 1e-4
        batch_size = 128    # somewhat smaller batch
        memory_size = 100000  
        target_update = 25  
        steps_done = 0

        # DQN components
        memory = ReplayMemory(memory_size)
        optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # For logging/evaluation
        eval_results = []
        highest_tile = 0
        target_tile = 2048
        total_spaces = 16
        best_average_score = 0

        with tqdm(range(episodes), desc="Training Progress", unit="episode") as pbar:
            for episode in pbar:
                # Ensure the model is in training mode
                self.net.train()
                
                # Initialize environment
                env.reset()
                state = state_to_tensor(env.get_state()).to(self.device)
                total_reward = 0

                while not env.is_done():
                    # Choose action
                    action = self.select_action(state, env, steps_done)
                    steps_done += 1

                    # Step environment
                    next_state_grid, score_diff, done, _ = env.step(action)

                    
                    next_state_grid_array = np.array(next_state_grid).flatten()

                    #  Highest tile and empty spaces normalize 
                    highest_tile_number = np.max(next_state_grid_array)
                    empty_spaces = sum(1 for x in next_state_grid if x == 0)
                    normalized_highest_tile = highest_tile_number / target_tile

                    # Reward shaping
                    reward = score_diff - 0.01
                    reward += normalized_highest_tile * 10  # Bonus for achieving higher tiles
                    reward += (empty_spaces / total_spaces) * 5  # Bonus for maintaining board flexibility
                        
                    if done:
                        penalty = -50 - (target_tile - highest_tile_number) * 0.05
                        reward += penalty

                    # Convert next state to tensor
                    next_state = state_to_tensor(next_state_grid).to(self.device)

                    # Store transition in replay memory
                    action_idx = list(self.action_map.values()).index(action)
                    memory.push(state, action_idx, reward, next_state, done)

                    # Move to next state
                    state = next_state
                    total_reward += reward

                    # Optimize (if memory is sufficient)
                    if len(memory) >= batch_size:
                        self.optimize_model(memory, gamma, optimizer, batch_size)

                    # Periodically update the target network
                    if steps_done % target_update == 0:
                        self.target_net.load_state_dict(self.net.state_dict())

                # Keep track of highest reward 
                if highest_tile < highest_tile_number: 
                    highest_tile = highest_tile_number

                # Update the progress bar
                pbar.set_postfix({
                    "Reward": f"{total_reward:.2f}",
                    "Highest tile" : f"{highest_tile:.0f}",
                    "Steps": steps_done,
                })

                # Periodic evaluation with recorder
                if episode % eval_interval == 0 and episode > 0:
                    avg_score, max_score, min_score, score_std = self.eval(env, eval_episodes, recorder, episode)
                    eval_results.append((episode, avg_score, score_std))
                    tqdm.write(f"[Eval at Episode {episode}] avg_score={avg_score:.2f}, max_score={max_score:.2f}, min_score={min_score:.2f}, std={score_std:.2f}")
                    
                    with open("agent/dqn/logs/training_log.csv", "a") as log_file:
                        log_file.write(f"{episode},{avg_score},{max_score},{min_score},{score_std}\n")

                    # Save the best model based on average score
                    if best_average_score < avg_score:
                        best_average_score = avg_score
                        torch.save(self.net.state_dict(), f"agent/dqn/model_chkpt/dqn_2048_best.pth")
                    torch.save(self.net.state_dict(), f"agent/dqn/model_chkpt/dqn_2048_{episode}.pth")

        # Save final model
        torch.save(self.net.state_dict(), "agent/dqn/model_chkpt/dqn_2048_final.pth")

        return eval_results

    # ---------------------------------------------
    # 3.3 Single Training Step (Optimize Model)
    # ---------------------------------------------
    def optimize_model(self, memory, gamma, optimizer, batch_size):
        """Sample from replay memory and do a single optimization step."""
        # Sample a batch
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = memory.sample(batch_size)

        # Prepare tensors
        device = self.device
        batch_state = torch.cat(batch_state).to(device)       # shape: (batch_size, 1, 4, 4)
        batch_next_state = torch.cat(batch_next_state).to(device)
        batch_action = torch.tensor(batch_action, dtype=torch.long, device=device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=device)
        batch_done = torch.tensor(batch_done, dtype=torch.bool, device=device)

        # Current Q values
        # shape of q_values: (batch_size, 4)
        q_values = self.net(batch_state)
        # gather the Q-value corresponding to each chosen action
        q_values = q_values.gather(1, batch_action.unsqueeze(1)).squeeze(1)  # shape: (batch_size,)

        # Next Q values (Double DQN approach)
        with torch.no_grad():
            # Get the best action from the policy net
            best_actions = self.net(batch_next_state).argmax(dim=1, keepdim=True)  # shape: (batch_size, 1)
            # Evaluate those actions on the target net
            target_q = self.target_net(batch_next_state).gather(1, best_actions).squeeze(1)

            # For terminal states, next Q = 0
            target_q[batch_done] = 0.0

        # Bellman backup
        target_value = batch_reward + gamma * target_q

        # Compute loss
        loss = nn.SmoothL1Loss()(q_values, target_value)

        # Gradient clipping here
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ------------------------------
    # 3.4 Evaluation Function
    # ------------------------------
    def eval(self, env, n_episodes=10, recorder=None, episode=None):
        """
        Evaluate the model in a deterministic environment using GPU if available.
        - n_episodes: number of validation games.
        - recorder: GameRecorder object for saving the best game.
        - episode: Current training episode, used for naming the saved game file.
        """
        self.net.eval()  # Set evaluation mode
        self.net.to(self.device)  # Ensure the model is on the correct device (GPU/CPU)

        best_score = -float('inf')  # Initialize best score to a very low value
        best_game = None  # Placeholder for the best game recording

        scores = []  # Store the total scores for all episodes

        # Loop through evaluation episodes
        for _ in tqdm(range(n_episodes), desc="Evaluation Progress", leave=False):
            env.reset()  # Reset the environment
            total_score = 0

            # Start recording for this game
            if recorder is not None:
                recorder.reset()
                recorder.start()

            # Play the episode until the game is done
            while not env.is_done():
                with torch.no_grad():  # No gradients needed for evaluation
                    # Identify valid actions
                    valid_actions = [
                        action for action in self.action_map.values()
                        if env.is_valid_action(action)
                    ]

                    if not valid_actions:
                        print("No valid actions left, ending game.")
                        break

                    # Predict the greedy move
                    state = state_to_tensor(env.get_state()).to(self.device)
                    q_values = self.net(state).squeeze(0).cpu().numpy()
                    action_idx = np.argmax(q_values)
                    action = self.action_map[action_idx]

                    # If the predicted action is invalid, fallback to a random valid action
                    if action not in valid_actions:
                        action = random.choice(valid_actions)

                # Take the chosen action in the environment
                old_state = env.get_state()
                new_grid, reward, done, _ = env.step(action)
                total_score += reward

                # Record the step if the recorder is active
                if recorder is not None and recorder.active:
                    recorder.record_step(
                        state=old_state,
                        action=action,
                        next_state=new_grid,
                        reward=reward,
                        done=done,
                        score=env.get_score()
                    )

            # Append the episode score
            scores.append(total_score)

            # If this game has the best score, save its recording
            if total_score > best_score:
                best_score = total_score
                best_game = list(recorder.recording) if recorder else None

            # Stop recording for this game
            if recorder is not None:
                recorder.stop()

        # Save the best game recording
        if recorder is not None and best_game is not None:
            recorder.recording = best_game
            filename = f"agent/dqn/recorded_games/evaluation_game_{episode}_{best_score}.json" if episode is not None else f"agent/dqn/recorded_games/evaluation_game_{best_score}.json"
            recorder.save_to_json(filename)

        # Calculate statistics for evaluation
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        score_std = np.std(scores)

        self.net.train()  # Switch back to training mode
        return avg_score, max_score, min_score, score_std
