import math
import random
import numpy as np
import torch
import time
import queue
import threading
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import deque


# =============================================================================
#                               1. DQN NETWORK
# =============================================================================
# class DQN(nn.Module):
#     """
#     A simpler MLP network for 2048.
#     The board is 4x4, so we flatten it into 16 inputs.
#     Output: Q-values for 4 possible actions (up, down, left, right).
#     """

#     def __init__(self):
#         super().__init__()

#         self.fc = nn.Sequential(
#             nn.Linear(16, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),

#             nn.Linear(512, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),

#             nn.Linear(512, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),

#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),

#             nn.Linear(256, 4)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x shape: (batch_size, 1, 4, 4)
#         Flatten to: (batch_size, 16)
#         Returns Q-values for each action: shape (batch_size, 4).
#         """
#         x = x.view(x.size(0), -1)
#         return self.fc(x)

class DQN(nn.Module):
    """
    A large Dueling DQN network for 2048.
    1) We flatten the 4x4 grid (16 inputs).
    2) Pass through big hidden layers (e.g., 1024 neurons each).
    3) Split into two heads:
       - Value stream (outputs 1 value for the state)
       - Advantage stream (outputs 4 values for each action)
    4) Q-values = Value + (Advantage - mean(Advantage))
    """

    def __init__(self):
        super().__init__()

        # Common feature extractor
        # 16 -> 1024 -> 1024 -> 1024 -> feature
        self.feature = nn.Sequential(
            nn.Linear(16, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
        )

        # Value stream: produces a single scalar value for the state
        self.value_stream = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Advantage stream: produces an advantage for each action (4)
        self.advantage_stream = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        """
        x shape: (batch_size, 1, 4, 4)
        Flatten to: (batch_size, 16)
        Returns Q-values for each action: shape (batch_size, 4).
        """
        # Flatten
        x = x.view(x.size(0), -1)

        # Extract features
        features = self.feature(x)  # shape: (batch_size, 1024)

        # Compute value and advantage
        value = self.value_stream(features)          # shape: (batch_size, 1)
        advantage = self.advantage_stream(features)  # shape: (batch_size, 4)

        # Combine into Q-values
        # Q(s,a) = V(s) + [ A(s,a) - mean(A(s,a) over actions ) ]
        mean_advantage = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - mean_advantage)

        return q_values




# =============================================================================
#                     2. UTILITY FUNCTIONS & REPLAY MEMORY
# =============================================================================
def state_to_tensor(grid):
    """
    Convert the 4x4 grid to log2 scale (0 if cell=0, else log2(cell_value)).
    Output shape: (1, 1, 4, 4) as a PyTorch tensor.
    """
    mat = []
    for row in grid:
        row_vals = [math.log2(val) if val > 0 else 0 for val in row]
        mat.append(row_vals)
    mat = np.array(mat, dtype=np.float32)  # shape (4, 4)
    mat = np.expand_dims(mat, axis=(0, 1)) # shape (1, 1, 4, 4)
    return torch.from_numpy(mat)


def boltzmann_exploration(q_values, temperature=1.0):
    """
    Boltzmann (softmax) exploration: sample an action index
    with probability ~ exp(Q / temperature).
    """
    probabilities = torch.softmax(q_values / temperature, dim=0).cpu().numpy()
    return np.random.choice(len(q_values), p=probabilities)


class ReplayMemory:
    """
    Simple replay buffer to store (state, action, reward, next_state, done).
    """

    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.memory)


# =============================================================================
#                     3. MAIN DQN 2048 AGENT CLASS
# =============================================================================
class DQN2048:
    """
    A Deep Q-Network (DQN) based agent for the 2048 game.
    Includes:
      - a policy network (self.net)
      - a target network (self.target_net)
      - training loops (train, train_in_thread)
      - evaluation and inference methods
    """

    def __init__(self, model_path=None):
        """
        - model_path: path to a saved model (if any). If None, create a random policy.
        - self.net: the main policy network
        - self.target_net: the fixed target network
        """
        # Initialize policy network
        self.net = DQN()

        # Load weights if specified; otherwise create a random policy checkpoint
        if model_path:
            self.net.load_state_dict(torch.load(model_path))
        else:
            self.create_random_policy()

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        # Initialize target network
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()

        # Action map: 0->'up', 1->'down', 2->'left', 3->'right'
        self.action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

        print(f"Using device: {self.device}")

    # -------------------------------------------------------------------------
    #                          CREATION / INFERENCE
    # -------------------------------------------------------------------------
    def create_random_policy(self):
        """
        Optionally save initial random weights to a file.
        """
        torch.save(self.net.state_dict(), "agent/dqn/model_chkpt/dqn_2048_init.pth")

    def predict_move(self, state, valid_actions):
        """
        Predict the next move for a given state, ensuring it is valid.
        If no valid action is found, picks a random valid action.
        :param state: current 4x4 game state (list of lists)
        :param valid_actions: list of valid actions (e.g., ['up', 'down'])
        :return: chosen action (e.g., 'up')
        """
        state_tensor = state_to_tensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.net(state_tensor).squeeze(0).cpu().numpy()

        # Sort all possible actions by Q-value (descending)
        sorted_actions = sorted(
            self.action_map.items(),
            key=lambda x: q_values[x[0]],
            reverse=True
        )

        # Pick the first valid action from the sorted list
        for action_idx, action_name in sorted_actions:
            if action_name in valid_actions:
                return action_name

        # Fallback: random valid action if none matched
        if valid_actions:
            return random.choice(valid_actions)
        raise ValueError("No valid actions available")

    # -------------------------------------------------------------------------
    #                       ACTION SELECTION (EPS-GREEDY)
    # -------------------------------------------------------------------------
    def select_action(
        self,
        state,
        env,
        steps_done,
        eps_start=1.0,
        eps_end=0.1,
        eps_decay=100000,
        temperature=1.0
    ):
        """
        Epsilon-greedy with optional Boltzmann exploration.
        :param state: Torch tensor of shape (1,1,4,4)
        :param env: GameEngine environment
        :param steps_done: how many steps have been taken
        :param eps_start, eps_end, eps_decay: epsilon scheduling
        :param temperature: for Boltzmann exploration
        :return: chosen action (str: 'up', 'down', 'left', or 'right')
        """
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)

        # Exploration
        if random.random() < eps_threshold:
            # Boltzmann exploration
            with torch.no_grad():
                q_values = self.net(state).squeeze(0)  # shape: (4,)
                action_idx = boltzmann_exploration(q_values, temperature=temperature)
                chosen_action = self.action_map[action_idx]
            # Validate action
            if env.is_valid_action(chosen_action):
                return chosen_action
            valid_actions = [a for a in self.action_map.values() if env.is_valid_action(a)]
            return random.choice(valid_actions) if valid_actions else random.choice(self.action_map.values())

        # Exploitation
        else:
            with torch.no_grad():
                q_values = self.net(state).squeeze(0).cpu().numpy()
                action_idx = np.argmax(q_values)
                chosen_action = self.action_map[action_idx]
            # Validate action
            if env.is_valid_action(chosen_action):
                return chosen_action
            valid_actions = [a for a in self.action_map.values() if env.is_valid_action(a)]
            return random.choice(valid_actions) if valid_actions else random.choice(self.action_map.values())

    # -------------------------------------------------------------------------
    #                       BACKGROUND TRAINING THREAD
    # -------------------------------------------------------------------------
    def train_in_thread(
        self,
        env,
        episodes,
        gamma,
        lr_start,
        lr_end,
        batch_size,
        memory_size,
        target_update,
        eps_start,
        eps_end,
        eps_decay,
        temperature,
        result_queue
    ):
        """
        Trains the DQN in a background thread, sending progress updates to `result_queue`.
        Now includes periodic and best-score checkpoint saving.
        """
        from agent.dqn.ai import ReplayMemory

        steps_done = 0
        highest_tile_overall = 0
        best_score = -float('inf')   # Track best game score

        # Set up replay memory and optimizer
        memory = ReplayMemory(memory_size)
        current_lr = lr_start
        optimizer = optim.Adam(self.net.parameters(), lr=current_lr)

        for episode in range(1, episodes + 1):
            start_time = time.time()

            self.net.train()
            env.reset()
            state = state_to_tensor(env.get_state()).to(self.device)

            total_reward = 0.0
            steps_this_episode = 0
            done = False
            episode_loss = 0.0
            loss_count = 0
            highest_tile_ep = 0

            while not done:
                # Epsilon schedule
                epsilon = eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)

                action = self.select_action(
                    state,
                    env,
                    steps_done,
                    eps_start=eps_start,
                    eps_end=eps_end,
                    eps_decay=eps_decay,
                    temperature=temperature
                )
                steps_done += 1
                steps_this_episode += 1

                next_state_grid, score_diff, done, _ = env.step(action)

                # Track highest tile
                flat_grid = [val for row in next_state_grid for val in row]
                highest_tile_local = max(flat_grid)
                if highest_tile_local > highest_tile_ep:
                    highest_tile_ep = highest_tile_local

                # Basic reward shaping
                reward = score_diff

                next_state = state_to_tensor(next_state_grid).to(self.device)
                action_idx = list(self.action_map.values()).index(action)
                memory.push(state, action_idx, reward, next_state, done)

                state = next_state
                total_reward += reward

                # Optimize if enough samples
                if len(memory) >= batch_size:
                    loss_val = self.optimize_model(memory, gamma, optimizer, batch_size)
                    episode_loss += loss_val
                    loss_count += 1

                # Update target net periodically
                if steps_done % target_update == 0:
                    self.target_net.load_state_dict(self.net.state_dict())

            # Update overall highest tile
            if highest_tile_ep > highest_tile_overall:
                highest_tile_overall = highest_tile_ep

            # Steps per second
            end_time = time.time()
            elapsed = max(end_time - start_time, 1e-6)
            steps_per_sec = steps_this_episode / elapsed

            avg_loss = episode_loss / (loss_count if loss_count > 0 else 1)
            episode_score = env.get_score()

            # 1) Check if we have a new best score
            if episode_score > best_score:
                best_score = episode_score
                # Save a "best" checkpoint
                torch.save(self.net.state_dict(), "agent/dqn/model_chkpt/dqn_2048_best.pth")
                print(f"[Checkpoint] New best score={best_score:.0f} at Episode {episode}, model saved!")

            # 2) Periodically save a checkpoint (e.g. every 500 episodes)
            if episode % 500 == 0:
                ckpt_path = f"agent/dqn/model_chkpt/dqn_2048_{episode}.pth"
                torch.save(self.net.state_dict(), ckpt_path)
                print(f"[Checkpoint] Episode {episode}, model saved to {ckpt_path}")

            # Send progress to the main thread via queue
            msg = {
                "episode": episode,
                "episodes_total": episodes,
                "steps_done": steps_done,
                "steps_per_sec": steps_per_sec,
                "highest_tile_overall": highest_tile_overall,
                "epsilon": epsilon,
                "reward": total_reward,
                "loss": avg_loss,
                "score": episode_score,
            }
            result_queue.put(msg)

        # At the end of all training
        torch.save(self.net.state_dict(), "agent/dqn/model_chkpt/dqn_2048_final.pth")
        print("[Checkpoint] Final model saved!")
        result_queue.put(None)

    # -------------------------------------------------------------------------
    #                    TRADITIONAL CLI TRAINING LOOP
    # -------------------------------------------------------------------------
    def train(
        self,
        env,
        episodes=5000,
        eval_interval=250,
        eval_episodes=50,
        recorder=None
    ):
        """
        Main training loop using a synchronous approach + tqdm progress bar.
        :param env: 2048 environment
        :param episodes: number of training episodes
        :param eval_interval: how often to evaluate
        :param eval_episodes: number of evaluation episodes
        :param recorder: optional GameRecorder
        :return: list of evaluation results
        """
        gamma = 0.99
        lr = 5e-4
        batch_size = 128
        memory_size = 100000
        target_update = 25
        steps_done = 0

        memory = ReplayMemory(memory_size)
        optimizer = optim.Adam(self.net.parameters(), lr=lr)

        eval_results = []
        highest_tile = 0
        target_tile = 2048
        total_spaces = 16
        best_average_score = 0

        with tqdm(range(episodes), desc="Training Progress", unit="episode") as pbar:
            for episode in pbar:
                self.net.train()
                env.reset()
                state = state_to_tensor(env.get_state()).to(self.device)
                total_reward = 0

                while not env.is_done():
                    action = self.select_action(state, env, steps_done)
                    steps_done += 1

                    next_state_grid, score_diff, done, _ = env.step(action)

                    next_state_grid_array = np.array(next_state_grid).flatten()
                    highest_tile_number = np.max(next_state_grid_array)
                    empty_spaces = sum(1 for x in next_state_grid if x == 0)
                    normalized_highest_tile = highest_tile_number / target_tile

                    # Reward shaping
                    reward = score_diff - 0.01
                    reward += normalized_highest_tile * 10
                    reward += (empty_spaces / total_spaces) * 5

                    if done:
                        penalty = -50 - (target_tile - highest_tile_number) * 0.05
                        reward += penalty

                    next_state = state_to_tensor(next_state_grid).to(self.device)
                    action_idx = list(self.action_map.values()).index(action)
                    memory.push(state, action_idx, reward, next_state, done)

                    state = next_state
                    total_reward += reward

                    if len(memory) >= batch_size:
                        self.optimize_model(memory, gamma, optimizer, batch_size)

                    if steps_done % target_update == 0:
                        self.target_net.load_state_dict(self.net.state_dict())

                if highest_tile < highest_tile_number:
                    highest_tile = highest_tile_number

                # Update progress bar
                pbar.set_postfix({
                    "Reward": f"{total_reward:.2f}",
                    "Highest tile": f"{highest_tile:.0f}",
                    "Steps": steps_done,
                })

                # Periodic evaluation
                if episode % eval_interval == 0 and episode > 0:
                    avg_score, max_score, min_score, score_std = self.eval(env, eval_episodes, recorder, episode)
                    eval_results.append((episode, avg_score, score_std))
                    tqdm.write(
                        f"[Eval at Episode {episode}] "
                        f"avg_score={avg_score:.2f}, max_score={max_score:.2f}, "
                        f"min_score={min_score:.2f}, std={score_std:.2f}"
                    )
                    with open("agent/dqn/logs/training_log.csv", "a") as log_file:
                        log_file.write(f"{episode},{avg_score},{max_score},{min_score},{score_std}\n")

                    # Save best model by average score
                    if best_average_score < avg_score:
                        best_average_score = avg_score
                        torch.save(self.net.state_dict(), "agent/dqn/model_chkpt/dqn_2048_best.pth")
                    torch.save(self.net.state_dict(), f"agent/dqn/model_chkpt/dqn_2048_{episode}.pth")

        torch.save(self.net.state_dict(), "agent/dqn/model_chkpt/dqn_2048_final.pth")
        return eval_results

    # -------------------------------------------------------------------------
    #                       SINGLE TRAINING STEP (OPTIMIZE)
    # -------------------------------------------------------------------------
    def optimize_model(self, memory, gamma, optimizer, batch_size):
        """
        Sample from replay memory and run a single optimization step.
        Uses a Double DQN approach for the target update.
        Returns loss.item() for logging.
        """
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = memory.sample(batch_size)

        device = self.device
        batch_state = torch.cat(batch_state).to(device)
        batch_next_state = torch.cat(batch_next_state).to(device)
        batch_action = torch.tensor(batch_action, dtype=torch.long, device=device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=device)
        batch_done = torch.tensor(batch_done, dtype=torch.bool, device=device)

        # Current Q values
        q_values = self.net(batch_state)  # shape: (batch_size, 4)
        q_values = q_values.gather(1, batch_action.unsqueeze(1)).squeeze(1)  # shape: (batch_size,)

        # Next Q values (Double DQN)
        with torch.no_grad():
            best_actions = self.net(batch_next_state).argmax(dim=1, keepdim=True)
            target_q = self.target_net(batch_next_state).gather(1, best_actions).squeeze(1)
            target_q[batch_done] = 0.0

        target_value = batch_reward + gamma * target_q

        # SmoothL1 (Huber) loss
        loss = nn.SmoothL1Loss()(q_values, target_value)

        # Gradient clipping & optimization
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    # -------------------------------------------------------------------------
    #                               EVALUATION
    # -------------------------------------------------------------------------
    def eval(self, env, n_episodes=10, recorder=None, episode=None):
        """
        Evaluate the model over n_episodes with no exploration.
        If recorder is provided, record the best game.
        """
        self.net.eval()
        self.net.to(self.device)

        best_score = float('-inf')
        best_game = None
        scores = []

        for _ in tqdm(range(n_episodes), desc="Evaluation Progress", leave=False):
            env.reset()
            total_score = 0

            if recorder is not None:
                recorder.reset()
                recorder.start()

            while not env.is_done():
                valid_actions = [
                    a for a in self.action_map.values()
                    if env.is_valid_action(a)
                ]
                if not valid_actions:
                    print("No valid actions left, ending game.")
                    break

                with torch.no_grad():
                    state = state_to_tensor(env.get_state()).to(self.device)
                    q_vals = self.net(state).squeeze(0).cpu().numpy()
                    action_idx = np.argmax(q_vals)
                    action = self.action_map[action_idx]

                    # Fallback if invalid
                    if action not in valid_actions:
                        action = random.choice(valid_actions)

                old_state = env.get_state()
                new_grid, reward, done, _ = env.step(action)
                total_score += reward

                if recorder is not None and recorder.active:
                    recorder.record_step(
                        state=old_state,
                        action=action,
                        next_state=new_grid,
                        reward=reward,
                        done=done,
                        score=env.get_score()
                    )

            scores.append(total_score)
            if total_score > best_score:
                best_score = total_score
                best_game = list(recorder.recording) if recorder else None

            if recorder is not None:
                recorder.stop()

        # Optionally save the best game
        if recorder is not None and best_game is not None:
            recorder.recording = best_game
            filename = (
                f"agent/dqn/recorded_games/evaluation_game_{episode}_{best_score}.json"
                if episode is not None
                else f"agent/dqn/recorded_games/evaluation_game_{best_score}.json"
            )
            recorder.save_to_json(filename)

        avg_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        score_std = np.std(scores)

        self.net.train()  # back to training mode
        return avg_score, max_score, min_score, score_std
