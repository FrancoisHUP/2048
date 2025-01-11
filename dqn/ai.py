
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
from collections import deque

# class DQN(nn.Module):
#     def __init__(self, input_size=16, hidden_size=128, output_size=4):
#         super(DQN, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, output_size)
#         )
    
#     def forward(self, x):
#         return self.net(x)

# def state_to_tensor(grid):
#     flat = []
#     for row in grid:
#         for val in row:
#             if val > 0:
#                 flat.append(math.log2(val))
#             else:
#                 flat.append(0)
#     return torch.tensor(flat, dtype=torch.float32).unsqueeze(0)  # shape (1,16)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1),  # Input: 1x4x4
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),  # Output: 64x2x2
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * 2, 128),  # Flatten CNN output
            nn.ReLU(),
            nn.Linear(128, 4),  # 4 actions
        )

    def forward(self, x):
        x = x.view(-1, 1, 4, 4)  # Reshape input to 1x4x4
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

def state_to_tensor(grid):
    mat = []
    for row in grid:
        row_vals = []
        for val in row:
            row_vals.append(math.log2(val) if val > 0 else 0)
        mat.append(row_vals)
    mat = np.array(mat, dtype=np.float32)   # shape (4,4)
    mat = np.expand_dims(mat, axis=(0,1))   # shape (1,1,4,4)
    return torch.from_numpy(mat)

class ReplayMemory:
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
    
class DQN2048:
    def __init__(self, model_path=None):
        self.net = DQN()
        if model_path:
            self.net.load_state_dict(torch.load(model_path))
        else:
            self.create_random_policy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        print(f"Using device: {self.device}")
        self.action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

    def create_random_policy(self):
        torch.save(self.net.state_dict(), "dqn/model_chkpt/dqn_2048.pth")

    def predict_move(self, grid):
        state = state_to_tensor(grid).to(self.device)
        with torch.no_grad():
            q_values = self.net(state)
            action_idx = q_values.argmax(dim=1).item()
        return self.action_map[action_idx]

    def select_action(self, state, env, steps_done, eps_start=1.0, eps_end=0.1, eps_decay=10000):
        """
        Select an action using epsilon-greedy policy.
        - state: current game state (tensor)
        - env: GameEngine instance
        - steps_done: current step count
        """
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)
        if random.random() < eps_threshold:
            # Exploration: pick a random valid action
            valid_actions = [action for action in self.action_map.values() if env.is_valid_action(action)]
            return random.choice(valid_actions) if valid_actions else random.choice(self.action_map.values())
        else:
            # Exploitation: choose greedy action
            q_values = self.net(state).squeeze(0).detach().cpu().numpy()  # Detach tensor before converting to NumPy
            action_indices = np.argsort(q_values)[::-1]  # Sort actions by Q-value
            for action_idx in action_indices:
                action = self.action_map[action_idx]
                if env.is_valid_action(action):
                    return action
            # Fallback if no valid actions are found
            return random.choice(self.action_map.values())

    def train(self, env, episodes=10000, eval_interval=500, eval_episodes=10):
        gamma = 0.99
        lr = 1e-4
        batch_size = 256
        memory_size = 100000
        target_update = 1000

        policy_net = DQN()
        target_net = DQN()
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        memory = ReplayMemory(memory_size)

        steps_done = 0
        eval_results = []

        with tqdm(range(episodes), desc="Training Progress", unit="episode") as pbar:
            for episode in pbar:
                env.reset()
                state = state_to_tensor(env.get_state()).to(self.device)
                total_reward = 0

                while not env.is_done():
                    action = self.select_action(state, env, steps_done)
                    steps_done += 1

                    next_state_grid, score_diff, done, _ = env.step(action)
                    reward = score_diff - 0.1
                    next_state = state_to_tensor(next_state_grid).to(self.device)

                    action_idx = list(self.action_map.values()).index(action)
                    memory.push(state, action_idx, reward, next_state, done)

                    if len(memory) >= batch_size:
                        self.optimize_model(memory, policy_net, target_net, gamma, optimizer, batch_size)

                    state = next_state
                    total_reward += reward

                if steps_done % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                # Update progress bar description
                pbar.set_postfix({
                    "Episode": episode,
                    "Reward": total_reward,
                    "Steps": steps_done,
                })

                # Periodic evaluation
                if episode % eval_interval == 0 and episode > 0:
                    avg_score, score_std = self.eval(env, eval_episodes)
                    eval_results.append((episode, avg_score, score_std))
                    tqdm.write(f"Episode {episode} ({eval_episodes} episodes evaluation): Avg Score = {avg_score:.2f}, Std = {score_std:.2f}")

        # Save the trained model
        torch.save(policy_net.state_dict(), "dqn/model_chkpt/dqn_2048.pth")

        return eval_results  # Return evaluation results for further analysis

    def optimize_model(self, memory, policy_net, target_net, gamma, optimizer, batch_size):
        """Optimize the policy network."""
        # Sample a batch from memory
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = memory.sample(batch_size)

        # Move tensors to the same device as the model
        device = self.device  # Use the device set in the class
        batch_state = torch.cat(batch_state).to(device)
        batch_next_state = torch.cat(batch_next_state).to(device)
        batch_action = torch.tensor(batch_action, dtype=torch.long).to(device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32).to(device)
        batch_done = torch.tensor(batch_done, dtype=torch.bool).to(device)

        # Ensure policy_net and target_net are on the same device
        policy_net.to(device)
        target_net.to(device)

        # Compute Q(s, a)
        q_values = policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)

        # Compute target Q values using the target network
        # Double DQN approach:
        with torch.no_grad():
            next_action = policy_net(batch_next_state).argmax(dim=1)
            max_next_q_values = target_net(batch_next_state).gather(1, next_action.unsqueeze(1)).squeeze(1)
        # with torch.no_grad():
        #     max_next_q_values = target_net(batch_next_state).max(dim=1)[0]
        #     max_next_q_values[batch_done] = 0.0  # No future reward if the episode is done
        expected_q_values = batch_reward + gamma * max_next_q_values

        # Compute loss
        loss = nn.MSELoss()(q_values, expected_q_values)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def eval(self, env, n_episodes=10):
        """
        Evaluate the model in a deterministic environment using GPU if available.
        - env_class: class of your GameEngine
        - n_episodes: number of validation games
        """
        # print(f"Evaluating {n_episodes} games")
        self.net.eval()
        self.net.to(self.device)  # Ensure model is on GPU (or CPU if unavailable)

        scores = []
        for _ in tqdm(range(n_episodes), desc="Evaluation Progress", leave=False):
            total_score = 0

            while not env.is_done():
                with torch.no_grad():
                    # Get valid actions
                    valid_actions = [action for action in self.action_map.values() if env.is_valid_action(action)]

                    if not valid_actions:
                        print("No valid actions left, ending game.")
                        break  # Exit the loop if no valid actions (shouldn't happen often)

                    # Predict the greedy move and verify it's valid
                    action = self.predict_move(env.get_state())
                    if action not in valid_actions:
                        # print(f"Action '{action}' is invalid, choosing a valid one.")
                        action = random.choice(valid_actions)  # Fallback to a random valid action

                # Step the environment
                _, reward, _, _ = env.step(action)
                total_score += reward

            scores.append(total_score)
            env.reset()

        avg_score = np.mean(scores)
        score_std = np.std(scores)
        return avg_score, score_std
        