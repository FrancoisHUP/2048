import random
import numpy as np
from tqdm import tqdm

class DummyAI:
    """Example: picks random moves from ['left','right','up','down']"""
    
    def predict_move(self, state):
        # For this dummy AI, we ignore 'state' and pick a random action
        return random.choice(['left', 'right', 'up', 'down'])
    
    def eval(self, env, n_episodes=10, recorder=None):
        """
        Evaluate the model in a deterministic environment.
        
        :param env: The environment (GameEngine).
        :param n_episodes: Number of games to run.
        :param recorder: A GameRecorder object, or None (no recording).
        """
        print(f"Evaluating {n_episodes} games")

        # 1) Randomly pick up to 3 episodes to record (if recorder is provided)
        if recorder is not None:
            num_to_record = min(3, n_episodes)
            episodes_to_record = set(random.sample(range(n_episodes), num_to_record))
        else:
            episodes_to_record = set()

        scores = []

        # 2) Loop over the number of episodes
        for episode_index in tqdm(range(n_episodes), desc="Evaluation Progress", leave=False):
            env.reset()

            # If this episode is one we plan to record
            if episode_index in episodes_to_record and recorder is not None:
                recorder.reset()
                recorder.start()

            # 3) Play until the game is done
            while not env.is_done():
                # Capture old state
                old_state = env.get_state()
                
                # Choose an action (dummy random)
                action = self.predict_move(old_state)
                
                # Environment step
                new_grid, reward, done, _ = env.step(action)
                
                # If we're recording, store this transition
                if recorder is not None and recorder.active:
                    recorder.record_step(
                        state=old_state,
                        action=action,
                        next_state=new_grid,
                        reward=reward,
                        done=done,
                        score=env.score  # or env.get_score()
                    )

            # The game is finished
            scores.append(env.score)

            # 4) If we were recording this episode, stop and save
            if episode_index in episodes_to_record and recorder is not None and recorder.active:
                recorder.stop()
                # Save each recorded episode to a unique file
                recorder.save_to_json(f"evaluation_game_{episode_index}.json")

        # 5) Compute average score and standard deviation over the episodes
        avg_score = np.mean(scores)
        score_std = np.std(scores)
        return avg_score, score_std