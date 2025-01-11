import random
import numpy as np
from tqdm import tqdm 

class DummyAI:
    """Example: picks random moves from ['left','right','up','down']"""
    def predict_move(self, state):
        return random.choice(['left', 'right', 'up', 'down'])
    
    def eval(self, env, n_episodes=10):
        print(f"Evaluating {n_episodes} games")
        scores = []
        for _ in tqdm(range(n_episodes), desc="Evaluation Progress", leave=False):
            while not env.is_done():  
                # env.print_grid()
                move = self.predict_move(env.get_state())
                # print(move)
                env.step(move)
                
            scores.append(env.score)
            env.reset()
        
        avg_score = np.mean(scores)
        score_std = np.std(scores)
        return avg_score, score_std