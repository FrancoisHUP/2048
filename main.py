import tkinter as tk
from game.gameui import Game2048GUI
from game.game_engine import GameEngine, GameRecorder
from agent.dummy.ai import DummyAI
from agent.dqn.ai import DQN2048

def evaluate_dummy():
    engine = GameEngine()
    ai = DummyAI()  

     # Evaluate the AI
    recorder = GameRecorder()
    avg_score, score_std = ai.eval(engine,100, recorder) # recorder
    print(avg_score, score_std)

def train_dqn():
    engine = GameEngine()
    ai = DQN2048() # "agent\dqn\model_chkpt\dqn_2048.pth"  
    recorder = GameRecorder()

    # Train the AI
    ai.train(engine, episodes=100000, recorder=recorder)

def eval_dqn():
    engine = GameEngine()
    ai = DQN2048("agent/dqn/model_chkpt/dqn_2048_best.pth")

    # Evaluate the AI
    recorder = GameRecorder()
    avg_score, max_score, min_score, score_std = ai.eval(engine, 100, recorder) # recorder
    print(f"avg_score={avg_score:.2f}, max_score={max_score:.2f}, min_score={min_score:.2f}, std={score_std:.2f}")

def play_game():
    root = tk.Tk()
    engine = GameEngine()

    # Inference with the Game hook
    game_gui = Game2048GUI(root, engine)
    root.mainloop()

def main():
    # train_dqn()
    # eval_dqn()
    play_game()
    # evaluate_dummy()

if __name__ == "__main__":
    main()
