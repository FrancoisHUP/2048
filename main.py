import tkinter as tk
from gameui import Game2048GUI
from game_engine import GameEngine
from dummy.ai import DummyAI
from dqn.ai import DQN2048

def main():
    # root = tk.Tk()
    engine = GameEngine()
    # ai = DummyAI()  
    # ai = DQN2048("dqn\model_chkpt\dqn_2048.pth")  
    ai = DQN2048()  

    # Train the AI
    ai.train(engine,30000)

    # Evaluate the AI
    avg_score, score_std = ai.eval(engine,100)
    print(avg_score, score_std)

    # Inference with the Game hook
    # game_gui = Game2048GUI(root, engine, ai_model=ai)
    # root.mainloop()

if __name__ == "__main__":
    main()
