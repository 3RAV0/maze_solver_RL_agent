# maze_solver_RL_agent
Reinforcement Learning agent that learns to solve randomly generated mazes using Q-learning.  Includes a Gymnasium-compatible Maze environment, BFS baseline comparison, and optional Pygame visualization.


# RL Maze Agent 🧩🤖

This project implements a **Q-learning reinforcement learning agent** that learns to solve mazes.  
It comes with:
- A **Gymnasium-compatible environment** (`ids_maze_env.py`)
- A **Q-learning training loop** (`train_q_learning.py`) with evaluation and optional Pygame visualization
- **Maze generator & player** (`maze16_rl_demo.py`) to create and save new mazes
- **BFS baseline** (`maze_utils.py`) for shortest-path comparison
- **Experiment runner** (`run_experiment.py`) to load mazes and launch training automatically

---

## 📂 Project Structure
rl-maze-agent/
├─ ids_maze_env.py # Gym environment for the maze
├─ maze_utils.py # BFS shortest path utility
├─ train_q_learning.py # Q-learning loop + viewer
├─ run_experiment.py # Entry point: choose maze, train, log
├─ maze16_rl_demo.py # Generate & play mazes manually
├─ requirements.txt
└─ README.md


1. Generate or play a maze
Run the demo and use keys to navigate:
python maze16_rl_demo.py

Controls:
WASD / Arrow keys → Move
O → Save current maze as maze<N>.npy
L → Load latest saved maze
N → Generate a new random maze


2. Train the RL agent
python run_experiment.py
Select a saved maze*.npy or type latest for the most recent.

BFS shortest path is computed for reference.

Training starts, with periodic evaluation.

Optionally, a Pygame window opens to watch the agent explore and solve.

Example Maze

<img width="532" height="532" alt="image" src="https://github.com/user-attachments/assets/4810aa22-dba9-4a1d-afc4-e6c2caa269c1" />


