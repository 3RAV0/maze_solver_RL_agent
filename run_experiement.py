# run_experiment.py
# Kullanım: python run_experiment.py
# Not: ids_maze_env.py, maze_utils.py ve train_q_learning.py aynı klasörde olmalı.
# Gerekirse: pip install gymnasium

import os, glob, time, json
import numpy as np
from ids_maze_env import MazeEnv
from maze_utils import bfs_shortest_length
from train_q_learning import train_q_learning

def find_maze_files():
    files = []
    for f in glob.glob("maze*.npy"):
        name = os.path.splitext(os.path.basename(f))[0]
        if name.startswith("maze"):
            try:
                idx = int(name[4:])
                files.append((idx, f))
            except:
                files.append((-1, f))
    files.sort(key=lambda x: x[0])
    return files

def pick_maze():
    mazes = find_maze_files()
    if not mazes:
        raise FileNotFoundError("No maze*.npy found. Save one first with 'O' in the demo.")

    print("\nAvailable mazes:")
    for i, (idx, path) in enumerate(mazes, start=1):
        label = f"(maze{idx})" if idx >= 0 else "(other)"
        print(f"  [{i}] {path} {label}")
    print("Choose: number / filename / 'latest' / empty for latest")

    choice = input("Load which maze? ").strip()

    if choice == "" or choice.lower() in ("latest", "l"):
        selected = mazes[-1][1]
        print(f"[INFO] Using latest: {selected}")
        return selected

    if choice.isdigit():
        num = int(choice)
        if 1 <= num <= len(mazes):
            return mazes[num - 1][1]
        else:
            raise ValueError(f"Invalid index: {num}")

    if os.path.exists(choice):
        return choice

    alt = os.path.join(os.getcwd(), choice)
    if os.path.exists(alt):
        return alt

    raise FileNotFoundError(f"Not found: {choice}")

def check_walls(walls: np.ndarray):
    if walls.ndim != 3 or walls.shape[2] != 4:
        raise ValueError(f"Expected shape (H, W, 4). Got {walls.shape}")
    return walls.shape[0], walls.shape[1]

def main():
    maze_file = pick_maze()
    walls = np.load(maze_file)
    h, w = check_walls(walls)
    print(f"[INFO] Maze loaded: {maze_file}  shape={walls.shape} (H={h}, W={w})")

    ref_len = bfs_shortest_length(walls)
    if ref_len is None:
        print("[WARN] BFS could not find a path. Check the maze before training.")
        return
    print(f"[INFO] BFS shortest path = {ref_len} steps")

    env = MazeEnv(walls)

    config = dict(
        episodes=5000,
        alpha=0.2,
        gamma=0.98,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.995,
        eval_every=50,
        success_streak_target=5
    )
    print("[INFO] Training started...")
    start_time = time.perf_counter()
    Q, stats = train_q_learning(
        env,
        ref_len,
        eval_every=50,
        watch_every_episodes=500,
        watch_render=True,
        watch_every_evals=10,
        render_fps=20,
        stop_on_converge=False,
        watch_mode_before="epsilon",
        watch_mode_after="greedy",
        watch_eps=0.25
    )
    end_time = time.perf_counter()

    print("\n=== Summary ===")
    print(f"Maze: {maze_file}")
    print(f"BFS path length: {stats['bfs_len']}")
    print(f"First BFS match at episode: {stats['episodes_until_match']}")
    print(f"Converged after {config['success_streak_target']} matches: {stats['converged_at_ep']}")
    print(f"Last eval path length: {stats['last_eval_path_len']}")
    print(f"Wallclock sec: {stats['wallclock_sec']:.2f} (total {end_time - start_time:.2f})")

    os.makedirs("runs", exist_ok=True)
    log_file = os.path.join("runs", f"run_{os.path.basename(maze_file).replace('.npy','')}.json")
    output = {"maze": maze_file, "cfg": config, "stats": stats}
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[INFO] Log saved: {log_file}")

if __name__ == "__main__":
    main()
