#!/usr/bin/env python3
"""
maze16_rl_demo.py
- Random 16x16 maze (DFS backtracker)
- Arrow/WASD to move a "ball" from START to GOAL
- Press O to save current maze to 'maze16_fixed.npy' (freeze for RL training)
- Press L to load 'maze16_fixed.npy'
- Press N to generate a new random maze
- Optional: --seed <int> to generate a reproducible maze on start
- Optional: --player ./ball.png  --goal ./reward.png  (32x32 recommended, with alpha)
"""

import argparse
import numpy as np
import random
import pygame
import glob
import os

HEIGHT, WIDTH = 24, 24
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}

def create_maze(seed=None):
    if seed is not None:
        random.seed(seed)
    maze = np.ones((HEIGHT, WIDTH, 4), dtype=np.uint8)
    visited = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    def valid_neighbors(y, x):
        for idx, (dy, dx) in enumerate(DIRECTIONS):
            ny, nx = y + dy, x + dx
            if 0 <= ny < HEIGHT and 0 <= nx < WIDTH:
                yield idx, ny, nx

    stack = [(0, 0)]
    visited[0, 0] = 1
    while stack:
        y, x = stack[-1]
        unvisited = [(i, ny, nx) for i, ny, nx in valid_neighbors(y, x) if not visited[ny, nx]]
        if not unvisited:
            stack.pop()
            continue
        i, ny, nx = random.choice(unvisited)
        maze[y, x, i] = 0
        maze[ny, nx, OPPOSITE[i]] = 0
        visited[ny, nx] = 1
        stack.append((ny, nx))
    return maze

def draw(maze, player, target, screen, cell=32, pad=10, player_img=None, goal_img=None):
    screen.fill((255, 255, 255))
    for y in range(HEIGHT):
        for x in range(WIDTH):
            cx = pad + x * cell
            cy = pad + y * cell
            if maze[y, x, 0]: pygame.draw.line(screen, (0, 0, 0), (cx, cy), (cx + cell, cy), 2)
            if maze[y, x, 1]: pygame.draw.line(screen, (0, 0, 0), (cx, cy + cell), (cx + cell, cy + cell), 2)
            if maze[y, x, 2]: pygame.draw.line(screen, (0, 0, 0), (cx, cy), (cx, cy + cell), 2)
            if maze[y, x, 3]: pygame.draw.line(screen, (0, 0, 0), (cx + cell, cy), (cx + cell, cy + cell), 2)
    gx = pad + target[1] * cell + cell // 2
    gy = pad + target[0] * cell + cell // 2
    if goal_img is not None:
        rect = goal_img.get_rect(center=(gx, gy))
        screen.blit(goal_img, rect)
    else:
        pygame.draw.circle(screen, (0, 200, 0), (gx, gy), cell // 3)
    px = pad + player[1] * cell + cell // 2
    py = pad + player[0] * cell + cell // 2
    if player_img is not None:
        rect = player_img.get_rect(center=(px, py))
        screen.blit(player_img, rect)
    else:
        pygame.draw.circle(screen, (0, 120, 255), (px, py), cell // 3)

def move(maze, position, action):
    y, x = position
    if maze[y, x, action] == 0:
        dy, dx = DIRECTIONS[action]
        return (y + dy, x + dx)
    return (y, x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--player", type=str, default=None)
    parser.add_argument("--goal", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()

    if args.load:
        maze = np.load(args.load)
    else:
        maze = create_maze(seed=args.seed)

    pygame.init()
    cell, pad = 32, 10
    screen = pygame.display.set_mode((WIDTH * cell + 2 * pad, HEIGHT * cell + 2 * pad))
    pygame.display.set_caption("Maze (WASD/Arrows, N-new, O-save, L-load)")

    player_img = pygame.image.load(args.player).convert_alpha() if args.player else None
    goal_img = pygame.image.load(args.goal).convert_alpha() if args.goal else None
    if player_img: player_img = pygame.transform.smoothscale(player_img, (cell, cell))
    if goal_img: goal_img = pygame.transform.smoothscale(goal_img, (cell, cell))

    start, goal = (0, 0), (HEIGHT - 1, WIDTH - 1)
    pos, moves = list(start), 0

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_w): pos = list(move(maze, tuple(pos), 0)); moves += 1
                if event.key in (pygame.K_DOWN, pygame.K_s): pos = list(move(maze, tuple(pos), 1)); moves += 1
                if event.key in (pygame.K_LEFT, pygame.K_a): pos = list(move(maze, tuple(pos), 2)); moves += 1
                if event.key in (pygame.K_RIGHT, pygame.K_d): pos = list(move(maze, tuple(pos), 3)); moves += 1
                if event.key == pygame.K_n:
                    maze, pos, moves = create_maze(), list(start), 0
                if event.key == pygame.K_o:
                    files = glob.glob("maze*.npy")
                    numbers = []
                    for f in files:
                        name = os.path.splitext(os.path.basename(f))[0]
                        if name.startswith("maze"):
                            try: numbers.append(int(name[4:]))
                            except: pass
                    next_id = max(numbers) + 1 if numbers else 1
                    filename = f"maze{next_id}.npy"
                    np.save(filename, maze)
                    print(f"Saved {filename}")
                if event.key == pygame.K_l:
                    try:
                        files = glob.glob("maze*.npy")
                        if files:
                            numbered = []
                            for f in files:
                                name = os.path.splitext(os.path.basename(f))[0]
                                if name.startswith("maze"):
                                    try: numbered.append((int(name[4:]), f))
                                    except: pass
                            if numbered:
                                latest = sorted(numbered)[-1][1]
                                maze, pos, moves = np.load(latest), list(start), 0
                                print(f"Loaded {latest}")
                            else:
                                print("No valid maze*.npy found")
                        else:
                            print("No saved mazes found")
                    except Exception as e:
                        print("Load failed:", e)
        draw(maze, tuple(pos), goal, screen, cell=cell, pad=pad, player_img=player_img, goal_img=goal_img)
        if tuple(pos) == goal:
            pygame.display.set_caption(f"Solved in {moves} moves! Press N for new maze, L to load, O to save.")
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
