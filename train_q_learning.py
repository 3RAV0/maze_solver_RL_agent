import time
import numpy as np

def run_episode_text(env, policy_fn, max_steps=500, verbose=True):
    state, _ = env.reset()
    steps = 0
    if verbose:
        print("=== BOT RUN START ===")
    for _ in range(max_steps):
        action = policy_fn(state)
        state, reward, done, truncated, _ = env.step(action)
        steps += 1
        w = getattr(env, "W", None) or getattr(env, "width", None) or env.walls.shape[1]
        if verbose:
            y, x = divmod(state, w)
            print(f"Step {steps}: pos=({y},{x}) action={action} reward={reward}")
        if done:
            if verbose:
                print(f"Reached goal in {steps} steps ✅")
            return steps
        if truncated:
            if verbose:
                print("Truncated (max steps reached)")
            return None
    return None


class MazeViewer:
    def __init__(self, env, cell_size=None, margin=10):
        try:
            import pygame
        except Exception as e:
            raise RuntimeError("pygame not installed. Install with: pip install pygame") from e
        self.env = env
        self.walls = env.walls
        self.height, self.width = self.walls.shape[:2]
        self.margin = margin
        self.cell = 32 if cell_size is None and max(self.height, self.width) <= 20 else (cell_size or 24)
        import pygame
        pygame.init()
        self.pg = pygame
        self.screen = pygame.display.set_mode(
            (self.width * self.cell + 2 * self.margin, self.height * self.cell + 2 * self.margin)
        )
        pygame.display.set_caption("RL Maze Viewer")

    def _draw(self, pos, goal):
        pg = self.pg
        self.screen.fill((255, 255, 255))
        for y in range(self.height):
            for x in range(self.width):
                cx = self.margin + x * self.cell
                cy = self.margin + y * self.cell
                if self.walls[y, x, 0]: pg.draw.line(self.screen, (0, 0, 0), (cx, cy), (cx + self.cell, cy), 2)
                if self.walls[y, x, 1]: pg.draw.line(self.screen, (0, 0, 0), (cx, cy + self.cell), (cx + self.cell, cy + self.cell), 2)
                if self.walls[y, x, 2]: pg.draw.line(self.screen, (0, 0, 0), (cx, cy), (cx, cy + self.cell), 2)
                if self.walls[y, x, 3]: pg.draw.line(self.screen, (0, 0, 0), (cx + self.cell, cy), (cx + self.cell, cy + self.cell), 2)
        gy = self.margin + goal[0] * self.cell + self.cell // 2
        gx = self.margin + goal[1] * self.cell + self.cell // 2
        pg.draw.circle(self.screen, (0, 200, 0), (gx, gy), self.cell // 3)
        py = self.margin + pos[0] * self.cell + self.cell // 2
        px = self.margin + pos[1] * self.cell + self.cell // 2
        pg.draw.circle(self.screen, (0, 120, 255), (px, py), self.cell // 3)
        pg.display.flip()

    def watch(self, policy_fn=None, fps=15, max_steps=2000, mode="greedy", eps=0.2):
        import numpy as _np
        pg = self.pg
        clock = pg.time.Clock()
        state, _ = self.env.reset()
        steps = 0
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT: return None
                if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE: return None
            if mode == "epsilon":
                if _np.random.rand() < eps:
                    action = _np.random.randint(0, 4)
                else:
                    action = policy_fn(state) if policy_fn else 0
            else:
                action = policy_fn(state) if policy_fn else 0
            state, reward, done, truncated, _ = self.env.step(action)
            steps += 1
            y, x = divmod(state, self.width)
            goal = getattr(self.env, "goal", None) or getattr(self.env, "goal_cell", None) or (self.height - 1, self.width - 1)
            self._draw((y, x), goal)
            if done:
                pg.display.set_caption(f"Reached goal in {steps} steps ✅ (ESC to close)")
                return steps
            if truncated or steps >= max_steps:
                pg.display.set_caption("Truncated (ESC to close)")
                return None
            clock.tick(fps)


def evaluate(env, policy_fn):
    state, _ = env.reset()
    steps = 0
    for _ in range(env.max_steps):
        action = policy_fn(state)
        state, reward, done, truncated, _ = env.step(action)
        steps += 1
        if done: return steps
        if truncated: return None
    return None


def train_q_learning(
    env,
    bfs_len,
    episodes=5000,
    alpha=0.2,
    gamma=0.98,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.995,
    eval_every=50,
    success_streak_target=5,
    watch_every_evals=None,
    watch_every_episodes=None,
    watch_verbose=False,
    watch_render=True,
    render_fps=15,
    stop_on_converge=True,
    watch_mode_before="epsilon",
    watch_mode_after="greedy",
    watch_eps=0.25,
):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def _greedy_choice(q_row):
        max_val = q_row.max()
        candidates = np.flatnonzero(q_row == max_val)
        return int(np.random.choice(candidates))

    def greedy_policy(s): return _greedy_choice(Q[s])

    eps = eps_start
    t0 = time.perf_counter()
    first_success, streak = None, 0
    viewer = None

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        while not done:
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = greedy_policy(state)
            next_state, reward, done_flag, truncated, _ = env.step(action)
            done = done_flag or truncated
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state

        eps = max(eps_end, eps * eps_decay)

        if ep % eval_every == 0:
            length = evaluate(env, greedy_policy)
            print(f"[Eval] ep={ep}, greedy length={length}")
            if length is not None and bfs_len is not None and length <= bfs_len:
                streak += 1
                if streak == 1 and first_success is None:
                    first_success = ep
            else:
                streak = 0
            if streak >= success_streak_target and stop_on_converge:
                wallclock = time.perf_counter() - t0
                return Q, {
                    "episodes_until_match": first_success,
                    "wallclock_sec": wallclock,
                    "bfs_len": bfs_len,
                    "last_eval_path_len": length,
                    "converged_at_ep": ep,
                }

        watch_trigger = False
        if watch_every_episodes is not None and ep % watch_every_episodes == 0:
            watch_trigger = True
        elif watch_every_episodes is None and ep % eval_every == 0 and (ep // eval_every) % watch_every_evals == 0:
            watch_trigger = True

        if watch_trigger:
            mode = watch_mode_after if streak > 0 else watch_mode_before
            if watch_render:
                if viewer is None:
                    try:
                        viewer = MazeViewer(env)
                    except RuntimeError as e:
                        print(f"[Viewer] {e}. Falling back to text mode.")
                        watch_render = False
                if viewer:
                    steps = viewer.watch(policy_fn=greedy_policy, fps=render_fps, mode=mode, eps=watch_eps)
                    print(f"[Watch][pygame][{mode}] ep={ep}, steps={steps}")
            else:
                steps = run_episode_text(env, greedy_policy, verbose=watch_verbose)
                print(f"[Watch][text] ep={ep}, steps={steps}")

    wallclock = time.perf_counter() - t0
    return Q, {
        "episodes_until_match": first_success,
        "wallclock_sec": wallclock,
        "bfs_len": bfs_len,
        "last_eval_path_len": evaluate(env, lambda s: _greedy_choice(Q[s])),
        "converged_at_ep": None,
    }
