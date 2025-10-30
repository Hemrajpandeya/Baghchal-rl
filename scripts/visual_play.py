# scripts/visual_play.py
import os, io, argparse
import numpy as np

# Use a headless backend on cluster BEFORE importing pyplot
if not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import imageio.v2 as imageio

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from baghchal.rl.env_baghchal import (
    BhagChalEnv, TIGER, GOAT, N, NEIGHBORS, rc
)

# Colors & sizes
T_COLOR = "tab:red"     # Tigers (filled)
G_COLOR = "tab:blue"    # Goats (hollow)
T_MS = 22               # marker size Tigers
G_MS = 18               # marker size Goats
EDGE_COLOR = "0.75"     # board lines (grey)

def mask_fn(env):
    return np.asarray(env.valid_action_mask(), dtype=bool)

def draw_board(board, step, info, ax):
    """Draw board using env's actual connectivity (NEIGHBORS)."""
    ax.clear()
    ax.set_title(f"Step {step} | winner={info.get('winner')} | reason={info.get('reason')}")
    ax.set_xlim(-0.5, N-0.5)
    ax.set_ylim(-0.5, N-0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(N)); ax.set_yticks(range(N))

    # edges once per pair u<v
    for u in range(N*N):
        ru, cu = rc(u); x1, y1 = cu, N-1-ru
        for v in NEIGHBORS[u]:
            if v <= u:
                continue
            rv, cv = rc(v); x2, y2 = cv, N-1-rv
            ax.plot([x1, x2], [y1, y2], linewidth=2, color=EDGE_COLOR)

    # pieces
    for r in range(N):
        for c in range(N):
            v = int(board[r, c])
            if v == TIGER:
                ax.plot(c, N-1-r, marker="o", markersize=T_MS, mfc=T_COLOR, mec=T_COLOR)
            elif v == GOAT:
                ax.plot(c, N-1-r, marker="o", markersize=G_MS, mfc="white", mec=G_COLOR, mew=2)

    ax.margins(0.05)
    ax.set_xlabel("cols"); ax.set_ylabel("rows")

def pick_random_valid_action(base_env, rng):
    m = base_env.valid_action_mask()
    valid = np.flatnonzero(m)
    return int(rng.choice(valid)) if valid.size else 0

def main():
    p = argparse.ArgumentParser()
    # default to the model trained vs random goat
    p.add_argument("--model", default="models/bc_maskppo_5x5_vs_rand.zip")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--delay", type=float, default=0.8, help="seconds per move")
    p.add_argument("--gif", default="", help="save GIF to this path (e.g., playback.gif)")
    p.add_argument("--deterministic", action="store_true", help="use deterministic policy for Tiger")
    args = p.parse_args()

    assert os.path.exists(args.model), f"Model not found: {args.model}"

    # Env: Tiger = model (masked), Goat = random via our own picker
    env = BhagChalEnv(seed=args.seed, reward_perspective=TIGER)
    base = env
    env = ActionMasker(env, mask_fn)  # training-like masking wrapper
    model = MaskablePPO.load(args.model, env=env, device="cuda")
    print("Loaded:", os.path.abspath(args.model))
    print("num_timesteps:", int(getattr(model, "num_timesteps", -1)))

    obs, info = env.reset()
    done, step = False, 0
    rng = np.random.default_rng(args.seed)

    # Fixed-size figure reused every frame (no jitter)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    if hasattr(fig.canvas, "manager") and hasattr(fig.canvas.manager, "set_window_title"):
        fig.canvas.manager.set_window_title("BhagChal: Tiger (model) vs Goat (random)")

    live = bool(os.environ.get("DISPLAY")) and not args.gif
    images = []

    # first frame
    draw_board(base.board, step, info, ax)
    if live:
        plt.pause(args.delay)
    elif args.gif:
        buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        images.append(imageio.imread(buf)); buf.close()

    while not done and step < 300:
        if base.current_player == TIGER:
            # *** KEY: pass action masks to predict so Tiger always picks a legal action
            m = base.valid_action_mask()
            a, _ = model.predict(
                obs,
                deterministic=args.deterministic,
                action_masks=m
            )
            a = int(a)
            # paranoia: if still illegal, fallback to a random valid
            if not m[a]:
                valid = np.flatnonzero(m)
                a = int(rng.choice(valid)) if valid.size else 0
            action = a
        else:
            action = pick_random_valid_action(base, rng)

        obs, reward, terminated, truncated, info = env.step(int(action))
        step += 1

        draw_board(base.board, step, info, ax)
        if live:
            plt.pause(args.delay)
        done = terminated or truncated

        if args.gif:
            buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
            images.append(imageio.imread(buf)); buf.close()

    print(f"Game over at step {step}. winner={info.get('winner')} reason={info.get('reason')}")
    if args.gif and images:
        fps = 1.0 / args.delay  # convert delay to frames per second
        imageio.mimsave(args.gif, images, duration=args.delay)  # keep duration per frame
        print(f"Saved GIF to {args.gif} (≈{fps:.2f} FPS)")

    if live:
        print("Close the window to exit.")
        plt.show()

if __name__ == "__main__":
    main()
