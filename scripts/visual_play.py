# scripts/visual_play.py
import os, io, argparse
import numpy as np

if not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import imageio.v2 as imageio

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from baghchal.rl.env_baghchal import BhagChalEnv, TIGER, GOAT, N, NEIGHBORS, rc
from baghchal.rl.wrappers import TigerVsRandomGoat, TigerVsRuleGoat

T_COLOR = "tab:red"; G_COLOR = "tab:blue"
T_MS = 22; G_MS = 18; EDGE_COLOR = "0.75"

def mask_fn(env):
    return np.asarray(env.valid_action_mask(), dtype=bool)

def apply_opponent(raw_env, opponent, seed):
    opponent = (opponent or "random").lower()
    if opponent == "random":
        return TigerVsRandomGoat(raw_env, seed=seed)
    else:
        return TigerVsRuleGoat(raw_env, policy_name=opponent)

def draw_board(board, step, info, ax):
    ax.clear()
    ax.set_title(f"Step {step} | winner={info.get('winner')} | reason={info.get('reason')}")
    ax.set_xlim(-0.5, N-0.5); ax.set_ylim(-0.5, N-0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    for u in range(N*N):
        ru, cu = rc(u); x1, y1 = cu, N-1-ru
        for v in NEIGHBORS[u]:
            if v <= u: continue
            rv, cv = rc(v); x2, y2 = cv, N-1-rv
            ax.plot([x1, x2], [y1, y2], linewidth=2, color=EDGE_COLOR)
    for r in range(N):
        for c in range(N):
            v = int(board[r,c])
            if v == TIGER:
                ax.plot(c, N-1-r, marker="o", markersize=T_MS, mfc=T_COLOR, mec=T_COLOR)
            elif v == GOAT:
                ax.plot(c, N-1-r, marker="o", markersize=G_MS, mfc="white", mec=G_COLOR, mew=2)
    ax.margins(0.05); ax.set_xlabel("cols"); ax.set_ylabel("rows")

def pick_random_valid_action(base_env, rng):
    m = base_env.valid_action_mask()
    valid = np.flatnonzero(m)
    return int(rng.choice(valid)) if valid.size else 0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/bc_maskppo_5x5_random.zip")
    p.add_argument("--opponent", default="random", choices=["random","safety","mobility","edge"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--delay", type=float, default=0.8)
    p.add_argument("--gif", default="")
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()

    assert os.path.exists(args.model), f"Model not found: {args.model}"

    raw = BhagChalEnv(seed=args.seed, reward_perspective=TIGER)
    wrapped = apply_opponent(raw, args.opponent, args.seed)
    env = ActionMasker(wrapped, mask_fn)

    # base for drawing (peel wrappers to the naked env that has .board)
    base = wrapped
    while hasattr(base, "env"):
        base = base.env

    model = MaskablePPO.load(args.model, env=env, device="cuda")
    print("Loaded:", os.path.abspath(args.model))
    print("num_timesteps:", int(getattr(model, "num_timesteps", -1)))

    obs, info = env.reset()
    done, step = False, 0
    rng = np.random.default_rng(args.seed)

    fig, ax = plt.subplots(figsize=(6,6), dpi=120)
    draw_board(base.board, step, info, ax)

    live = bool(os.environ.get("DISPLAY")) and not args.gif
    images = []
    if live: plt.pause(args.delay)
    elif args.gif:
        buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        images.append(imageio.imread(buf)); buf.close()

    while not done and step < 300:
        if base.current_player == TIGER:
            m = base.valid_action_mask()
            a, _ = model.predict(obs, deterministic=args.deterministic, action_masks=m)
            a = int(a)
            if not m[a]:  # paranoia fallback
                valid = np.flatnonzero(m)
                a = int(rng.choice(valid)) if valid.size else 0
        else:
            # when using rule goats, the wrapper will auto-play the goat turn,
            # but if opponent=random we still need a random goat action
            if args.opponent == "random":
                a = pick_random_valid_action(base, rng)
            else:
                # let wrapper handle goat turns; just no-op for viz
                a = None

        if a is not None:
            obs, reward, term, trunc, info = env.step(a)
        else:
            # advance via a dummy tiger pass? No: call env.step only when we have an action.
            # When opponent is rule-based, the wrapper did goat moves during last step()
            # so we need to trigger tiger's turn with a dummy legal move request; easiest is:
            m = base.valid_action_mask()
            # choose a legal tiger no-op? There is no no-op, so just refresh obs by stepping zero-length:
            # Instead, reseat via draw; the env already advanced in the last tiger step.
            # We'll just redraw and continue; state already updated by wrapper.
            obs, reward, term, trunc, info = obs, 0.0, info.get("winner") is not None, False, info

        step += 1
        draw_board(base.board, step, info, ax)
        if live: plt.pause(args.delay)
        done = term or trunc

        if args.gif:
            buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
            images.append(imageio.imread(buf)); buf.close()

    if args.gif and images:
        imageio.mimsave(args.gif, images, duration=args.delay)
        print(f"Saved GIF to {args.gif} (≈{1.0/args.delay:.2f} FPS)")

if __name__ == "__main__":
    main()
