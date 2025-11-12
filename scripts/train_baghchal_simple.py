# scripts/train_baghchal_simple.py
import os
import argparse
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from baghchal.rl.env_baghchal import BhagChalEnv, TIGER
from baghchal.rl.wrappers import TigerVsRandomGoat, TigerVsRuleGoat

def mask_fn(env):
    return np.asarray(env.valid_action_mask(), dtype=bool)

def wrap_goat(env, opponent: str, seed: int):
    opponent = (opponent or "random").lower()
    if opponent == "random":
        return TigerVsRandomGoat(env, seed=seed)
    elif opponent in {"safety", "mobility", "edge"}:
        return TigerVsRuleGoat(env, policy_name=opponent)
    else:
        raise ValueError(f"Unknown opponent: {opponent}")

def make_env(seed=0, opponent="random"):
    def _thunk():
        env = BhagChalEnv(seed=seed, reward_perspective=TIGER)
        env = wrap_goat(env, opponent, seed)
        env = ActionMasker(env, mask_fn)  # AFTER the wrapper
        env = Monitor(env, filename=f"logs/monitor_{opponent}_{seed}.monitor.csv",
                      info_keywords=("winner","reason"))
        return env
    return _thunk

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--opponent", default="random", choices=["random","safety","mobility","edge"])
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/tb", exist_ok=True)

    vec_env = DummyVecEnv([make_env(seed=args.seed, opponent=args.opponent)])

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device="cuda",
        tensorboard_log="logs/tb"
    )

    tb_name = f"ppo_5x5_vs_{args.opponent}"
    model.learn(total_timesteps=args.timesteps, tb_log_name=tb_name)
    out_path = f"models/bc_maskppo_5x5_{args.opponent}"
    print("SAVING THE MODEL >>>>>>>>>>>", out_path)
    model.save(out_path)
    vec_env.close()

if __name__ == "__main__":
    main()
