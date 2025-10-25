import argparse
import os
import sys

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

from baghchal.rl.env_baghchal import BhagChalEnv, mask_fn


def make_env(seed: int, rank: int):
    def _thunk():
        env = BhagChalEnv(seed=seed + rank)
        env = ActionMasker(env, mask_fn)  # <- import from module, picklable on Windows
        return env
    return _thunk


def build_env(n_envs: int, seed: int, force_dummy: bool = False):
    if n_envs == 1 or force_dummy:
        return DummyVecEnv([make_env(seed, 0)])
    return SubprocVecEnv([make_env(seed, i) for i in range(n_envs)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--logdir", type=str, default="runs/baghchal_ppo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dummy", action="store_true", help="Force DummyVecEnv (single process) on Windows")
    args = parser.parse_args()

    set_random_seed(args.seed)

    # If spawn/pickling still causes trouble, pass --dummy to force DummyVecEnv
    vec_env = build_env(args.num_envs, args.seed, force_dummy=args.dummy)

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=4096,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log=args.logdir,
        verbose=1,
    )
    model.learn(total_timesteps=args.total_timesteps)
    os.makedirs("models", exist_ok=True)
    model.save("models/baghchal_ppo_latest")


if __name__ == "__main__":
    main()
