# scripts/train_baghchal_simple.py
import os
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from baghchal.rl.env_baghchal import BhagChalEnv, TIGER
from baghchal.rl.wrappers import TigerVsRandomGoat

def mask_fn(env):
    return np.asarray(env.valid_action_mask(), dtype=bool)

def make_env(seed=0):
    def _thunk():
        env = BhagChalEnv(seed=seed, reward_perspective=TIGER)
        env = TigerVsRandomGoat(env, seed=seed)   # Goat = random
        env = ActionMasker(env, mask_fn)          # AFTER the wrapper
        env = Monitor(env, filename=f"logs/monitor_{seed}.monitor.csv",
                    info_keywords=("winner","reason"))
        return env
    return _thunk

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/tb", exist_ok=True)

    vec_env = DummyVecEnv([make_env(seed=0)])
    print("Wrapper chain check:")
    e = vec_env.envs[0]
    while hasattr(e, "env"):
        print(" ->", type(e).__name__)
        e = e.env
    print(" ->", type(e).__name__)

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device="cuda",
        tensorboard_log="logs/tb"
    )

    model.learn(total_timesteps=100_000, tb_log_name="ppo_5x5_vs_rand")
    print("SAVING THE MODEL >>>>>>>>>>>")
    model.save("models/bc_maskppo_5x5_vs_rand")
    vec_env.close()

    # Quick test: use the SAME wrapper so goat stays random
    test_env = BhagChalEnv(seed=123, reward_perspective=TIGER)
    from baghchal.rl.wrappers import TigerVsRandomGoat
    test_env = TigerVsRandomGoat(test_env, seed=123)
    test_env = ActionMasker(test_env, mask_fn)
    test_env = Monitor(test_env)

    obs, info = test_env.reset()
    done, steps, ep_ret = False, 0, 0.0
    while not done and steps < 300:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(int(action))
        ep_ret += float(reward)
        done = terminated or truncated
        steps += 1
    print(f"[Test] steps={steps}, return={ep_ret:.3f}")

if __name__ == "__main__":
    main()
