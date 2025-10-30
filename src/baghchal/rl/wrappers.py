# src/baghchal/rl/wrappers.py
import numpy as np
import gymnasium as gym
from baghchal.rl.env_baghchal import GOAT

class TigerVsRandomGoat(gym.Wrapper):
    def __init__(self, env, seed=None):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)

    # ✅ Add this so ActionMasker/mask_fn can see the mask
    def valid_action_mask(self):
        return self.env.valid_action_mask()

    def _random_goat_action(self):
        mask = self.env.valid_action_mask()
        valid = np.flatnonzero(mask)
        return int(self.rng.choice(valid)) if valid.size else 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        terminated = truncated = False
        while getattr(self.env, "current_player", None) == GOAT and not (terminated or truncated):
            a = self._random_goat_action()
            obs, _, terminated, truncated, info = self.env.step(a)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        while not (terminated or truncated) and self.env.current_player == GOAT:
            a = self._random_goat_action()
            obs, r2, terminated, truncated, info = self.env.step(a)
            reward += float(r2)
        return obs, reward, terminated, truncated, info
