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

# --- append below your TigerVsRandomGoat ---

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from baghchal.rl.env_baghchal import (
    GOAT, TIGER, N, rc, iof, NEIGHBORS,
    CAPTURE_TRIPLES, DIRECTED_EDGES,
    PLACE_OFFSET, STEP_OFFSET, CAP_OFFSET
)

# ------------ helpers ------------
def threatened_goats(env) -> List[Tuple[int,int,int]]:
    """All (u, mid, w) where TIGER at u can capture GOAT at mid by landing at empty w."""
    out = []
    B = env.board
    for (u, mid, w) in CAPTURE_TRIPLES:
        ru, cu = rc(u); rm, cm = rc(mid); rw, cw = rc(w)
        if B[ru, cu] == TIGER and B[rm, cm] == GOAT and B[rw, cw] == 0:
            out.append((u, mid, w))
    return out

def tiger_mobility(env) -> int:
    """Tiger legal steps+captures count (quick heuristic)."""
    save = env.current_player
    env.current_player = TIGER
    mask = env.valid_action_mask()
    env.current_player = save
    # ignore placement range for tiger
    return int(mask[STEP_OFFSET:CAP_OFFSET].sum() + mask[CAP_OFFSET:].sum())

def is_goat_place(a: int) -> bool: return a < STEP_OFFSET
def is_goat_step(a: int) -> bool:  return STEP_OFFSET <= a < CAP_OFFSET
def step_edge(a: int):             return DIRECTED_EDGES[a - STEP_OFFSET]

# ------------ goat policies ------------
class GoatPolicy:
    def pick(self, env, rng: np.random.Generator) -> int:
        raise NotImplementedError

class SafetyFirstGoat(GoatPolicy):
    def pick(self, env, rng):
        mask = env.valid_action_mask()
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return 0

        th = threatened_goats(env)

        # 1) Block immediate capture by placing on landing 'w'
        if th and env.goats_to_place > 0:
            for (_, _, w) in th:
                a = PLACE_OFFSET + w
                if a in valid:
                    return int(a)

        # 2) Safe placements (avoid becoming the 'mid' in a capture)
        def unsafe_place(i: int) -> bool:
            for (u, mid, w) in CAPTURE_TRIPLES:
                if mid == i:
                    ru, cu = rc(u); rw, cw = rc(w)
                    if env.board[ru, cu] == TIGER and env.board[rw, cw] == 0:
                        return True
            return False

        if env.goats_to_place > 0:
            safe_places = [a for a in valid if is_goat_place(a) and not unsafe_place(a - PLACE_OFFSET)]
            if safe_places:
                return int(rng.choice(safe_places))

        # 3) Movement phase: avoid stepping onto a 'mid' square
        def unsafe_arrival(v: int) -> bool:
            for (u, mid, w) in CAPTURE_TRIPLES:
                if mid == v:
                    ru, cu = rc(u); rw, cw = rc(w)
                    if env.board[ru, cu] == TIGER and env.board[rw, cw] == 0:
                        return True
            return False

        safe_steps = []
        for a in valid:
            if is_goat_step(a):
                _, v = step_edge(a)
                if not unsafe_arrival(v):
                    safe_steps.append(a)
        if safe_steps:
            return int(rng.choice(safe_steps))

        return int(rng.choice(valid))

class MobilityBlockerGoat(GoatPolicy):
    """Safety-first + greedily minimize tiger mobility."""
    def pick(self, env, rng):
        mask = env.valid_action_mask()
        valid = np.flatnonzero(mask)
        if valid.size == 0: return 0

        th = threatened_goats(env)

        # Urgent: block capture by placing at landing w
        if th and env.goats_to_place > 0:
            for (_, _, w) in th:
                a = PLACE_OFFSET + w
                if a in valid:
                    return int(a)

        def unsafe_place(i: int) -> bool:
            for (u, mid, w) in CAPTURE_TRIPLES:
                if mid == i:
                    ru, cu = rc(u); rw, cw = rc(w)
                    if env.board[ru, cu] == TIGER and env.board[rw, cw] == 0:
                        return True
            return False

        def unsafe_arrival(v: int) -> bool:
            for (u, mid, w) in CAPTURE_TRIPLES:
                if mid == v:
                    ru, cu = rc(u); rw, cw = rc(w)
                    if env.board[ru, cu] == TIGER and env.board[rw, cw] == 0:
                        return True
            return False

        # build safe candidates
        cand = []
        for a in valid:
            if is_goat_place(a):
                i = a - PLACE_OFFSET
                if not unsafe_place(i):
                    cand.append(a)
            elif is_goat_step(a):
                _, v = step_edge(a)
                if not unsafe_arrival(v):
                    cand.append(a)
        if not cand:
            cand = valid.tolist()

        # score by -tiger_mobility (simulate quickly)
        B = env.board
        best = (float("-inf"), cand[0])
        save_turn = env.current_player
        save_goats = env.goats_to_place

        for a in cand:
            env.current_player = GOAT
            score = float("-inf")
            if is_goat_place(a):
                i = a - PLACE_OFFSET
                r, c = rc(i)
                if B[r, c] != 0:
                    env.current_player, env.goats_to_place = save_turn, save_goats
                    continue
                B[r, c] = GOAT; env.goats_to_place -= 1
                score = -float(tiger_mobility(env))
                B[r, c] = 0; env.goats_to_place = save_goats
            else:
                u, v = step_edge(a)
                ru, cu = rc(u); rv, cv = rc(v)
                if B[ru, cu] != GOAT or B[rv, cv] != 0:
                    env.current_player = save_turn
                    continue
                B[rv, cv] = GOAT; B[ru, cu] = 0
                score = -float(tiger_mobility(env))
                B[ru, cu] = GOAT; B[rv, cv] = 0
            env.current_player = save_turn
            if score > best[0]:
                best = (score, a)

        return int(best[1])

class EdgeInwardGoat(MobilityBlockerGoat):
    """Placement prefers edge-middles; otherwise use mobility heuristic."""
    edge_middles = {iof(0,2), iof(2,0), iof(2,4), iof(4,2)}
    def pick(self, env, rng):
        mask = env.valid_action_mask()
        valid = np.flatnonzero(mask)
        if env.goats_to_place > 0:
            for i in sorted(self.edge_middles):
                a = PLACE_OFFSET + i
                if a in valid:
                    # also avoid immediate mid-of-capture
                    bad = False
                    for (u, mid, w) in CAPTURE_TRIPLES:
                        if mid == i:
                            ru, cu = rc(u); rw, cw = rc(w)
                            if env.board[ru, cu] == TIGER and env.board[rw, cw] == 0:
                                bad = True; break
                    if not bad:
                        return int(a)
        return super().pick(env, rng)

# ---------- wrapper that uses a rule goat ----------
@dataclass
class TigerVsRuleGoat(gym.Wrapper):
    env: gym.Env
    policy_name: str = "mobility"  # "safety" | "mobility" | "edge"

    def __post_init__(self):
        super().__init__(self.env)
        self.rng = np.random.default_rng()
        if self.policy_name == "safety":
            self.policy = SafetyFirstGoat()
        elif self.policy_name == "edge":
            self.policy = EdgeInwardGoat()
        else:
            self.policy = MobilityBlockerGoat()

    # let ActionMasker see through us
    def valid_action_mask(self):
        return self.env.valid_action_mask()

    def _goat_action(self) -> int:
        return self.policy.pick(self.env, self.rng)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        done = False
        while getattr(self.env, "current_player", None) == GOAT and not done:
            a = self._goat_action()
            obs, _, term, trunc, info = self.env.step(a)
            done = term or trunc
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(int(action))
        while not (term or trunc) and self.env.current_player == GOAT:
            a = self._goat_action()
            obs, r2, term, trunc, info = self.env.step(a)
            reward += float(r2)
        return obs, reward, term, trunc, info
