from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- Board / rules ---
N = 5
GOAT, TIGER = 1, 2
CORNER_IDXS = [0, 4, 20, 24]
TOTAL_GOATS = 20
TIGERS_TO_WIN_CAPTURE = 5
GOATS_TO_WIN_BLOCK = True
MAX_STEPS = 300

def in_bounds(r, c): return 0 <= r < N and 0 <= c < N
def rc(i: int) -> Tuple[int, int]: return divmod(i, N)
def iof(r: int, c: int) -> int: return r * N + c

def generate_neighbors() -> List[List[int]]:
    neigh = [[] for _ in range(N * N)]
    dirs_ortho = [(1,0),(-1,0),(0,1),(0,-1)]
    dirs_diag = [(1,1),(1,-1),(-1,1),(-1,-1)]
    for r in range(N):
        for c in range(N):
            i = iof(r,c)
            for dr,dc in dirs_ortho:
                rr,cc = r+dr, c+dc
                if in_bounds(rr,cc): neigh[i].append(iof(rr,cc))
            if ((r + c) % 2) == 0:
                for dr,dc in dirs_diag:
                    rr,cc = r+dr, c+dc
                    if in_bounds(rr,cc): neigh[i].append(iof(rr,cc))
    return [sorted(set(lst)) for lst in neigh]

NEIGHBORS = generate_neighbors()

def generate_directed_edges() -> List[Tuple[int,int]]:
    edges = []
    for u in range(N*N):
        for v in NEIGHBORS[u]:
            edges.append((u,v))
    return edges

DIRECTED_EDGES = generate_directed_edges()
EDGE_INDEX = {e:k for k,e in enumerate(DIRECTED_EDGES)}

def generate_capture_triples() -> List[Tuple[int,int,int]]:
    triples = []
    for u in range(N*N):
        ru, cu = rc(u)
        for mid in NEIGHBORS[u]:
            rm, cm = rc(mid)
            dr, dc = rm - ru, cm - cu
            w_r, w_c = rm + dr, cm + dc
            if in_bounds(w_r, w_c):
                w = iof(w_r, w_c)
                if w in NEIGHBORS[mid]:
                    triples.append((u, mid, w))
    return list(dict.fromkeys(triples))

CAPTURE_TRIPLES = generate_capture_triples()
CAPTURE_INDEX = {t:k for k,t in enumerate(CAPTURE_TRIPLES)}

# Action layout
PLACE_OFFSET = 0
NUM_PLACE = N * N
STEP_OFFSET = NUM_PLACE
NUM_STEPS = len(DIRECTED_EDGES)
CAP_OFFSET = STEP_OFFSET + NUM_STEPS
NUM_CAP = len(CAPTURE_TRIPLES)
NUM_ACTIONS = NUM_PLACE + NUM_STEPS + NUM_CAP

class BhagChalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: Optional[int] = None, reward_perspective: int = TIGER):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.reward_perspective = reward_perspective  # TIGER or GOAT
        # planes: [tiger, goat, empty, current_player, goats_left/20, goats_captured/5, phase_is_move]
        self.observation_space = spaces.Box(low=0, high=1, shape=(N, N, 7), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.board = np.zeros((N, N), dtype=np.int8)
        self.current_player = GOAT
        self.goats_to_place = TOTAL_GOATS
        self.goats_captured = 0
        self.steps = 0

    # --- helpers ---
    def seed(self, seed: Optional[int] = None): self.rng = np.random.default_rng(seed)
    def _phase_move(self) -> bool: return self.goats_to_place == 0

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None: self.seed(seed)
        self.board[:] = 0
        for i in CORNER_IDXS:
            r,c = rc(i); self.board[r,c] = TIGER
        self.current_player = GOAT
        self.goats_to_place = TOTAL_GOATS
        self.goats_captured = 0
        self.steps = 0
        return self._obs(), {"goats_to_place": self.goats_to_place}

    def _obs(self) -> np.ndarray:
        t = (self.board == TIGER).astype(np.float32)
        g = (self.board == GOAT).astype(np.float32)
        e = (self.board == 0).astype(np.float32)
        cp = np.full_like(t, 1.0 if self.current_player == GOAT else 0.0, dtype=np.float32)
        gleft = np.full_like(t, self.goats_to_place / TOTAL_GOATS, dtype=np.float32)
        gcapt = np.full_like(t, min(self.goats_captured, TIGERS_TO_WIN_CAPTURE) / TIGERS_TO_WIN_CAPTURE, dtype=np.float32)
        phase = np.full_like(t, 1.0 if self._phase_move() else 0.0, dtype=np.float32)
        return np.stack([t,g,e,cp,gleft,gcapt,phase], axis=-1)

    def _cells_of(self, piece: int) -> List[int]:
        locs = np.where(self.board == piece)
        return [iof(int(r), int(c)) for r,c in zip(locs[0], locs[1])]

    def _empty(self, i: int) -> bool:
        r,c = rc(i); return self.board[r,c] == 0

    def _valid_place_actions(self) -> List[int]:
        if self.current_player != GOAT or self.goats_to_place <= 0: return []
        empties = np.where(self.board.reshape(-1) == 0)[0].tolist()
        return [PLACE_OFFSET + i for i in empties]

    def _valid_step_actions(self) -> List[int]:
        acts = []
        if self._phase_move():
            from_cells = self._cells_of(GOAT if self.current_player == GOAT else TIGER)
        else:
            if self.current_player == GOAT: return []
            from_cells = self._cells_of(TIGER)
        for u in from_cells:
            for v in NEIGHBORS[u]:
                if self._empty(v):
                    acts.append(STEP_OFFSET + EDGE_INDEX[(u,v)])
        return acts

    def _valid_capture_actions(self) -> List[int]:
        if self.current_player != TIGER: return []
        acts = []
        for u in self._cells_of(TIGER):
            ru, cu = rc(u)
            for mid in NEIGHBORS[u]:
                rm, cm = rc(mid)
                dr, dc = rm - ru, cm - cu
                w_r, w_c = rm + dr, cm + dc
                if not in_bounds(w_r, w_c): continue
                w = iof(w_r, w_c)
                if w not in NEIGHBORS[mid]: continue
                if self.board[rm,cm] == GOAT and self.board[w_r,w_c] == 0:
                    acts.append(CAP_OFFSET + CAPTURE_INDEX[(u, mid, w)])
        return acts

    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        for a in self._valid_place_actions(): mask[a] = True
        for a in self._valid_step_actions(): mask[a] = True
        for a in self._valid_capture_actions(): mask[a] = True
        return mask

    def _tiger_legal_count(self) -> int:
        save = self.current_player
        self.current_player = TIGER
        cnt = len(self._valid_step_actions()) + len(self._valid_capture_actions())
        self.current_player = save
        return cnt

    def step(self, action: int):
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict = {}

        mask = self.valid_action_mask()
        if not mask[action]:
            reward = -0.1  # illegal (shouldn't happen with masks)
        else:
            if action < STEP_OFFSET:
                i = action - PLACE_OFFSET
                r,c = rc(i)
                self.board[r,c] = GOAT
                self.goats_to_place -= 1
            elif action < CAP_OFFSET:
                edge_idx = action - STEP_OFFSET
                u, v = DIRECTED_EDGES[edge_idx]
                ru, cu = rc(u); rv, cv = rc(v)
                moving = self.current_player
                if self.board[ru,cu] == moving and self.board[rv,cv] == 0:
                    self.board[rv,cv] = moving
                    self.board[ru,cu] = 0
                else:
                    reward = -0.1
            else:
                cap_idx = action - CAP_OFFSET
                u, mid, w = CAPTURE_TRIPLES[cap_idx]
                ru, cu = rc(u); rm, cm = rc(mid); rw, cw = rc(w)
                if self.board[ru,cu] == TIGER and self.board[rm,cm] == GOAT and self.board[rw,cw] == 0:
                    self.board[rw,cw] = TIGER
                    self.board[ru,cu] = 0
                    self.board[rm,cm] = 0
                    self.goats_captured += 1
                else:
                    reward = -0.1

        # --- termination / reward from fixed perspective ---
        winner: Optional[int] = None
        reason: Optional[str] = None

        if self.goats_captured >= TIGERS_TO_WIN_CAPTURE:
            winner, reason = TIGER, "capture"
        elif GOATS_TO_WIN_BLOCK and self._tiger_legal_count() == 0:
            winner, reason = GOAT, "block"
        elif self.steps >= MAX_STEPS:
            truncated = True
            reason = "draw"

        if winner is not None:
            terminated = True
            reward = 1.0 if winner == self.reward_perspective else -1.0
        elif truncated:
            reward = 0.0

        info.update({"winner": winner, "reason": reason})

        # advance turn (harmless even on terminal states)
        self.current_player = GOAT if self.current_player == TIGER else TIGER
        obs = self._obs()
        return obs, reward, terminated, truncated, info


def mask_fn(env):
    # ActionMasker calls this in each step; must return a 1D boolean mask
    return env.valid_action_mask()
