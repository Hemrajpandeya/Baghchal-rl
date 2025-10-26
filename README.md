# BaghChal RL (start with 5×5)

Teach a computer to play **BaghChal** using **Reinforcement Learning**.

- **Now:** 5×5 board (simpler to start)
- **Later:** 7×7 + multi-jump using the same codebase
- RL algorithm: **MaskablePPO** (from `sb3-contrib`) so the agent avoids illegal moves

---

## 1) Requirements

- Python 3.9+ (3.10/3.11 recommended)
- Git (optional but recommended)

> If you’re on Windows, install [Python](https://www.python.org/downloads/) and check “Add Python to PATH”.

---

## 2) Setup (one-time)

### Windows (PowerShell)
```powershell
git clone https://github.com/Hemrajpandeya/Baghchal-rl
cd Baghchal-rl

python -m venv .venv
.venv\Scripts\activate

python -m pip install -U pip
pip install gymnasium stable-baselines3 "sb3-contrib" numpy tensorboard

# make local package importable as `from baghchal...`
pip install -e .
