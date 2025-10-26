# 🐅 BaghChal-RL (5×5 now → 7×7 later)

Train an RL agent to play **BaghChal** (Tiger & Goat).  
Uses **Gymnasium** + **Stable-Baselines3 (MaskablePPO)**.

---

## Quick Start — GPU (Conda)

```bash
# 1) create & activate env
conda create -n baghchal-rl-gpu python=3.11 -y
conda activate baghchal-rl-gpu

# 2) install GPU PyTorch (bundled CUDA)
conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=12.1
# (fallback if needed) conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=11.8

# 3) other deps
conda install -y -c conda-forge gymnasium numpy tensorboard
pip install -U stable-baselines3 sb3-contrib

# 4) clone & install this project
git clone https://github.com/Hemrajpandeya/Baghchal-rl.git
cd Baghchal-rl
pip install -e .

# 5) sanity check
python quickcheck.py
# Expect:
# Obs shape: (5, 5, 7)
# Num actions: 217
# Valid actions at start: 21

# 6) train (uses GPU)
# ensure scripts/train_baghchal_simple.py contains: device="cuda"
python scripts/train_baghchal_simple.py
