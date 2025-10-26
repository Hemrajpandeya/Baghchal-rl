python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install gymnasium stable-baselines3 "sb3-contrib" numpy tensorboard
pip install -e .
python scripts/train_baghchal_simple.py
