from baghchal.rl.env_baghchal import BhagChalEnv
env = BhagChalEnv(seed=0)
obs, info = env.reset()
mask = env.valid_action_mask()
print("Obs shape:", getattr(obs, "shape", type(obs)))
print("Num actions:", env.action_space.n)
print("Valid actions at start:", int(mask.sum()))
