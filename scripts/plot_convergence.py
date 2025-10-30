import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy, X_TIMESTEPS

# The Monitor wrapper wrote logs into ./logs (monitor.csv lives there)
results = load_results("logs")
x, y = ts2xy(results, X_TIMESTEPS)  # x = timesteps, y = episode rewards

plt.figure()
plt.plot(x, y)
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("BaghChal Training Convergence")
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/convergence.png", dpi=150)
print("Saved: logs/convergence.png")
