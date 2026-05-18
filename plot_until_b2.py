import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "./npz_logs"

b_values = [0.1, 0.5, 1.0, 2.0]

plt.figure(figsize=(8,6))

for b in b_values:

    data = np.load(f"{SAVE_DIR}/laplace_b{b}_mean_std.npz")

    t = data["timesteps"]
    mean = data["mean"]
    std = data["std"]

    plt.plot(t, mean, label=f"Laplace b={b}")

    plt.fill_between(
        t,
        mean - std,
        mean + std,
        alpha=0.2
    )


plt.xlabel("Timesteps")
plt.ylabel("Mean Episode Return")
plt.title("Laplace Prior Parameter Sweep")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(f"{SAVE_DIR}/laplace_b_sweep_until_2.png")
plt.show()

print("Saved: laplace_b_sweep_until_2.png")