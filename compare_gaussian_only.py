# compare_gaussian_only.py

import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "./npz_logs"

# 全 rho
rho_list = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

plt.figure(figsize=(9, 6))

for rho in rho_list:

    data = np.load(
        f"{SAVE_DIR}/gaussian_rho{rho}_mean_std.npz"
    )

    t = data["timesteps"]
    mean = data["mean"]
    std = data["std"]

    plt.plot(
        t,
        mean,
        label=f"Gaussian rho={rho}"
    )

    plt.fill_between(
        t,
        mean - std,
        mean + std,
        alpha=0.2,
    )

plt.xlabel("Timesteps")
plt.ylabel("Mean Episode Return")

plt.title("Gaussian Prior Parameter Sweep")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{SAVE_DIR}/gaussian_only_comparison.png",
    dpi=300
)

plt.show()
