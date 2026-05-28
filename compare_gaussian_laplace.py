import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "./npz_logs"

params = [0.1, 0.5, 1.0, 2.0]

plt.figure(figsize=(8, 6))

for p in params:

    # Gaussian
    g = np.load(
        f"{SAVE_DIR}/gaussian_rho{p}_mean_std.npz"
    )

    plt.plot(
        g["timesteps"],
        g["mean"],
        label=f"Gaussian rho={p}"
    )

    # Laplace
    l = np.load(
        f"{SAVE_DIR}/laplace_b{p}_mean_std.npz"
    )

    plt.plot(
        l["timesteps"],
        l["mean"],
        linestyle="--",
        label=f"Laplace b={p}"
    )

plt.xlabel("Timesteps")
plt.ylabel("Mean Episode Return")

plt.title("Gaussian vs Laplace Prior")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{SAVE_DIR}/gaussian_vs_laplace.png",
    dpi=300
)

plt.show()
