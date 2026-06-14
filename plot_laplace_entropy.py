# plot_laplace_entropy.py

import numpy as np
import matplotlib.pyplot as plt
import os

SAVE_DIR = "./npz_logs"


def load(scale):
    data = np.load(f"{SAVE_DIR}/laplace_{scale}_entropy_mean_std.npz")
    return data["mean"], data["std"]


def main():

    scales = [0.1, 0.5, 1.0, 2.0]

    plt.figure(figsize=(7, 5))

    for scale in scales:

        mean, std = load(scale)

        x = np.arange(len(mean))

        plt.plot(x, mean, label=f"scale={scale}")
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Training steps (evaluation index)")
    plt.ylabel("Policy entropy")
    plt.title("Laplace Prior: Entropy Comparison")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    out_path = os.path.join(SAVE_DIR, "laplace_entropy_only.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
