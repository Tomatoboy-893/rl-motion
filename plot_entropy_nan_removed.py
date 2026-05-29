# plot_entropy_nan_removed.py

import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "./npz_logs"

# ===============================
# load
# ===============================
gaussian = np.load(
    f"{SAVE_DIR}/gaussian_1.0_entropy_mean_std.npz"
)

laplace = np.load(
    f"{SAVE_DIR}/laplace_1.0_entropy_mean_std.npz"
)

# ===============================
# data
# ===============================
g_mean = gaussian["mean"]
g_std = gaussian["std"]

l_mean = laplace["mean"]
l_std = laplace["std"]

# ===============================
# NaN remove
# ===============================
g_mask = np.isfinite(g_mean)
l_mask = np.isfinite(l_mean)

g_x = np.arange(len(g_mean))[g_mask]
l_x = np.arange(len(l_mean))[l_mask]

g_mean = g_mean[g_mask]
g_std = g_std[g_mask]

l_mean = l_mean[l_mask]
l_std = l_std[l_mask]

# ===============================
# plot
# ===============================
plt.figure(figsize=(8,6))

# Gaussian
plt.plot(
    g_x,
    g_mean,
    label="Gaussian Prior"
)

plt.fill_between(
    g_x,
    g_mean - g_std,
    g_mean + g_std,
    alpha=0.2,
)

# Laplace
plt.plot(
    l_x,
    l_mean,
    label="Laplace Prior"
)

plt.fill_between(
    l_x,
    l_mean - l_std,
    l_mean + l_std,
    alpha=0.2,
)

# ===============================
# figure
# ===============================
plt.xlabel("Training Updates")
plt.ylabel("Policy Entropy")

plt.title("Policy Entropy Comparison")

plt.grid(True)
plt.legend()

plt.tight_layout()

# ===============================
# save
# ===============================
plt.savefig(
    f"{SAVE_DIR}/entropy_comparison_nan_removed.png",
    dpi=300
)

plt.show()

print("===================================")
print("Saved: entropy_comparison_nan_removed.png")
print("===================================")
