import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

SAVE_DIR = "./npz_logs_humanoid"
SCALES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
NUM_RUNS = 5

plt.figure(figsize=(9, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(SCALES)))

# 1. ADR (各スケール) のプロット
for idx, scale in enumerate(SCALES):
    returns_all = []
    timesteps = None

    for run in range(NUM_RUNS):
        # 明示的に returns 側の npz パスを指定（_entropy.npz を回避）
        path = f"{SAVE_DIR}/gaussian_scale{scale}_run{run}.npz"

        if not os.path.exists(path):
            continue

        data = np.load(path)
        if "returns" not in data:
            continue

        returns = data["returns"]
        t = data["timesteps"]

        returns_all.append(returns)

        if timesteps is None:
            timesteps = t

    if len(returns_all) == 0:
        continue

    min_len = min(len(r) for r in returns_all)
    returns_all = np.array([r[:min_len] for r in returns_all])
    timesteps = timesteps[:min_len]

    if timesteps[0] > 0:
        timesteps = timesteps - timesteps[0]

    mean = np.nanmean(returns_all, axis=0)
    std = np.nanstd(returns_all, axis=0)

    smooth_sigma = 5.0 
    mean_smoothed = gaussian_filter1d(mean, sigma=smooth_sigma)
    std_smoothed = gaussian_filter1d(std, sigma=smooth_sigma)

    plt.plot(
        timesteps,
        mean_smoothed,
        linewidth=2,
        color=colors[idx],
        label=f"Gaussian rho={scale}"
    )

    plt.fill_between(
        timesteps,
        mean_smoothed - std_smoothed,
        mean_smoothed + std_smoothed,
        color=colors[idx],
        alpha=0.15
    )

# 2. 標準SAC（ベースライン）のプロット
baseline_returns = []
baseline_timesteps = None

for run in range(NUM_RUNS):
    path = f"{SAVE_DIR}/sac_baseline_run{run}.npz"
    if os.path.exists(path):
        data = np.load(path)
        if "returns" in data:
            baseline_returns.append(data["returns"])
            if baseline_timesteps is None:
                baseline_timesteps = data["timesteps"]

if len(baseline_returns) > 0:
    min_len_b = min(len(r) for r in baseline_returns)
    baseline_returns = np.array([r[:min_len_b] for r in baseline_returns])
    
    if baseline_timesteps is not None:
        baseline_timesteps = baseline_timesteps[:min_len_b]
        if baseline_timesteps[0] > 0:
            baseline_timesteps = baseline_timesteps - baseline_timesteps[0]
    else:
        baseline_timesteps = np.arange(min_len_b) * 5000

    b_mean = np.nanmean(baseline_returns, axis=0)
    b_std = np.nanstd(baseline_returns, axis=0)

    b_mean_smooth = gaussian_filter1d(b_mean, sigma=smooth_sigma)
    b_std_smooth = gaussian_filter1d(b_std, sigma=smooth_sigma)

    plt.plot(
        baseline_timesteps,
        b_mean_smooth,
        linewidth=2.5,
        color="black",
        linestyle="--",
        label="Standard SAC"
    )
    plt.fill_between(
        baseline_timesteps,
        b_mean_smooth - b_std_smooth,
        b_mean_smooth + b_std_smooth,
        color="black",
        alpha=0.1
    )

plt.xlabel("Timesteps", fontsize=13)
plt.ylabel("Mean Episode Return", fontsize=13)
plt.title("Humanoid-v5 Performance Comparison", fontsize=15)

plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x > 0 else '0'))
if timesteps is not None:
    plt.xlim(0, timesteps[-1])

plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(loc="upper left")
plt.tight_layout()

output_file = f"{SAVE_DIR}/humanoid_gaussian_return.png"
plt.savefig(output_file, dpi=300)
plt.close()

print("===================================")
print("🎉 グラフ生成が完了しました！")
print(f"保存先: {output_file}")
print("===================================")
EOF
