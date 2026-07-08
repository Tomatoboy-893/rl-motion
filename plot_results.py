import os
import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "./npz_logs_humanoid"
SCALES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
NUM_RUNS = 5

plt.figure(figsize=(8, 6))

for scale in SCALES:
    returns_all = []
    timesteps = None

    for run in range(NUM_RUNS):
        path = f"{SAVE_DIR}/gaussian_scale{scale}_run{run}.npz"

        if not os.path.exists(path):
            print(f"Not found: {path}")
            continue

        data = np.load(path)
        returns = data["returns"]
        t = data["timesteps"]

        returns_all.append(returns)

        if timesteps is None:
            timesteps = t

    if len(returns_all) == 0:
        print(f"No data for scale={scale}")
        continue

    # 長さを揃える
    min_len = min(len(r) for r in returns_all)
    returns_all = np.array([r[:min_len] for r in returns_all])
    timesteps = timesteps[:min_len]

    # 💡 累積してしまっている横軸を、各ラン単体の「0スタート」に綺麗に補正
    if timesteps[0] > 0:
        timesteps = timesteps - timesteps[0]

    # 💡 NaNを安全に除外して平均と標準偏差を計算
    mean = np.nanmean(returns_all, axis=0)
    std = np.nanstd(returns_all, axis=0)

    # プロット
    plt.plot(
        timesteps,
        mean,
        linewidth=2,
        label=f"scale={scale}"
    )

    plt.fill_between(
        timesteps,
        mean - std,
        mean + std,
        alpha=0.2
    )

plt.xlabel("Timesteps (per run)", fontsize=13)
plt.ylabel("Mean Episode Return", fontsize=13)
plt.title("Humanoid-v5 Gaussian Prior", fontsize=15)

# X軸の表記を 1.0M, 2.0M のように見やすくフォーマット
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x > 0 else '0'))
plt.xlim(0, timesteps[-1])

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{SAVE_DIR}/humanoid_gaussian_return.png",
    dpi=300
)
plt.close()

print("===================================")
print("Saved:")
print(f"{SAVE_DIR}/humanoid_gaussian_return.png")
print("===================================")
