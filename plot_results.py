import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # 💡 線を滑らかにする最強の関数

SAVE_DIR = "./npz_logs_humanoid"
SCALES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
NUM_RUNS = 5

plt.figure(figsize=(9, 6))

# 💡 参考グラフに近い、見やすくて綺麗なカラーマップを設定
colors = plt.cm.tab10(np.linspace(0, 1, len(SCALES)))

for idx, scale in enumerate(SCALES):
    returns_all = []
    timesteps = None

    for run in range(NUM_RUNS):
        path = f"{SAVE_DIR}/gaussian_scale{scale}_run{run}.npz"

        if not os.path.exists(path):
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

    # 横軸の0スタート補正
    if timesteps[0] > 0:
        timesteps = timesteps - timesteps[0]

    # 平均と標準偏差を計算
    mean = np.nanmean(returns_all, axis=0)
    std = np.nanstd(returns_all, axis=0)

    # 💡 階段状のガタガタを綺麗に消し去り、参考グラフのように滑らかにする処理
    # sigmaの値（3〜7程度）を大きくするほど、よりツルツルの滑らかな線になります
    smooth_sigma = 5.0 
    mean_smoothed = gaussian_filter1d(mean, sigma=smooth_sigma)
    std_smoothed = gaussian_filter1d(std, sigma=smooth_sigma)

    # 💡 グラフのプロット（カクカクを排除し、滑らかな線のみを描画）
    plt.plot(
        timesteps,
        mean_smoothed,
        linewidth=2,
        color=colors[idx],
        label=f"Gaussian rho={scale}"  # 参考グラフの表記に合わせました
    )

    plt.fill_between(
        timesteps,
        mean_smoothed - std_smoothed,
        mean_smoothed + std_smoothed,
        color=colors[idx],
        alpha=0.15  # シャドウの濃さを参考グラフ風に調整
    )

plt.xlabel("Timesteps", fontsize=13)
plt.ylabel("Mean Episode Return", fontsize=13)
plt.title("Gaussian Prior Parameter Sweep", fontsize=15)  # タイトルも統一

# X軸の表記を 1.0M, 2.0M のように見やすくフォーマット
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x > 0 else '0'))
plt.xlim(0, timesteps[-1])

plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(loc="upper left")
plt.tight_layout()

plt.savefig(
    f"{SAVE_DIR}/humanoid_gaussian_return.png",
    dpi=300
)
plt.close()

print("===================================")
print("🎉 参考グラフと同じ滑らかさで保存しました！")
print(f"{SAVE_DIR}/humanoid_gaussian_return.png")
print("===================================")
