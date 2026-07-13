import os
import numpy as np
import matplotlib.pyplot as plt

# 設定
SAVE_DIR = "./npz_logs_humanoid"
SCALES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
WINDOW_SIZE = 15

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# 💡 2行 × 3列 の大きなグラフ全体のキャンバスを作成 (横15インチ, 縦9インチ)
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
axes = axes.flatten()  # 2D配列のインデックスを1次元にしてループしやすくする

for idx, scale in enumerate(SCALES):
    ax = axes[idx]  # 今回担当する小窓
    all_returns = []
    min_len = 999999
    t_fixed = None

    # 5つのシードデータを集計
    for seed in range(5):
        npz_path = f"{SAVE_DIR}/gaussian_scale{scale}_run{seed}.npz"
        if not os.path.exists(npz_path):
            continue
            
        data = np.load(npz_path)
        r = data["returns"]
        t = data["timesteps"]
        
        all_returns.append(r)
        if len(r) < min_len:
            min_len = len(r)
            t_fixed = t

    if len(all_returns) == 0:
        # データがない枠は非表示にする
        ax.axis('off')
        continue

    all_returns = np.array([r[:min_len] for r in all_returns])
    t_fixed = t_fixed[:min_len]

    # 平均と標準偏差の計算
    raw_mean = all_returns.mean(axis=0)
    raw_std = all_returns.std(axis=0)

    smooth_mean = moving_average(raw_mean, WINDOW_SIZE)
    smooth_std = moving_average(raw_std, WINDOW_SIZE)
    trim = WINDOW_SIZE // 2

    # 小窓ごとにプロット
    ax.plot(
        t_fixed[trim:-trim],
        smooth_mean[trim:-trim],
        linewidth=2.0,
        color=COLORS[idx],
        label=f"Scale = {scale}"
    )
    
    ax.fill_between(
        t_fixed[trim:-trim],
        smooth_mean[trim:-trim] - smooth_std[trim:-trim],
        smooth_mean[trim:-trim] + smooth_std[trim:-trim],
        color=COLORS[idx],
        alpha=0.15
    )

    # 各小窓の装飾
    ax.set_title(f"Scale = {scale}", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(0, 3_000_000)
    
    # 横軸の表記を「1.0M, 2.0M, 3.0M」に統一
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x > 0 else '0'))
    
    # 左側の列のグラフにだけ縦軸ラベルをつける（見栄えすっきり）
    if idx % 3 == 0:
        ax.set_ylabel("Mean Episode Return", fontsize=10)
    
    # 下側の行のグラフにだけ横軸ラベルをつける
    if idx >= 3:
        ax.set_xlabel("Timesteps", fontsize=10)

# 全体のタイトル調整とレイアウト最適化
plt.suptitle("Humanoid-v5 SAC with Gaussian Prior (Individual Parameters)", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout()

# 1枚の大きな画像として保存
output_path = f"{SAVE_DIR}/humanoid_6plots_grid.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("====================================================")
print(f"🎉 2行3列の6面マルチプロット画像を生成しました！")
print(f"保存先: {output_path}")
print("====================================================")
