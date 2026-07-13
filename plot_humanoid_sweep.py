import os
import numpy as np
import matplotlib.pyplot as plt

# 設定
SAVE_DIR = "./npz_logs_humanoid"
SCALES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# 💡 移動平均のウィンドウサイズ（大きくするほど線がツルツルになります。15〜20が最適）
WINDOW_SIZE = 15

def moving_average(data, window_size):
    """データの移動平均を計算する関数（端の処理も綺麗に行います）"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

plt.figure(figsize=(9, 6))

for idx, scale in enumerate(SCALES):
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
        continue

    all_returns = np.array([r[:min_len] for r in all_returns])
    t_fixed = t_fixed[:min_len]

    # 平均と標準偏差の生データ
    raw_mean = all_returns.mean(axis=0)
    raw_std = all_returns.std(axis=0)

    # 💡 ここで移動平均を適用して、激しいギザギザ（ノイズ）だけを綺麗に溶かします！
    smooth_mean = moving_average(raw_mean, WINDOW_SIZE)
    smooth_std = moving_average(raw_std, WINDOW_SIZE)

    # グラフの端っこ（ウィンドウの半分）は移動平均の計算上歪みやすいので、綺麗な部分だけをプロット
    trim = WINDOW_SIZE // 2
    
    plt.plot(
        t_fixed[trim:-trim],
        smooth_mean[trim:-trim],
        linewidth=2.2,
        color=COLORS[idx],
        label=f"Scale = {scale}"
    )
    
    plt.fill_between(
        t_fixed[trim:-trim],
        smooth_mean[trim:-trim] - smooth_std[trim:-trim],
        smooth_mean[trim:-trim] + smooth_std[trim:-trim],
        color=COLORS[idx],
        alpha=0.10  # シャドウを少し薄くして重ね合わせを見やすく
    )

# グラフの装飾
plt.xlabel("Timesteps", fontsize=12, fontweight="bold", labelpad=8)
plt.ylabel("Mean Episode Return", fontsize=12, fontweight="bold", labelpad=8)
plt.title("Humanoid-v5 SAC with Gaussian Prior (Smoothed Trends)", fontsize=14, fontweight="bold", pad=15)

plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x > 0 else '0'))
plt.xlim(0, 3_000_000)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", fontsize=10)
plt.tight_layout()

output_path = f"{SAVE_DIR}/humanoid_sweep_final.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("====================================================")
print(f"🎉 ノイズを消し去ったツルツル統合グラフを再生成しました！")
print("====================================================")
