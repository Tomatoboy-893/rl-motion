import os
import numpy as np
import matplotlib.pyplot as plt

# 設定
SAVE_DIR = "./npz_logs_humanoid"
SCALES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

# 論文でよく使われるスタイリッシュなカラーパレット
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

plt.figure(figsize=(9, 6))

for idx, scale in enumerate(SCALES):
    all_returns = []
    min_len = 999999
    t_fixed = None

    # 5つのシードデータを集計
    for seed in range(5):
        npz_path = f"{SAVE_DIR}/gaussian_scale{scale}_run{seed}.npz"
        if not os.path.exists(npz_path):
            print(f"Warning: Missing {npz_path}")
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

    # 長さを揃えて行列化
    all_returns = np.array([r[:min_len] for r in all_returns])
    t_fixed = t_fixed[:min_len]

    # 平均と標準偏差を計算
    mean = all_returns.mean(axis=0)
    std = all_returns.std(axis=0)

    # 💡 600点の高密度データなので、そのままplotするだけでツルツルになります！
    plt.plot(
        t_fixed,
        mean,
        linewidth=2.0,
        color=COLORS[idx],
        label=f"Scale = {scale}"
    )
    
    # 標準偏差のシャドウ（半透明）
    plt.fill_between(
        t_fixed,
        mean - std,
        mean + std,
        color=COLORS[idx],
        alpha=0.12
    )

# グラフの装飾
plt.xlabel("Timesteps", fontsize=12, fontweight="bold", labelpad=8)
plt.ylabel("Mean Episode Return", fontsize=12, fontweight="bold", labelpad=8)
plt.title("Humanoid-v5 SAC with Gaussian Prior (5,000 Step Freq)", fontsize=14, fontweight="bold", pad=15)

# 横軸を「1.0M, 2.0M, 3.0M」のように100万単位の見やすい表記にする
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x > 0 else '0'))
plt.xlim(0, 3_000_000)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", fontsize=10)
plt.tight_layout()

# 画像として保存
output_path = f"{SAVE_DIR}/humanoid_sweep_final.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("====================================================")
print(f"🎉 6スケール分のツルツル統合グラフを生成しました！")
print(f"保存先: {output_path}")
print("====================================================")
