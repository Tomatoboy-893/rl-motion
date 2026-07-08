import os
import numpy as np
import matplotlib.pyplot as plt

# 設定
log_dir = "npz_logs_humanoid"
scales = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
num_seeds = 5

# 💡 移動平均（スムーズさ）のウィンドウサイズ（大きくするほど滑らかになります）
def smooth_curve(current, previous, weight=0.65):
    smoothed = []
    for point in current:
        if smoothed:
            previous_smoothed = smoothed[-1]
            smoothed.append(previous_smoothed * weight + point * (1 - weight))
        else:
            smoothed.append(point)
    return np.array(smoothed)

colors = plt.cm.plasma(np.linspace(0, 0.85, len(scales))) # 見やすいグラデーションカラー
plt.figure(figsize=(9, 6))

for idx, scale in enumerate(scales):
    rewards_all = []
    
    for seed in range(num_seeds):
        reward_path = os.path.join(log_dir, f"gaussian_scale{scale}_run{seed}.npz")
        if os.path.exists(reward_path):
            with np.load(reward_path) as data:
                rewards = data['returns']
                if len(rewards.shape) > 1:
                    rewards = rewards.mean(axis=1)
                rewards_all.append(rewards)
                
    if not rewards_all:
        continue
        
    # 全ランの長さを揃える
    min_len = min(len(r) for r in rewards_all)
    rewards_all = np.array([r[:min_len] for r in rewards_all])
    
    # 💡 横軸をすべての実験で 0 からスタートするように統一 (最大300万歩)
    # 元コードの eval_freq が 20000 だった場合のスケール
    timesteps = np.linspace(0, 3000000, min_len)
    
    mean_r = np.nanmean(rewards_all, axis=0)
    std_r = np.nanstd(rewards_all, axis=0) / np.sqrt(num_seeds)
    
    # 💡 滑らかに補正
    mean_r_smoothed = smooth_curve(mean_r, mean_r)
    
    # プロット
    plt.plot(timesteps, mean_r_smoothed, label=f"scale={scale}", color=colors[idx], linewidth=2.5)
    plt.fill_between(timesteps, mean_r_smoothed - std_r, mean_r_smoothed + std_r, color=colors[idx], alpha=0.1)

plt.title("Humanoid-v5 Sweep Results (Comparison)", fontsize=14, fontweight='bold')
plt.xlabel("Timesteps (per run)", fontsize=12)
plt.ylabel("Evaluation Mean Reward", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)

# X軸の表記を 1M, 2M, 3M のように見やすくフォーマット
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x > 0 else '0'))

plt.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig("humanoid_rewards_fixed.png", dpi=300)
plt.close()

print("🎉 修正版の綺麗なグラフを保存しました: humanoid_rewards_fixed.png")
