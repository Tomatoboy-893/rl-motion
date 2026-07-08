import os
import numpy as np
import matplotlib.pyplot as plt

# 設定
log_dir = "npz_logs_humanoid"
scales = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
num_seeds = 5

colors = plt.cm.plasma(np.linspace(0, 0.85, len(scales)))
plt.figure(figsize=(9, 6))

for idx, scale in enumerate(scales):
    rewards_all = []
    timesteps_all = []
    
    for seed in range(num_seeds):
        reward_path = os.path.join(log_dir, f"gaussian_scale{scale}_run{seed}.npz")
        if os.path.exists(reward_path):
            with np.load(reward_path) as data:
                rewards = data['returns']
                timesteps = data['timesteps'] if 'timesteps' in data.files else np.arange(len(rewards)) * 20000
                
                if len(rewards.shape) > 1:
                    rewards = rewards.mean(axis=1)
                
                # 横軸を各ラン単体の0から始まる数値に修正
                if timesteps[0] > 0:
                    timesteps = timesteps - timesteps[0]
                    
                rewards_all.append(rewards)
                timesteps_all.append(timesteps)
                
    if not rewards_all:
        continue
        
    # 全ランの長さを揃える
    min_len = min(len(r) for r in rewards_all)
    rewards_all = np.array([r[:min_len] for r in rewards_all])
    X_steps = timesteps_all[0][:min_len]
    
    # 1. 最初にシード間の平均を計算
    mean_r = np.nanmean(rewards_all, axis=0)
    std_r = np.nanstd(rewards_all, axis=0) / np.sqrt(num_seeds)
    
    # 万が一のNaNは0埋め
    mean_r = np.nan_to_num(mean_r, nan=0.0)
    std_r = np.nan_to_num(std_r, nan=0.0)
    
    # 💡 改善点：forループを廃止し、NumPyの一括コンボリューションで移動平均を計算（爆速）
    window_size = max(1, min_len // 15)
    kernel = np.ones(window_size) / window_size
    
    # 境界の処理を綺麗にするためmode='same'で一瞬で滑らかにする
    mean_smooth = np.convolve(mean_r, kernel, mode='same')
    std_smooth = np.convolve(std_r, kernel, mode='same')

    # プロット
    plt.plot(X_steps, mean_smooth, label=f"scale={scale}", color=colors[idx], linewidth=2.5)
    plt.fill_between(X_steps, mean_smooth - std_smooth, mean_smooth + std_smooth, color=colors[idx], alpha=0.1)

plt.title("Humanoid-v5 Sweep Results (Comparison)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Timesteps (per run)", fontsize=12)
plt.ylabel("Evaluation Mean Reward", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)

plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x > 0 else '0'))
plt.xlim(0, 3000000)

plt.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig("humanoid_rewards_fixed.png", dpi=300)
plt.close()

print("🎉 爆速で重ね合わせグラフを保存しました: humanoid_rewards_fixed.png")
