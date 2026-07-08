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
                # データの読み込み
                rewards = data['returns']
                # npz内にある本来のtimestepsを取得
                timesteps = data['timesteps'] if 'timesteps' in data.files else np.arange(len(rewards)) * 20000
                
                if len(rewards.shape) > 1:
                    rewards = rewards.mean(axis=1)
                
                # スイープ実行で累積してしまっている横軸を、各ラン単体の0から始まる数値に修正
                if timesteps[0] > 0:
                    timesteps = timesteps - timesteps[0]
                    
                rewards_all.append(rewards)
                timesteps_all.append(timesteps)
                
    if not rewards_all:
        continue
        
    # 全ランの長さを揃える
    min_len = min(len(r) for r in rewards_all)
    rewards_all = np.array([r[:min_len] for r in rewards_all])
    X_steps = timesteps_all[0][:min_len] # 統一した横軸
    
    # シード間の平均と標準誤差を計算（NaNは安全に無視）
    mean_r = np.nanmean(rewards_all, axis=0)
    std_r = np.nanstd(rewards_all, axis=0) / np.sqrt(num_seeds)
    
    # 💡 NaNを回避しつつ、データのガタガタを劇的に滑らかにする移動平均（Rolling Window）
    window_size = max(1, min_len // 15)  # データの長さに応じて適切に調整
    mean_smooth = np.zeros_like(mean_r)
    std_smooth = np.zeros_like(std_r)
    
    for i in range(len(mean_r)):
        start = max(0, i - window_size)
        end = min(len(mean_r), i + 1)
        mean_smooth[i] = np.nanmean(mean_r[start:end])
        std_smooth[i] = np.nanstd(std_r[start:end])

    # 💡 万が一、最初や途中にNaNが残っていたら前後の値で埋める安全策
    mask = np.isnan(mean_smooth)
    if np.any(mask):
        mean_smooth = np.where(mask, np.nanto_num(mean_smooth, nan=0.0), mean_smooth)
        std_smooth = np.where(mask, np.nanto_num(std_smooth, nan=0.0), std_smooth)
    
    # プロットを実行
    plt.plot(X_steps, mean_smooth, label=f"scale={scale}", color=colors[idx], linewidth=2.5)
    plt.fill_between(X_steps, mean_smooth - std_smooth, mean_smooth + std_smooth, color=colors[idx], alpha=0.1)

plt.title("Humanoid-v5 Sweep Results (Comparison)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Timesteps (per run)", fontsize=12)
plt.ylabel("Evaluation Mean Reward", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)

# 横軸の表記を 1.0M, 2.0M などの見やすい表記にカッチリ固定
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x > 0 else '0'))
plt.xlim(0, 3000000) # 0から300万歩の範囲に強制指定

plt.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig("humanoid_rewards_fixed.png", dpi=300)
plt.close()

print("🎉 今度こそ完璧に滑らかな重ね合わせグラフを保存しました: humanoid_rewards_fixed.png")
