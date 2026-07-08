import os
import numpy as np
import matplotlib.pyplot as plt

# 設定
log_dir = "npz_logs_humanoid"
scales = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
num_seeds = 5
eval_freq = 20000  # 評価の間隔（ステップ数）

# グラフの準備 (2行1列のプロット)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
colors = plt.cm.jet(np.linspace(0, 1, len(scales)))

for idx, scale in enumerate(scales):
    rewards_all = []
    entropies_all = []
    
    for seed in range(num_seeds):
        # ファイルパスの構築
        reward_path = os.path.join(log_dir, f"gaussian_scale{scale}_run{seed}.npz")
        entropy_path = os.path.join(log_dir, f"gaussian_scale{scale}_run{seed}_entropy.npz")
        
        # データの読み込み
        if os.path.exists(reward_path) and os.path.exists(entropy_path):
            with np.load(reward_path) as data:
                # SB3のEvalCallbackの出力形式（通常 'timesteps' と 'results'）
                # もし自作カスタムログなら、保存時のキー名（例: 'rewards'）に合わせてください
                rewards = data['results'].mean(axis=1) if len(data['results'].shape) > 1 else data['results']
                rewards_all.append(rewards)
                
            with np.load(entropy_path) as data:
                # エントロピーデータの読み込み（キー名は保存時に合わせたものにしてください、例: 'entropies'）
                entropies = data['entropies'] if 'entropies' in data else data[data.files[0]]
                entropies_all.append(entropies)
                
    if not rewards_all:
        continue
        
    # 最小の長さに揃える（万が一途中で落ちたランがあってもプロットできるように）
    min_len_r = min(len(r) for r in rewards_all)
    min_len_e = min(len(e) for e in entropies_all)
    
    rewards_all = np.array([r[:min_len_r] for r in rewards_all])
    entropies_all = np.array([e[:min_len_e] for e in entropies_all])
    
    # X軸（タイムステップ）の計算
    timesteps_r = np.arange(1, min_len_r + 1) * eval_freq
    timesteps_e = np.arange(1, min_len_e + 1) * eval_freq # もしエントロピーの保存頻度が違うなら要調整
    
    # 平均と標準誤差（ばらつき）の計算
    mean_r = np.mean(rewards_all, axis=0)
    std_r = np.std(rewards_all, axis=0) / np.sqrt(num_seeds) # 標準誤差
    
    mean_e = np.mean(entropies_all, axis=0)
    std_e = np.std(entropies_all, axis=0) / np.sqrt(num_seeds)

    # 1. リターンのプロット
    ax1.plot(timesteps_r, mean_r, label=f"scale={scale}", color=colors[idx], linewidth=2)
    ax1.fill_between(timesteps_r, mean_r - std_r, mean_r + std_r, color=colors[idx], alpha=0.15)
    
    # 2. エントロピーのプロット
    ax2.plot(timesteps_e, mean_e, label=f"scale={scale}", color=colors[idx], linewidth=2)
    ax2.fill_between(timesteps_e, mean_e - std_e, mean_e + std_e, color=colors[idx], alpha=0.15)

# グラフの装飾（リターン）
ax1.set_title("Evaluation Episode Reward", fontsize=14)
ax1.set_xlabel("Timesteps", fontsize=12)
ax1.set_ylabel("Mean Reward", fontsize=12)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(loc="lower right")

# グラフの装飾（エントロピー）
ax2.set_title("Policy Entropy", fontsize=14)
ax2.set_xlabel("Timesteps", fontsize=12)
ax2.set_ylabel("Entropy", fontsize=12)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(loc="upper right")

plt.tight_layout()
output_fig = "humanoid_sweep_results.png"
plt.savefig(output_fig, dpi=300)
print(f"🎉 グラフを保存しました: {output_fig}")
