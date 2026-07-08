import os
import numpy as np
import matplotlib.pyplot as plt

# 設定
log_dir = "npz_logs_humanoid"
scales = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
num_seeds = 5
eval_freq = 20000  # 評価の間隔

colors = plt.cm.jet(np.linspace(0, 1, len(scales)))

# データを格納する辞書
plot_data = {scale: {'timesteps_r': None, 'mean_r': None, 'std_r': None,
                     'timesteps_e': None, 'mean_e': None, 'std_e': None} for scale in scales}

# --- 1. データの読み込みと計算 ---
for idx, scale in enumerate(scales):
    rewards_all = []
    entropies_all = []
    
    for seed in range(num_seeds):
        reward_path = os.path.join(log_dir, f"gaussian_scale{scale}_run{seed}.npz")
        entropy_path = os.path.join(log_dir, f"gaussian_scale{scale}_run{seed}_entropy.npz")
        
        if os.path.exists(reward_path) and os.path.exists(entropy_path):
            with np.load(reward_path) as data:
                rewards = data['returns']
                if len(rewards.shape) > 1:
                    rewards = rewards.mean(axis=1)
                rewards_all.append(rewards)
                
            with np.load(entropy_path) as data:
                entropies = data['entropy']
                entropies_all.append(entropies)
                
    if not rewards_all:
        continue
        
    min_len_r = min(len(r) for r in rewards_all)
    min_len_e = min(len(e) for e in entropies_all)
    
    rewards_all = np.array([r[:min_len_r] for r in rewards_all])
    entropies_all = np.array([e[:min_len_e] for e in entropies_all])
    
    plot_data[scale]['timesteps_r'] = np.arange(1, min_len_r + 1) * eval_freq
    plot_data[scale]['timesteps_e'] = np.arange(1, min_len_e + 1) * eval_freq
    
    plot_data[scale]['mean_r'] = np.nanmean(rewards_all, axis=0)
    plot_data[scale]['std_r'] = np.nanstd(rewards_all, axis=0) / np.sqrt(num_seeds)
    
    plot_data[scale]['mean_e'] = np.nanmean(entropies_all, axis=0)
    plot_data[scale]['std_e'] = np.nanstd(entropies_all, axis=0) / np.sqrt(num_seeds)

# --- 2. リターン（報酬）のグラフ生成と保存 ---
plt.figure(figsize=(8, 5))
for idx, scale in enumerate(scales):
    d = plot_data[scale]
    if d['mean_r'] is not None:
        plt.plot(d['timesteps_r'], d['mean_r'], label=f"scale={scale}", color=colors[idx], linewidth=2)
        plt.fill_between(d['timesteps_r'], d['mean_r'] - d['std_r'], d['mean_r'] + d['std_r'], color=colors[idx], alpha=0.15)

plt.title("Evaluation Episode Reward", fontsize=14)
plt.xlabel("Timesteps", fontsize=12)
plt.ylabel("Mean Reward", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("humanoid_rewards.png", dpi=300)
plt.close()
print("🎉 リターンのグラフを保存しました: humanoid_rewards.png")

# --- 3. エントロピーのグラフ生成と保存 ---
plt.figure(figsize=(8, 5))
for idx, scale in enumerate(scales):
    d = plot_data[scale]
    if d['mean_e'] is not None:
        plt.plot(d['timesteps_e'], d['mean_e'], label=f"scale={scale}", color=colors[idx], linewidth=2)
        plt.fill_between(d['timesteps_e'], d['mean_e'] - d['std_e'], d['mean_e'] + d['std_e'], color=colors[idx], alpha=0.15)

plt.title("Policy Entropy", fontsize=14)
plt.xlabel("Timesteps", fontsize=12)
plt.ylabel("Entropy", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("humanoid_entropy.png", dpi=300)
plt.close()
print("🎉 エントロピーのグラフを保存しました: humanoid_entropy.png")
