# visualize_humanoid_comparison.py

import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "./npz_logs_humanoid"
seed = 0

# ファイル名の定義
base_prefix = f"{SAVE_DIR}/sac_baseline_seed{seed}"
gauss_prefix = f"{SAVE_DIR}/gaussian_rho1.0_seed{seed}"

# --- 1. 各種データのロード ---
base_data = np.load(f"{base_prefix}.npz")
gauss_data = np.load(f"{gauss_prefix}.npz")

base_ent = np.load(f"{base_prefix}_entropy.npz")
gauss_ent = np.load(f"{gauss_prefix}_entropy.npz")

base_loss = np.load(f"{base_prefix}_loss.npz")
gauss_loss = np.load(f"{gauss_prefix}_loss.npz")

# --- 2. グラフ描画のセットアップ (2行2列) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- 左上: エピソード報酬（リターン） ---
axes[0, 0].plot(base_data["timesteps"], base_data["returns"], label="SAC Baseline", color="tab:blue", alpha=0.8)
axes[0, 0].plot(gauss_data["timesteps"], gauss_data["returns"], label="Gaussian Prior (rho=1.0)", color="tab:orange", alpha=0.8)
axes[0, 0].set_xlabel("Timesteps")
axes[0, 0].set_ylabel("Mean Episode Return")
axes[0, 0].set_title("Episode Return Comparison")
axes[0, 0].grid(True)
axes[0, 0].legend()

# --- 右上: 方策エントロピー ---
axes[0, 1].plot(base_ent["timesteps"], base_ent["entropy"], label="SAC Baseline", color="tab:blue", alpha=0.8)
axes[0, 1].plot(gauss_ent["timesteps"], gauss_ent["entropy"], label="Gaussian Prior (rho=1.0)", color="tab:orange", alpha=0.8)
axes[0, 1].set_xlabel("Timesteps")
axes[0, 1].set_ylabel("Policy Entropy")
axes[0, 1].set_title("Policy Entropy Comparison")
axes[0, 1].grid(True)
axes[0, 1].legend()

# --- 左下: アクターロス ---
axes[1, 0].plot(base_loss["timesteps"], base_loss["actor_loss"], label="SAC Baseline", color="tab:blue", alpha=0.8)
axes[1, 0].plot(gauss_loss["timesteps"], gauss_loss["actor_loss"], label="Gaussian Prior (rho=1.0)", color="tab:orange", alpha=0.8)
axes[1, 0].set_xlabel("Timesteps")
axes[1, 0].set_ylabel("Actor Loss")
axes[1, 0].set_title("Actor Loss Comparison")
axes[1, 0].grid(True)
axes[1, 0].legend()

# --- 右下: クリティックロス ---
axes[1, 1].plot(base_loss["timesteps"], base_loss["critic_loss"], label="SAC Baseline", color="tab:blue", alpha=0.8)
axes[1, 1].plot(gauss_loss["timesteps"], gauss_loss["critic_loss"], label="Gaussian Prior (rho=1.0)", color="tab:orange", alpha=0.8)
axes[1, 1].set_xlabel("Timesteps")
axes[1, 1].set_ylabel("Critic Loss")
axes[1, 1].set_title("Critic Loss Comparison")
axes[1, 1].grid(True)
axes[1, 1].legend()

plt.tight_layout()

# --- 3. 画像として保存 ---
output_path = f"{SAVE_DIR}/humanoid_sac_vs_gaussian_rho1.0_all_metrics.png"
plt.savefig(output_path, dpi=300)
print(f"📊 全指標の比較グラフの保存が完了しました: {output_path}")

plt.show()
