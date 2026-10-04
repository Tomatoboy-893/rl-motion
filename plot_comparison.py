import numpy as np
import matplotlib.pyplot as plt

# 半チーター用のディレクトリに変更
SAVE_DIR = "./npz_logs_halfcheetah"
seed = 0

# ファイル名の定義
base_prefix = f"{SAVE_DIR}/sac_baseline_seed{seed}"
gauss_prefix = f"{SAVE_DIR}/gaussian_rho1.0_seed{seed}"

# --- 1. データのロード ---
base_data = np.load(f"{base_prefix}.npz")
gauss_data = np.load(f"{gauss_prefix}.npz")

base_ent = np.load(f"{base_prefix}_entropy.npz")
gauss_ent = np.load(f"{gauss_prefix}_entropy.npz")

base_loss = np.load(f"{base_prefix}_loss.npz")
gauss_loss = np.load(f"{gauss_prefix}_loss.npz")


# ==========================================
# グラフ1: エピソード報酬（リターン）
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(base_data["timesteps"], base_data["returns"], label="SAC Baseline", color="tab:blue", alpha=0.8)
plt.plot(gauss_data["timesteps"], gauss_data["returns"], label="Gaussian Prior (rho=1.0)", color="tab:orange", alpha=0.8)
plt.xlabel("Timesteps", fontsize=12)
plt.ylabel("Mean Episode Return", fontsize=12)
plt.title("HalfCheetah-v5: Episode Return Comparison", fontsize=14)
plt.grid(True)
plt.legend(fontsize=11)
plt.tight_layout()

return_path = f"{SAVE_DIR}/halfcheetah_comparison_return.png"
plt.savefig(return_path, dpi=300)
plt.close()
print(f"📈 1. リターンのグラフを保存しました: {return_path}")


# ==========================================
# グラフ2: 方策エントロピー
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(base_ent["timesteps"], base_ent["entropy"], label="SAC Baseline", color="tab:blue", alpha=0.8)
plt.plot(gauss_ent["timesteps"], gauss_ent["entropy"], label="Gaussian Prior (rho=1.0)", color="tab:orange", alpha=0.8)
plt.xlabel("Timesteps", fontsize=12)
plt.ylabel("Policy Entropy", fontsize=12)
plt.title("HalfCheetah-v5: Policy Entropy Comparison", fontsize=14)
plt.grid(True)
plt.legend(fontsize=11)
plt.tight_layout()

entropy_path = f"{SAVE_DIR}/halfcheetah_comparison_entropy.png"
plt.savefig(entropy_path, dpi=300)
plt.close()
print(f"📈 2. エントロピーのグラフを保存しました: {entropy_path}")


# ==========================================
# グラフ3: ロス（Actor / Critic の比較）
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Actor Loss
axes[0].plot(base_loss["timesteps"], base_loss["actor_loss"], label="SAC Baseline", color="tab:blue", alpha=0.8)
axes[0].plot(gauss_loss["timesteps"], gauss_loss["actor_loss"], label="Gaussian Prior (rho=1.0)", color="tab:orange", alpha=0.8)
axes[0].set_xlabel("Timesteps", fontsize=12)
axes[0].set_ylabel("Actor Loss", fontsize=12)
axes[0].set_title("Actor Loss Comparison", fontsize=14)
axes[0].grid(True)
axes[0].legend(fontsize=11)

# Critic Loss
axes[1].plot(base_loss["timesteps"], base_loss["critic_loss"], label="SAC Baseline", color="tab:blue", alpha=0.8)
axes[1].plot(gauss_loss["timesteps"], gauss_loss["critic_loss"], label="Gaussian Prior (rho=1.0)", color="tab:orange", alpha=0.8)
axes[1].set_xlabel("Timesteps", fontsize=12)
axes[1].set_ylabel("Critic Loss", fontsize=12)
axes[1].set_title("Critic Loss Comparison", fontsize=14)
axes[1].grid(True)
axes[1].legend(fontsize=11)

plt.tight_layout()
loss_path = f"{SAVE_DIR}/halfcheetah_comparison_loss.png"
plt.savefig(loss_path, dpi=300)
plt.close()
print(f"📈 3. ロス（Actor/Critic）のグラフを保存しました: {loss_path}")

print("\n✨ 半チーターのすべての個別グラフの生成が完了しました！")
