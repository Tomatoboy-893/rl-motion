# plot_comparison.py

import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "./npz_logs_humanoid"
seed = 0

# ファイル名の定義
base_prefix = f"{SAVE_DIR}/sac_baseline_seed{seed}"
gauss_prefix = f"{SAVE_DIR}/gaussian_rho1.0_seed{seed}"

# --- 1. データのロード ---
# リターンデータ
base_data = np.load(f"{base_prefix}.npz")
gauss_data = np.load(f"{gauss_prefix}.npz")

# エントロピーデータ
base_ent = np.load(f"{base_prefix}_entropy.npz")
gauss_ent = np.load(f"{gauss_prefix}_entropy.npz")

# --- 2. グラフ描画のセットアップ (1行2列) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- 左側: リターン（報酬）の比較 ---
axes[0].plot(base_data["timesteps"], base_data["returns"], label="SAC Baseline", color="tab:blue", alpha=0.8)
axes[0].plot(gauss_data["timesteps"], gauss_data["returns"], label="Gaussian Prior (rho=1.0)", color="tab:orange", alpha=0.8)
axes[0].set_xlabel("Timesteps")
axes[0].set_ylabel("Mean Episode Return")
axes[0].set_title("Humanoid-v5: Episode Return Comparison")
axes[0].grid(True)
axes[0].legend()

# --- 右側: エントロピー（方策の多様性）の比較 ---
axes[1].plot(base_ent["timesteps"], base_ent["entropy"], label="SAC Baseline", color="tab:blue", alpha=0.8)
axes[1].plot(gauss_ent["timesteps"], gauss_ent["entropy"], label="Gaussian Prior (rho=1.0)", color="tab:orange", alpha=0.8)
axes[1].set_xlabel("Timesteps")
axes[1].set_ylabel("Policy Entropy")
axes[1].set_title("Humanoid-v5: Policy Entropy Comparison")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()

# --- 3. 画像として保存 ---
output_path = f"{SAVE_DIR}/comparison_return_entropy.png"
plt.savefig(output_path, dpi=300)
print(f"📊 比較グラフの保存が完了しました: {output_path}")

plt.show()
