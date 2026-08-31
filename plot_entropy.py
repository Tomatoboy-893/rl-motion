import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

SAVE_DIR = "./npz_logs_humanoid"
SCALE_STRS = ["0.1", "0.2", "0.5", "1.0", "2.0", "5.0"]
NUM_RUNS = 5
SMOOTH_SIGMA = 5.0
TOTAL_STEPS = 3_000_000  # 300万ステップに固定

plt.figure(figsize=(10, 6), dpi=300)
colors = plt.cm.tab10(np.linspace(0, 1, len(SCALE_STRS)))

def load_entropy_data(file_prefix):
    entropy_all = []

    for run in range(NUM_RUNS):
        path = f"{SAVE_DIR}/{file_prefix}_run{run}_entropy.npz"

        if not os.path.exists(path):
            continue

        data = np.load(path)
        keys = data.files
        e_key = next((k for k in ["entropy", "policy_entropy", "arr_0"] if k in keys), keys[0])
        entropy_all.append(data[e_key])

    if len(entropy_all) == 0:
        return None, None

    min_len = min(len(e) for e in entropy_all)
    entropy_all = np.array([e[:min_len] for e in entropy_all])
    
    # 横軸を 0 ~ 3,000,000 ステップに統一補正
    timesteps = np.linspace(0, TOTAL_STEPS, min_len)

    mean = np.nanmean(entropy_all, axis=0)
    std = np.nanstd(entropy_all, axis=0)

    mean_smooth = gaussian_filter1d(mean, sigma=SMOOTH_SIGMA)
    std_smooth = gaussian_filter1d(std, sigma=SMOOTH_SIGMA)

    return timesteps, (mean_smooth, std_smooth)

# 1. ADR (各スケール) のプロット
for idx, scale_str in enumerate(SCALE_STRS):
    prefix = f"gaussian_scale{scale_str}"
    timesteps, data = load_entropy_data(prefix)
    
    if timesteps is None:
        print(f"⚠️ データが見つかりませんでした: ADR (rho={scale_str})")
        continue
        
    mean_smooth, std_smooth = data
    plt.plot(
        timesteps,
        mean_smooth,
        linewidth=2,
        color=colors[idx],
        linestyle="-",
        label=f"ADR (rho={scale_str})"
    )
    plt.fill_between(
        timesteps,
        mean_smooth - std_smooth,
        mean_smooth + std_smooth,
        color=colors[idx],
        alpha=0.1
    )

# 2. 標準SAC (ベースライン) のプロット
timesteps_base, data_base = load_entropy_data("sac_baseline")
if timesteps_base is not None:
    mean_base, std_base = data_base
    plt.plot(
        timesteps_base,
        mean_base,
        linewidth=2.5,
        color="black",
        linestyle="--",
        label="Standard SAC (Baseline)",
        zorder=10
    )
    plt.fill_between(
        timesteps_base,
        mean_base - std_base,
        mean_base + std_base,
        color="black",
        alpha=0.15
    )
else:
    print("⚠️ ベースラインデータが見つかりませんでした。")

plt.xlabel("Timesteps", fontsize=13)
plt.ylabel("Policy Entropy", fontsize=13)
plt.title("Humanoid-v5 Policy Entropy Comparison", fontsize=15)

plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x > 0 else '0'))

plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(loc="upper right", fontsize=10)
plt.tight_layout()

output_file = f"{SAVE_DIR}/humanoid_entropy_comparison.png"
plt.savefig(output_file, dpi=300)
plt.close()

print("===================================")
print("🎉 比較グラフの再出力が完了しました！")
print(f"保存先: {output_file}")
print("===================================")
