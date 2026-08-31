import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

SAVE_DIR = "./npz_logs_humanoid"
SCALES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
NUM_RUNS = 5
SMOOTH_SIGMA = 5.0

plt.figure(figsize=(10, 6), dpi=300)
colors = plt.cm.tab10(np.linspace(0, 1, len(SCALES)))

def plot_mean_std(label, color, linestyle="-", is_baseline=False):
    entropy_all = []
    timesteps = None

    for run in range(NUM_RUNS):
        if is_baseline:
            path = f"{SAVE_DIR}/sac_baseline_run{run}_entropy.npz"
        else:
            path = f"{SAVE_DIR}/gaussian_scale{color}_run{run}_entropy.npz" if isinstance(color, (int, float)) else f"{SAVE_DIR}/gaussian_scale{label.split('=')[-1]}_run{run}_entropy.npz"

        if not os.path.exists(path):
            continue

        data = np.load(path)
        keys = data.files
        e_key = next((k for k in ["entropy", "policy_entropy", "arr_0"] if k in keys), keys[0])
        t_key = next((k for k in ["timesteps", "eval_timesteps", "arr_1"] if k in keys), keys[1] if len(keys) > 1 else None)

        entropy_all.append(data[e_key])
        if timesteps is None and t_key is not None:
            timesteps = data[t_key]

    if len(entropy_all) == 0:
        print(f"⚠️ データが見つかりませんでした: {label}")
        return

    min_len = min(len(e) for e in entropy_all)
    entropy_all = np.array([e[:min_len] for e in entropy_all])
    
    if timesteps is not None:
        timesteps = timesteps[:min_len]
        if timesteps[0] > 0:
            timesteps = timesteps - timesteps[0]
    else:
        timesteps = np.arange(min_len) * 5000

    mean = np.nanmean(entropy_all, axis=0)
    std = np.nanstd(entropy_all, axis=0)

    mean_smooth = gaussian_filter1d(mean, sigma=SMOOTH_SIGMA)
    std_smooth = gaussian_filter1d(std, sigma=SMOOTH_SIGMA)

    plot_color = "black" if is_baseline else color

    plt.plot(
        timesteps,
        mean_smooth,
        linewidth=2.5 if is_baseline else 2,
        color=plot_color,
        linestyle=linestyle,
        label=label,
        zorder=10 if is_baseline else 3
    )
    plt.fill_between(
        timesteps,
        mean_smooth - std_smooth,
        mean_smooth + std_smooth,
        color=plot_color,
        alpha=0.15 if is_baseline else 0.1
    )

# 1. ADR (各スケール) のプロット
for idx, scale in enumerate(SCALES):
    plot_mean_std(label=f"ADR (rho={scale})", color=colors[idx], linestyle="-", is_baseline=False)

# 2. 標準SAC (ベースライン) のプロット
plot_mean_std(label="Standard SAC (Baseline)", color="black", linestyle="--", is_baseline=True)

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
print("🎉 比較グラフの出力が完了しました！")
print(f"保存先: {output_file}")
print("===================================")
