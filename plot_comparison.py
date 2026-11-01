import os
import glob
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "./npz_logs_pure_sac_bs256"
SAVE_PATH = "./pure_sac_all_metrics.png"

files = sorted(glob.glob(os.path.join(LOG_DIR, "*.npz")))

if not files:
    print(f"❌ '{LOG_DIR}' に .npz ファイルが見つかりません。")
    exit()

# 1. データの読み込みとキーの動的検出
sample_npz = np.load(files[0])
all_keys = list(sample_npz.keys())

# X軸（timesteps）と可視化対象の指標キーを分離
if "timesteps" in all_keys:
    x_key = "timesteps"
elif "t" in all_keys:
    x_key = "t"
else:
    x_key = None

metric_keys = [k for k in all_keys if k != x_key and np.issubdtype(sample_npz[k].dtype, np.number)]

print(f"🔍 検出された指標: {metric_keys}")

# 全シードのデータをロード
data_by_metric = {k: [] for k in metric_keys}
timesteps = sample_npz[x_key] if x_key else None

for f in files:
    data = np.load(f)
    for k in metric_keys:
        if k in data:
            data_by_metric[k].append(data[k])

# データ長の揃え込み処理
min_lens = {}
for k in metric_keys:
    if data_by_metric[k]:
        min_lens[k] = min(len(arr) for arr in data_by_metric[k])
        data_by_metric[k] = np.array([arr[:min_lens[k]] for arr in data_by_metric[k]])

# 2. グラフ描画（指標数に応じたサブプロット構成）
num_metrics = len(metric_keys)
fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 3.5 * num_metrics), sharex=True)

if num_metrics == 1:
    axes = [axes]

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

for idx, key in enumerate(metric_keys):
    ax = axes[idx]
    arr = data_by_metric[key]  # Shape: (Num_Seeds, Steps)
    color = colors[idx % len(colors)]
    
    # X軸の設定
    if x_key and timesteps is not None:
        x_axis = timesteps[:arr.shape[1]]
    else:
        x_axis = np.arange(arr.shape[1])

    # 各シードの個別線（薄く表示）
    for seed_idx in range(arr.shape[0]):
        ax.plot(x_axis, arr[seed_idx], color=color, alpha=0.25, linewidth=1)

    # 平均値と標準偏差（影）
    mean_val = np.nanmean(arr, axis=0)
    std_val = np.nanstd(arr, axis=0)

    ax.plot(x_axis, mean_val, color=color, linewidth=2.5, label=f"Mean {key}")
    ax.fill_between(x_axis, mean_val - std_val, mean_val + std_val, color=color, alpha=0.15)

    ax.set_title(f"Vanilla SAC: {key}", fontsize=12, fontweight="bold")
    ax.set_ylabel(key, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)

axes[-1].set_xlabel("Timesteps" if x_key else "Steps / Iterations", fontsize=11)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300)
print(f"✅ 全指標の可視化グラフを '{SAVE_PATH}' に保存しました。")
