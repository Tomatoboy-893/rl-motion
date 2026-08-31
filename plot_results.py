import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# データが保存されているディレクトリ
LOG_DIR = "./npz_logs_humanoid"
SAVE_FIG_PATH = "./humanoid_learning_curve.png"

def load_and_process_runs(pattern):
    """指定されたパターンのnpzファイルを全て読み込み、平均と標準偏差を計算する"""
    files = sorted(glob.glob(os.path.join(LOG_DIR, pattern)))
    if not files:
        print(f"⚠️ 該当するファイルが見つかりません: {pattern}")
        return None, None, None

    returns_list = []
    timesteps = None

    for f in files:
        data = np.load(f)
        returns_list.append(data["returns"])
        if timesteps is None:
            timesteps = data["timesteps"]

    # (num_runs, num_evals) の2次元配列に変換
    returns_array = np.array(returns_list)
    
    mean_returns = np.mean(returns_array, axis=0)
    std_returns = np.std(returns_array, axis=0)

    return timesteps, mean_returns, std_returns

def main():
    plt.figure(figsize=(10, 6))
    plt.rcParams["font.size"] = 12

    # 1. 標準SAC（ベースライン）の描画
    ts_base, mean_base, std_base = load_and_process_runs("sac_baseline_run*.npz")
    if mean_base is not None:
        plt.plot(ts_base, mean_base, label="Standard SAC Baseline", color="black", linewidth=2)
        plt.fill_between(ts_base, mean_base - std_base, mean_base + std_base, color="black", alpha=0.15)

    # 2. ADR（各スケール）データの描画（例: gaussian_scale0.5, 1.0 など）
    # ディレクトリ内にある gaussian_scale*_run0.npz からスケール名を自動取得
    adr_files = glob.glob(os.path.join(LOG_DIR, "gaussian_scale*_run0.npz"))
    scales = sorted(list(set([os.path.basename(f).split("_run")[0] for f in adr_files])))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, max(len(scales), 1)))

    for idx, scale_prefix in enumerate(scales):
        pattern = f"{scale_prefix}_run*.npz"
        ts, mean, std = load_and_process_runs(pattern)
        if mean is not None:
            label_name = scale_prefix.replace("gaussian_scale", "ADR Scale ")
            plt.plot(ts, mean, label=label_name, color=colors[idx], linewidth=2)
            plt.fill_between(ts, mean - std, mean + std, color=colors[idx], alpha=0.15)

    # グラフのレイアウト装飾
    plt.title("Humanoid-v5 Learning Performance Comparison", fontsize=14, fontweight="bold")
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Mean Episode Return", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    # 画像保存
    plt.savefig(SAVE_FIG_PATH, dpi=300)
    print(f"🎉 グラフを保存しました: {SAVE_FIG_PATH}")

if __name__ == "__main__":
    main()
