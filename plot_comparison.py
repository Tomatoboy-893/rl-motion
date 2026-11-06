import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def plot_evaluation_returns(
    base_dir=".",
    algos=["policy_mean_prior", "pure_sac"],
    labels=["Policy Mean Prior (Proposed)", "Pure SAC"],
    colors=["#d62728", "#1f77b4"],
    save_filename="humanoid_v5_returns.png",
):
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    for algo, label, color in zip(algos, labels, colors):
        dir_path = os.path.join(base_dir, f"npz_logs_{algo}_bs64")
        files = glob.glob(os.path.join(dir_path, f"{algo}_bs64_seed*.npz"))

        if not files:
            print(f"Warning: {dir_path} にファイルが見つかりません。")
            continue

        returns_list = []
        min_len = float("inf")
        common_timesteps = None

        # シードごとのデータを読み込み
        for f in sorted(files):
            data = np.load(f)
            t = data["timesteps"]
            r = data["returns"]

            returns_list.append(r)
            if len(r) < min_len:
                min_len = len(r)
                common_timesteps = t[:min_len]

        # 共通の長さ（ステップ数）に揃えて配列化
        returns_arr = np.array([r[:min_len] for r in returns_list])

        # 平均と標準偏差の計算
        mean_returns = np.mean(returns_arr, axis=0)
        std_returns = np.std(returns_arr, axis=0)

        # 描画
        plt.plot(
            common_timesteps,
            mean_returns,
            label=f"{label} (n={len(returns_list)})",
            color=color,
            linewidth=2.0,
        )
        plt.fill_between(
            common_timesteps,
            mean_returns - std_returns,
            mean_returns + std_returns,
            color=color,
            alpha=0.18,
        )

    # グラフ装飾
    plt.xlabel("Timesteps", fontsize=12, labelpad=8)
    plt.ylabel("Evaluation Mean Return", fontsize=12, labelpad=8)
    plt.title(
        "Humanoid-v5: Evaluation Performance Comparison (Batch Size = 64)",
        fontsize=13,
        pad=12,
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc="lower right", framealpha=0.9)

    # X軸を「M（Million）」表記にフォーマット
    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x*1e-6:.1f}M" if x > 0 else "0")
    )

    plt.tight_layout()
    plt.savefig(save_filename, bbox_inches="tight")
    print(f"Saved return plot to {save_filename}")
    plt.show()


if __name__ == "__main__":
    plot_evaluation_returns()
