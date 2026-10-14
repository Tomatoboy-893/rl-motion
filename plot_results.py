# plot_results.py
import os
import numpy as np
import matplotlib.pyplot as plt

# 保存先ディレクトリ
SAVE_DIR = "./npz_logs_moving_average_prior"

def main():
    # 1. 5シード統合データの読み込み
    data_path = os.path.join(SAVE_DIR, "adr_moving_avg_5seed_mean_std.npz")
    if not os.path.exists(data_path):
        print(f"[Error] File not found: {data_path}")
        return

    data = np.load(data_path)
    t = data["timesteps"]
    mean = data["mean"]
    std = data["std"]
    all_returns = data["all_returns"]  # 形: (5, 600) など

    print(f"Loaded data successfully from {data_path}")
    print(f"Timesteps shape: {t.shape}, Mean shape: {mean.shape}")

    # 2. 学習曲線（Mean Episodic Return ± std）の描画
    plt.figure(figsize=(8, 5), dpi=300)
    
    # 5シードそれぞれの薄い線を描画したい場合（お好みでコメントアウト解除）
    # for i, returns in enumerate(all_returns):
    #     plt.plot(t, returns, color="tab:purple", alpha=0.2, linewidth=1)

    # 平均 ＋ 標準偏差バンド
    plt.plot(t, mean, label="ADR (Moving Avg Prior) - Mean", color="tab:purple", linewidth=2)
    plt.fill_between(
        t, 
        mean - std, 
        mean + std, 
        color="tab:purple", 
        alpha=0.2, 
        label="± 1 std"
    )

    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Mean Episodic Return", fontsize=12)
    plt.title("HalfCheetah-v5: Learning Curve (5 Seeds)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, loc="lower right")
    plt.tight_layout()

    # 画像として保存
    output_path = os.path.join(SAVE_DIR, "custom_learning_curve.png")
    plt.savefig(output_path)
    plt.show()
    print(f"[Saved] Custom plot to {output_path}")

if __name__ == "__main__":
    main()
