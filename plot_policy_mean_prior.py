import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ログが保存されているフォルダ
LOG_DIR = "./npz_logs_policy_mean_prior"
SAVE_FIG_PATH = "./policy_mean_prior_metrics.png"

def load_and_plot():
    # フォルダ内のすべての .npz ファイルを取得
    file_paths = sorted(glob.glob(os.path.join(LOG_DIR, "policy_mean_prior_seed*.npz")))
    
    if not file_paths:
        print(f"エラー: {LOG_DIR} 内に .npz ファイルが見つかりませんでした。")
        return

    print(f"読み込み対象ファイル ({len(file_paths)}件):")
    for fp in file_paths:
        print(f" - {fp}")

    returns_list = []
    kl_list = []
    entropy_list = []
    beta_list = []
    alpha_list = []
    prior_mean_list = []
    
    timesteps = None

    for fp in file_paths:
        data = np.load(fp)
        if timesteps is None:
            timesteps = data["timesteps"]
        
        returns_list.append(data["returns"])
        kl_list.append(data["kl_values"])
        entropy_list.append(data["pi_entropies"])
        beta_list.append(data["beta_logs"])
        alpha_list.append(data["alpha_logs"])
        
        # prior_mean_logsが2次元（例: 次元ごとの値）の場合は絶対値の平均などを取る
        prior_mean = data["prior_mean_logs"]
        if prior_mean.ndim > 1:
            prior_mean = np.mean(np.abs(prior_mean), axis=-1)
        prior_mean_list.append(prior_mean)

    # 配列化 (n_seeds, n_steps)
    returns_arr = np.array(returns_list)
    kl_arr = np.array(kl_list)
    entropy_arr = np.array(entropy_list)
    beta_arr = np.array(beta_list)
    alpha_arr = np.array(alpha_list)
    prior_mean_arr = np.array(prior_mean_list)

    # 描画の準備 (2行3列のマルチプロット)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Policy-Mean Prior SAC (HalfCheetah-v5)", fontsize=16)

    # 共通のプロット用補助関数
    def plot_with_std(ax, data_arr, title, ylabel, color="tab:blue"):
        # ダウンサンプル・スムージングが必要な場合の処理
        x = timesteps if len(data_arr[0]) == len(timesteps) else np.linspace(0, timesteps[-1], len(data_arr[0]))
        mean = np.mean(data_arr, axis=0)
        std = np.std(data_arr, axis=0)
        
        ax.plot(x, mean, color=color, label="Mean")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Timesteps", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)

    # 1. Episode Returns
    plot_with_std(axes[0, 0], returns_arr, "Evaluation Returns", "Reward", color="tab:blue")
    
    # 2. KL Divergence
    plot_with_std(axes[0, 1], kl_arr, "KL Divergence", "KL", color="tab:orange")
    
    # 3. Policy Entropy
    plot_with_std(axes[0, 2], entropy_arr, "Policy Entropy", "Entropy", color="tab:green")
    
    # 4. Beta (KL Weight)
    plot_with_std(axes[1, 0], beta_arr, "Beta (KL Coefficient)", "Beta", color="tab:red")
    
    # 5. Alpha (Entropy Weight)
    plot_with_std(axes[1, 1], alpha_arr, "Alpha (Entropy Coefficient)", "Alpha", color="tab:purple")
    
    # 6. Prior Mean Magnitude
    plot_with_std(axes[1, 2], prior_mean_arr, "Prior Mean Magnitude (|μ|)", "Prior Mean", color="tab:brown")

    plt.tight_layout()
    plt.savefig(SAVE_FIG_PATH, dpi=300)
    print(f"\n[成功] グラフを {SAVE_FIG_PATH} に保存しました！")

if __name__ == "__main__":
    load_and_plot()
