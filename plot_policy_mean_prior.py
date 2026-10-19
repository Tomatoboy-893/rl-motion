import os
import glob
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "./npz_logs_policy_mean_prior"

def load_and_plot_separately():
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
        
        prior_mean = data["prior_mean_logs"]
        if prior_mean.ndim > 1:
            prior_mean = np.mean(np.abs(prior_mean), axis=-1)
        prior_mean_list.append(prior_mean)

    # 描画対象の定義 (ファイル名, データ配列, タイトル, Y軸ラベル, 色)
    metrics = {
        "eval_returns": (np.array(returns_list), "Evaluation Returns", "Reward", "tab:blue"),
        "kl_divergence": (np.array(kl_list), "KL Divergence", "KL", "tab:orange"),
        "policy_entropy": (np.array(entropy_list), "Policy Entropy", "Entropy", "tab:green"),
        "beta_coefficient": (np.array(beta_list), "Beta (KL Coefficient)", "Beta", "tab:red"),
        "alpha_coefficient": (np.array(alpha_list), "Alpha (Entropy Coefficient)", "Alpha", "tab:purple"),
        "prior_mean_magnitude": (np.array(prior_mean_list), "Prior Mean Magnitude (|μ|)", "Prior Mean", "tab:brown"),
    }

    for name, (data_arr, title, ylabel, color) in metrics.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        
        x = timesteps[:data_arr.shape[1]]
        mean = np.mean(data_arr, axis=0)
        std = np.std(data_arr, axis=0)
        
        # 点数が多すぎる場合のダウンサンプル
        if len(x) > 500:
            step = len(x) // 500
            x = x[::step]
            mean = mean[::step]
            std = std[::step]

        ax.plot(x, mean, color=color, label="Mean")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Timesteps", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        
        save_path = f"./{name}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"[保存完了] {save_path}")

if __name__ == "__main__":
    load_and_plot_separately()
