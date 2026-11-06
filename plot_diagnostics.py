import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_internal_diagnostics(base_dir=".", save_filename="comparison_results/diagnostics_comparison.png", max_points=2000):
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    fig.suptitle("Humanoid-v5: Training Diagnostics & Dynamic Parameters", fontsize=14, y=0.98)

    algos = ["policy_mean_prior", "pure_sac"]
    colors = {"policy_mean_prior": "#1f77b4", "pure_sac": "#ff7f0e"}
    labels = {"policy_mean_prior": "Policy Mean Prior (Ours)", "pure_sac": "Vanilla SAC"}

    def load_metric(algo, key):
        dir_path = os.path.join(base_dir, f"npz_logs_{algo}_bs64")
        files = glob.glob(os.path.join(dir_path, f"{algo}_bs64_seed*.npz"))
        data_list = []
        min_len = float("inf")
        
        for f in sorted(files):
            d = np.load(f)
            if key in d and d[key].size > 0:
                val = np.array(d[key]).flatten()
                data_list.append(val)
                min_len = min(min_len, len(val))
                
        if not data_list or min_len == float("inf"):
            return None, None, None
            
        arr = np.array([v[:min_len] for v in data_list])
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        x = np.arange(len(mean))

        # 描画点数が多すぎる場合にダウンサンプリング
        if len(mean) > max_points:
            stride = len(mean) // max_points
            mean = mean[::stride]
            std = std[::stride]
            x = x[::stride]

        return x, mean, std

    # 1. Policy Entropy
    ax = axs[0, 0]
    for algo in algos:
        x, mean, std = load_metric(algo, "pi_entropies")
        if mean is not None:
            ax.plot(x, mean, label=labels[algo], color=colors[algo], alpha=0.85, linewidth=1.2)
            ax.fill_between(x, mean - std, mean + std, color=colors[algo], alpha=0.15)
    ax.set_title("Policy Entropy ($H(\\pi)$)")
    ax.set_xlabel("Update Steps")
    ax.set_ylabel("Entropy")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 2. Log Alpha (Temperature)
    ax = axs[0, 1]
    for algo in algos:
        x, mean, std = load_metric(algo, "alpha_logs")
        if mean is not None:
            ax.plot(x, mean, label=labels[algo], color=colors[algo], alpha=0.85, linewidth=1.2)
            ax.fill_between(x, mean - std, mean + std, color=colors[algo], alpha=0.15)
    ax.set_title("Temperature Parameter (Log $\\alpha$)")
    ax.set_xlabel("Update Steps")
    ax.set_ylabel("Log $\\alpha$")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 3. KL Divergence (Proposed Only)
    ax = axs[1, 0]
    x, mean_kl, std_kl = load_metric("policy_mean_prior", "kl_values")
    if mean_kl is not None:
        ax.plot(x, mean_kl, label="KL( $\\pi$ || Prior )", color="#2ca02c", alpha=0.85, linewidth=1.2)
        ax.fill_between(x, mean_kl - std_kl, mean_kl + std_kl, color="#2ca02c", alpha=0.15)
        ax.set_title("KL Divergence from Prior")
        ax.set_xlabel("Update Steps")
        ax.set_ylabel("KL Value")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No KL Data Saved", ha='center', va='center')

    # 4. Log Beta (Proposed Only)
    ax = axs[1, 1]
    x, mean_beta, std_beta = load_metric("policy_mean_prior", "beta_logs")
    if mean_beta is not None:
        ax.plot(x, mean_beta, label="Log $\\beta$ (KL Coeff)", color="#9467bd", alpha=0.85, linewidth=1.2)
        ax.fill_between(x, mean_beta - std_beta, mean_beta + std_beta, color="#9467bd", alpha=0.15)
        ax.set_title("Adaptive KL Weight (Log $\\beta$)")
        ax.set_xlabel("Update Steps")
        ax.set_ylabel("Log $\\beta$")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No Beta Data Saved", ha='center', va='center')

    plt.tight_layout()
    plt.savefig(save_filename, bbox_inches="tight")
    print(f"Saved diagnostics plot to {save_filename}")

if __name__ == "__main__":
    plot_internal_diagnostics()
