import os
import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "./npz_logs_humanoid"
PREFIX = "sac_baseline"
NUM_SEEDS = 5

def load_loss_data(loss_key="actor_loss"):
    """複数シードのロスデータをロードしてパディング/集約する関数"""
    timesteps_list = []
    data_list = []
    
    for i in range(NUM_SEEDS):
        filename = f"{SAVE_DIR}/{PREFIX}_run{i}_loss.npz"
        if not os.path.exists(filename):
            print(f"⚠️ ファイルが見つかりません: {filename}")
            continue
            
        loaded = np.load(filename)
        timesteps_list.append(loaded["timesteps"])
        data_list.append(loaded[loss_key])
        
    return timesteps_list, data_list

def plot_mean_std(ax, timesteps_list, data_list, label, color):
    """平均と標準偏差（帯）をプロットするヘルパー関数"""
    if not data_list:
        return
    
    min_len = min([len(d) for d in data_list])
    t = timesteps_list[0][:min_len]
    data_matrix = np.array([d[:min_len] for d in data_list])
    
    mean = np.nanmean(data_matrix, axis=0)
    std = np.nanstd(data_matrix, axis=0)
    
    ax.plot(t, mean, label=label, color=color, linewidth=2)
    ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.2)

def main():
    # 左右に並べた2つのグラフ（Actor Loss / Critic Loss）を作成
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- 1. Actor Loss のプロット ---
    t_list, actor_losses = load_loss_data("actor_loss")
    plot_mean_std(axes[0], t_list, actor_losses, "Actor Loss", "blue")
    axes[0].set_title("Actor Loss")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # --- 2. Critic Loss のプロット ---
    t_list, critic_losses = load_loss_data("critic_loss")
    plot_mean_std(axes[1], t_list, critic_losses, "Critic Loss", "orange")
    axes[1].set_title("Critic Loss")
    axes[1].set_xlabel("Timesteps")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    output_path = f"{SAVE_DIR}/sac_baseline_loss_summary.png"
    plt.savefig(output_path, dpi=300)
    print(f"📊 ロスのグラフ保存が完了しました: {output_path}")

if __name__ == "__main__":
    main()
