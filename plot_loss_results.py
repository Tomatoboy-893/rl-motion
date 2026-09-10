import os
import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "./npz_logs_humanoid"
PREFIX = "sac_baseline"
NUM_SEEDS = 5

def load_data(data_type="return"):
    """複数シードのデータをロードしてパディング/集約する関数"""
    timesteps_list = []
    data_list = []
    
    for i in range(NUM_SEEDS):
        # data_type に応じて正しいファイル名を組み立てる
        if data_type == "entropy":
            filename = f"{SAVE_DIR}/{PREFIX}_run{i}_entropy.npz"
            key_name = "entropy"
        elif data_type == "return":
            filename = f"{SAVE_DIR}/{PREFIX}_run{i}.npz"
            key_name = "returns"
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
            
        if not os.path.exists(filename):
            print(f"⚠️ ファイルが見つかりません: {filename}")
            continue
            
        loaded = np.load(filename)
        timesteps_list.append(loaded["timesteps"])
        data_list.append(loaded[key_name])
        
    return timesteps_list, data_list

def load_loss_data(loss_key="actor_loss"):
    """Actor Loss と Critic Loss を個別に取得するための専用関数"""
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
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

    # --- 3. Entropy のプロット ---
    t_entropy, entropies_list = load_data("entropy") # ここを "entropy" に変更
    plot_mean_std(axes[2], t_entropy, entropies_list, "Entropy", "green")
    axes[2].set_title("Policy Entropy")
    axes[2].set_xlabel("Timesteps")
    axes[2].set_ylabel("Entropy")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    # --- 4. Returns (報酬) のプロット ---
    t_returns, returns_list = load_data("return") # ここを "return" に変更
    plot_mean_std(axes[3], t_returns, returns_list, "Return", "red")
    axes[3].set_title("Episode Return")
    axes[3].set_xlabel("Timesteps")
    axes[3].set_ylabel("Return")
    axes[3].grid(True, linestyle="--", alpha=0.5)
    axes[3].legend()

    plt.tight_layout()
    output_path = f"{SAVE_DIR}/sac_baseline_summary.png"
    plt.savefig(output_path, dpi=300)
    print(f"📊 グラフの保存が完了しました: {output_path}")

if __name__ == "__main__":
    main()
