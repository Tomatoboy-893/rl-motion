import os
import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "./npz_logs_humanoid"
SCALES = [0.5, 1.0, 5.0]
NUM_SEEDS = 5

def load_adr_loss_data(scale, loss_key="actor_loss"):
    """指定したスケールの複数シードのロスデータをロードする関数"""
    timesteps_list = []
    data_list = []
    
    for i in range(NUM_SEEDS):
        filename = f"{SAVE_DIR}/gaussian_scale{scale}_seed{i}_loss.npz"
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
    ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.15)

def main():
    # 左右に並べた2つのグラフ（Actor Loss / Critic Loss）を作成
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 見やすいようにカラーマップから色を割り当て
    colors = plt.cm.viridis(np.linspace(0, 1, len(SCALES)))

    for scale, color in zip(SCALES, colors):
        label_name = f"Scale = {scale}"

        # --- 1. Actor Loss のプロット ---
        t_list, actor_losses = load_adr_loss_data(scale, "actor_loss")
        plot_mean_std(axes[0], t_list, actor_losses, label_name, color)

        # --- 2. Critic Loss のプロット ---
        t_list, critic_losses = load_adr_loss_data(scale, "critic_loss")
        plot_mean_std(axes[1], t_list, critic_losses, label_name, color)

    # --- グラフの装飾 ---
    axes[0].set_title("ADR Actor Loss Comparison")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Actor Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].set_title("ADR Critic Loss Comparison")
    axes[1].set_xlabel("Timesteps")
    axes[1].set_ylabel("Critic Loss")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    output_path = f"{SAVE_DIR}/adr_loss_summary.png"
    plt.savefig(output_path, dpi=300)
    print(f"📊 ADR版ロスの比較グラフ保存が完了しました: {output_path}")

if __name__ == "__main__":
    main()
