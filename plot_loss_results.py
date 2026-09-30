import os
import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "./npz_logs_humanoid"
# 比較したいスケール（必要に応じてベースラインなども追加可能）
SCALES = [0.5, 1.0, 5.0]
NUM_SEEDS = 5

def load_loss_data(prefix_base, loss_key="actor_loss"):
    """指定したプレフィックス群（複数シード）のロスデータをロードする関数"""
    timesteps_list = []
    data_list = []
    
    for i in range(NUM_SEEDS):
        # 命名規則の候補（seed表記、run表記の両方に対応）
        candidates = [
            f"{SAVE_DIR}/{prefix_base}_seed{i}_loss.npz",
            f"{SAVE_DIR}/{prefix_base}_run{i}_loss.npz"
        ]
        
        filename = None
        for cand in candidates:
            if os.path.exists(cand):
                filename = cand
                break
                
        if filename is None:
            continue
            
        loaded = np.load(filename)
        if loss_key in loaded and "timesteps" in loaded:
            timesteps_list.append(loaded["timesteps"])
            data_list.append(loaded[loss_key])
        
    return timesteps_list, data_list

def plot_mean_std(ax, timesteps_list, data_list, label, color):
    """複数シードの平均と標準偏差（帯）をプロットするヘルパー関数"""
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
    
    # カラーマップの設定
    colors = plt.cm.viridis(np.linspace(0, 1, len(SCALES) + 1))

    # --- 1. 各スケール（ADR）のプロット ---
    for idx, scale in enumerate(SCALES):
        label_name = f"Scale = {scale}"
        prefix_base = f"gaussian_scale{scale}"
        color = colors[idx]

        # Actor Loss
        t_list, actor_losses = load_loss_data(prefix_base, "actor_loss")
        plot_mean_std(axes[0], t_list, actor_losses, label_name, color)

        # Critic Loss
        t_list, critic_losses = load_loss_data(prefix_base, "critic_loss")
        plot_mean_std(axes[1], t_list, critic_losses, label_name, color)

    # --- 2. （お好みで）ベースラインの追加 ---
    # もし sac_baseline のロス比較も入れたい場合はコメントアウトを外してください
    """
    base_t, base_actor = load_loss_data("sac_baseline", "actor_loss")
    plot_mean_std(axes[0], base_t, base_actor, "Baseline", "red")
    
    base_t, base_critic = load_loss_data("sac_baseline", "critic_loss")
    plot_mean_std(axes[1], base_t, base_critic, "Baseline", "red")
    """

    # --- グラフの装飾 ---
    axes[0].set_title("Actor Loss Comparison", fontsize=14)
    axes[0].set_xlabel("Timesteps", fontsize=12)
    axes[0].set_ylabel("Actor Loss", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=11)

    axes[1].set_title("Critic Loss Comparison", fontsize=14)
    axes[1].set_xlabel("Timesteps", fontsize=12)
    axes[1].set_ylabel("Critic Loss", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    output_path = f"{SAVE_DIR}/adr_loss_summary.png"
    plt.savefig(output_path, dpi=300)
    print(f"📊 ロスの比較グラフを保存しました: {output_path}")

if __name__ == "__main__":
    main()
