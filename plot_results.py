
import os
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "./npz_logs_full_comparison"

# 実際のファイル名のプレフィックスとラベル・カラーの設定
# (お持ちの命名規則に合わせて適宜調整してください)
configs = [
    {"prefix": "sac", "label": "Standard SAC", "color": "tab:blue"},
    {"prefix": "adr_std0.5", "label": "ADR std=0.5", "color": "tab:orange"},
    {"prefix": "adr_std1.0", "label": "ADR std=1.0", "color": "tab:green"},
    {"prefix": "adr_std2.0", "label": "ADR std=2.0", "color": "tab:red"},
]

def plot_metric(metric_name, ylabel_str, filename_str):
    plt.figure(figsize=(10, 6))
    
    for cfg in configs:
        prefix = cfg["prefix"]
        label = cfg["label"]
        color = cfg["color"]
        
        all_data = []
        steps_ref = None
        
        for seed in range(5):
            file_path = os.path.join(LOG_DIR, f"{prefix}_seed{seed}.npz")
            if os.path.exists(file_path):
                data = np.load(file_path)
                if "timesteps" in data and metric_name in data:
                    steps_ref = data["timesteps"]
                    all_data.append(data[metric_name])
                    
        if len(all_data) > 0:
            min_len = min(len(d) for d in all_data)
            trimmed_data = np.array([d[:min_len] for d in all_data])
            steps_trimmed = steps_ref[:min_len]
            
            mean = trimmed_data.mean(axis=0)
            std = trimmed_data.std(axis=0)
            
            plt.plot(steps_trimmed, mean, label=label, color=color, linewidth=2)
            plt.fill_between(steps_trimmed, mean - std, mean + std, color=color, alpha=0.2)
        else:
            print(f"⚠️ プレフィックス '{prefix}' の {metric_name} データが見つかりませんでした。")

    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel(ylabel_str, fontsize=12)
    plt.title(f"Humanoid-v5: Full Comparison ({ylabel_str})", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    out_path = os.path.join(LOG_DIR, filename_str)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📈 グラフを保存しました: {out_path}")

# リターン、Actor Loss、Critic Loss のプロット生成
# ※ npzファイル内に actor_loss / critic_loss キーがある想定です
plot_metric("returns", "Episode Return", "full_comparison_returns.png")
plot_metric("actor_loss", "Actor Loss", "full_comparison_actor_loss.png")
plot_metric("critic_loss", "Critic Loss", "full_comparison_critic_loss.png")
