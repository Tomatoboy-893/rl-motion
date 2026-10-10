
import os
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "./npz_logs_full_comparison" # 必要に応じてパスを絶対パスに変更してください

# 比較する設定のリスト（必要に応じてキーやラベルを追加・調整してください）
configs = [
    {"key": "sac_baseline", "label": "Standard SAC", "color": "tab:blue"},
    {"key": "adr_scale0.5", "label": "ADR scale=0.5", "color": "tab:orange"},
    {"key": "adr_scale1.0", "label": "ADR scale=1.0", "color": "tab:green"},
    {"key": "adr_std2.0", "label": "ADR std=2.0", "color": "tab:red"},
]

def plot_metric(metric_name, ylabel_str, filename_str):
    plt.figure(figsize=(10, 6))
    
    for cfg in configs:
        key = cfg["key"]
        label = cfg["label"]
        color = cfg["color"]
        
        all_data = []
        steps_ref = None
        
        for seed in range(5):
            # ファイル名のパターンに合わせて調整
            possible_filenames = [
                f"{key}_seed{seed}_{metric_name}.npz",
                f"{key}_run{seed}_{metric_name}.npz",
                f"{key}_{metric_name}_seed{seed}.npz",
                f"{key}_seed{seed}.npz"
            ]
            
            file_path = None
            for fname in possible_filenames:
                full_path = os.path.join(LOG_DIR, fname)
                if os.path.exists(full_path):
                    file_path = full_path
                    break
                    
            if file_path:
                data = np.load(file_path)
                # キーの自動探索
                keys = list(data.keys())
                if "timesteps" in data:
                    steps = data["timesteps"]
                    # メトリック名に一致するキーを探す
                    target_key = None
                    for k in keys:
                        if metric_name in k.lower():
                            target_key = k
                            break
                    if target_key is None and len(keys) >= 2:
                        target_key = [k for k in keys if k != "timesteps"][0]
                    
                    if target_key:
                        steps_ref = steps
                        all_data.append(data[target_key])

        if len(all_data) > 0:
            min_len = min(len(d) for d in all_data)
            trimmed_data = np.array([d[:min_len] for d in all_data])
            steps_trimmed = steps_ref[:min_len]
            
            mean = trimmed_data.mean(axis=0)
            std = trimmed_data.std(axis=0)
            
            plt.plot(steps_trimmed, mean, label=label, color=color, linewidth=2)
            plt.fill_between(steps_trimmed, mean - std, mean + std, color=color, alpha=0.2)
        else:
            print(f"⚠️ 条件 '{key}' の {metric_name} データが見つかりませんでした。")

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

# リターン、Actor Loss、Critic Loss を一括生成
plot_metric("return", "Episode Return", "full_comparison_returns.png")
plot_metric("actor_loss", "Actor Loss", "full_comparison_actor_loss.png")
plot_metric("critic_loss", "Critic Loss", "full_comparison_critic_loss.png")
