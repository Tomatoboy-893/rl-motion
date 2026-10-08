import os
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "/home/nuida_23rd/rl-motion/npz_logs_humanoid"

configs = [
    {"key": "sac_baseline", "label": "Standard SAC", "color": "tab:blue"},
    {"key": "gaussian_scale0.5", "label": "ADR scale=0.5", "color": "tab:orange"},
    {"key": "gaussian_scale1.0", "label": "ADR scale=1.0", "color": "tab:green"},
]

def plot_loss_type(loss_key_name, title_str, filename_str):
    plt.figure(figsize=(10, 6))
    
    for cfg in configs:
        key = cfg["key"]
        label = cfg["label"]
        color = cfg["color"]
        
        all_losses = []
        steps_ref = None
        
        for seed in range(5):
            possible_filenames = [
                f"{key}_seed{seed}_loss.npz",
                f"{key}_run{seed}_loss.npz",
                f"{key}_loss_seed{seed}.npz"
            ]
            
            file_path = None
            for fname in possible_filenames:
                full_path = os.path.join(LOG_DIR, fname)
                if os.path.exists(full_path):
                    file_path = full_path
                    break
                    
            if file_path:
                data = np.load(file_path)
                # ファイル内に複数のロス（actor/critic等）が格納されているキーを探す
                # キーの例: 'actor_loss', 'critic_loss', または単に 'loss' など
                found_key = None
                for k in data.keys():
                    if loss_key_name in k.lower():
                        found_key = k
                        break
                
                if found_key and "timesteps" in data:
                    steps = data["timesteps"]
                    loss = data[found_key]
                    steps_ref = steps
                    all_losses.append(loss)
                elif len(data.keys()) >= 2:
                    # フォールバック
                    keys = list(data.keys())
                    steps = data[keys[0]]
                    loss = data[keys[1]]
                    steps_ref = steps
                    all_losses.append(loss)

        if len(all_losses) > 0:
            min_len = min(len(l) for l in all_losses)
            trimmed_losses = np.array([l[:min_len] for l in all_losses])
            steps_trimmed = steps_ref[:min_len]
            
            mean = trimmed_losses.mean(axis=0)
            std = trimmed_losses.std(axis=0)
            
            plt.plot(steps_trimmed, mean, label=label, color=color, linewidth=2)
            plt.fill_between(steps_trimmed, mean - std, mean + std, color=color, alpha=0.2)
        else:
            print(f"⚠️ 条件 '{key}' の {loss_key_name} データが見つかりませんでした。")

    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel(title_str, fontsize=12)
    plt.title(f"Humanoid-v5: {title_str} Comparison", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    out_path = os.path.join(LOG_DIR, filename_str)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📈 グラフを保存しました: {out_path}")

# Actor Loss と Critic Loss をそれぞれ生成
plot_loss_type("actor", "Actor Loss", "humanoid_actor_loss_comparison.png")
plot_loss_type("critic", "Critic Loss", "humanoid_critic_loss_comparison.png")
