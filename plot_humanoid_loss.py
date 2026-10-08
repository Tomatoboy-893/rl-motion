import os
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "/home/nuida_23rd/rl-motion/npz_logs_humanoid"

# 比較したい条件の定義（ベースライン、scale=0.5、scale=1.0 のロスデータ）
configs = [
    {"key": "sac_baseline", "label": "Standard SAC (Loss)", "color": "tab:blue"},
    {"key": "gaussian_scale0.5", "label": "ADR scale=0.5 (Loss)", "color": "tab:orange"},
    {"key": "gaussian_scale1.0", "label": "ADR scale=1.0 (Loss)", "color": "tab:green"},
]

plt.figure(figsize=(10, 6))

for cfg in configs:
    key = cfg["key"]
    label = cfg["label"]
    color = cfg["color"]
    
    all_losses = []
    steps_ref = None
    
    # 5シード分（0〜4）のロスデータを探索してロード
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
            # キー構造に合わせてステップとロスを取得
            keys = list(data.keys())
            if "timesteps" in data and "loss" in data:
                steps = data["timesteps"]
                loss = data["loss"]
            elif len(keys) >= 2:
                steps, loss = data[keys[0]], data[keys[1]]
            else:
                continue
            
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
        print(f"⚠️ 条件 '{key}' に該当するロスファイルが見つかりませんでした。")

plt.xlabel("Timesteps", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Humanoid-v5: Loss Comparison (SAC vs ADR scale=0.5, 1.0)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

output_path = os.path.join(LOG_DIR, "humanoid_loss_comparison.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"📈 ロスの比較グラフを保存しました: {output_path}")
