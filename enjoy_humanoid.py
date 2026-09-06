import os
import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "./npz_logs_humanoid"
SCALES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
NUM_SEEDS = 5

plt.figure(figsize=(10, 6))

for scale in SCALES:
    returns_list = []
    timesteps = None
    
    for i in range(NUM_SEEDS):
        # 命名規則のゆれ（seed{i} または run{i}）の両方に対応
        path_seed = os.path.join(SAVE_DIR, f"gaussian_scale{scale}_seed{i}.npz")
        path_run = os.path.join(SAVE_DIR, f"gaussian_scale{scale}_run{i}.npz")
        
        path = path_seed if os.path.exists(path_seed) else (path_run if os.path.exists(path_run) else None)
        
        if path and os.path.exists(path):
            data = np.load(path)
            returns_list.append(data["returns"])
            if timesteps is None:
                timesteps = data["timesteps"]
                
    if returns_list:
        # 複数シードの平均と標準偏差を計算
        returns_array = np.array(returns_list)
        mean_ret = np.mean(returns_array, axis=0)
        std_ret = np.std(returns_array, axis=0)
        
        if timesteps is not None and len(timesteps) == len(mean_ret):
            # グラフにプロット（平均線＋濃淡の分散範囲）
            line, = plt.plot(timesteps, mean_ret, label=f"Scale {scale}")
            plt.fill_between(
                timesteps, 
                mean_ret - std_ret, 
                mean_ret + std_ret, 
                color=line.get_color(), 
                alpha=0.2
            )

plt.xlabel("Timesteps", fontsize=12)
plt.ylabel("Evaluation Mean Return", fontsize=12)
plt.title("Humanoid-v5: Learning Curves Comparison by Scale", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

output_path = "humanoid_learning_curves.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"📈 学習曲線のグラフを保存しました: {output_path}")
