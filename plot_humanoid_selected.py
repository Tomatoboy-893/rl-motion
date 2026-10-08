import os
import numpy as np
import matplotlib.pyplot as plt

# ログが格納されているディレクトリ
LOG_DIR = "/home/nuida_23rd/rl-motion/npz_logs_humanoid"

# 比較したい条件の定義（ベースライン、scale=0.5、scale=1.0）
# ファイル名のプレフィックスに合わせて調整しています
configs = [
    {"key": "sac_baseline", "label": "Standard SAC", "color": "tab:blue"},
    {"key": "gaussian_scale0.5", "label": "ADR (scale=0.5)", "color": "tab:orange"},
    {"key": "gaussian_scale1.0", "label": "ADR (scale=1.0)", "color": "tab:green"},
]

plt.figure(figsize=(10, 6))

for cfg in configs:
    key = cfg["key"]
    label = cfg["label"]
    color = cfg["color"]
    
    # 5シード分のデータを集計するためのリスト
    all_returns = []
    timesteps_ref = None
    
    # 5シード（seed0〜4 または run0〜4）のファイルを探索して読み込む
    for seed in range(5):
        # ファイル命名規則のパターンに対応（seedX または runX）
        possible_filenames = [
            f"{key}_seed{seed}.npz",
            f"{key}_run{seed}.npz"
        ]
        
        file_path = None
        for fname in possible_filenames:
            full_path = os.path.join(LOG_DIR, fname)
            if os.path.exists(full_path):
                file_path = full_path
                break
                
        if file_path:
            data = np.load(file_path)
            # 格納されているキー名（例: timesteps, returns など）に合わせて取得
            # ※一般的なキー名構造を想定
            if "timesteps" in data and "returns" in data:
                t = data["timesteps"]
                r = data["returns"]
            elif "results" in data: # 代替キーのフォールバック
                t = np.arange(len(data["results"])) * 5000 # 評価間隔に応じた仮ステップ
                r = data["results"]
            else:
                # キーが異なる場合のデバッグ用
                keys = list(data.keys())
                if len(keys) >= 2:
                    t, r = data[keys[0]], data[keys[1]]
                else:
                    continue
            
            timesteps_ref = t
            all_returns.append(r)

    if len(all_returns) > 0:
        # 長さを揃えて平均と標準偏差を計算
        min_len = min(len(r) for r in all_returns)
        trimmed_returns = np.array([r[:min_len] for r in all_returns])
        t_trimmed = timesteps_ref[:min_len]
        
        mean = trimmed_returns.mean(axis=0)
        std = trimmed_returns.std(axis=0)
        
        # プロット描画
        plt.plot(t_trimmed, mean, label=label, color=color, linewidth=2)
        plt.fill_between(t_trimmed, mean - std, mean + std, color=color, alpha=0.2)
    else:
        print(f"⚠️ 条件 '{key}' に該当するシードファイルが見つかりませんでした。")

plt.xlabel("Timesteps", fontsize=12)
plt.ylabel("Mean Episodic Return", fontsize=12)
plt.title("Humanoid-v5: SAC vs ADR (scale=0.5, 1.0)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

output_path = os.path.join(LOG_DIR, "humanoid_comparison_sac_scale0.5_1.0.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"📈 比較グラフを保存しました: {output_path}")
