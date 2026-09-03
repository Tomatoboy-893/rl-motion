import os
import numpy as np
import pandas as pd

SAVE_DIR = "./npz_logs_humanoid"
SCALE_STRS = ["0.1", "0.2", "0.5", "1.0", "2.0", "5.0"]
NUM_RUNS = 5

def load_npz_data(prefix, key_type="return"):
    """npzファイルからデータ配列を読み込む"""
    all_runs = []
    for run in range(NUM_RUNS):
        filename = f"{prefix}_run{run}_entropy.npz" if key_type == "entropy" else f"{prefix}_run{run}.npz"
        path = os.path.join(SAVE_DIR, filename)
        
        if not os.path.exists(path):
            continue
            
        data = np.load(path)
        keys = data.files
        
        if key_type == "entropy":
            k = next((k for k in ["entropy", "policy_entropy", "arr_0"] if k in keys), keys[0])
        else:
            k = next((k for k in ["returns", "episode_rewards", "arr_0"] if k in keys), keys[0])
            
        all_runs.append(data[k])
        
    if not all_runs:
        return None
        
    min_len = min(len(r) for r in all_runs)
    return np.array([r[:min_len] for r in all_runs])

results = []

# 対象モデルのリスト作成 (標準SAC + ADR各スケール)
models = [("Standard SAC", "sac_baseline")]
for s in SCALE_STRS:
    models.append((f"ADR (rho={s})", f"gaussian_scale{s}"))

print("データ集計中...\n")

for label, prefix in models:
    returns_data = load_npz_data(prefix, key_type="return")
    entropy_data = load_npz_data(prefix, key_type="entropy")
    
    if returns_data is None or entropy_data is None:
        print(f"⚠️ スキップ（データ不備）: {label}")
        continue
        
    # 1. 全ステップ・全シード中での「最高リターン (Max Return)」
    max_return = np.max(returns_data)
    
    # 2. 学習最終盤（ラスト10%のステップ）における統計値
    last_10_percent_idx = int(returns_data.shape[1] * 0.9)
    
    # 最終盤の平均リターン ± 標準偏差
    final_return_mean = np.mean(returns_data[:, last_10_percent_idx:])
    final_return_std = np.std(returns_data[:, last_10_percent_idx:])
    
    # 最終盤の平均エントロピー ± 標準偏差
    final_entropy_mean = np.mean(entropy_data[:, last_10_percent_idx:])
    final_entropy_std = np.std(entropy_data[:, last_10_percent_idx:])
    
    results.append({
        "Model": label,
        "Max Return": f"{max_return:.1f}",
        "Final Return (Mean±Std)": f"{final_return_mean:.1f} ± {final_return_std:.1f}",
        "Final Entropy (Mean±Std)": f"{final_entropy_mean:.2f} ± {final_entropy_std:.2f}"
    })

# DataFrame化してターミナル表示 & CSV保存
df = pd.DataFrame(results)
print("=" * 80)
print("【Humanoid-v5 実験データ定量評価サマリー】")
print("=" * 80)
print(df.to_string(index=False))
print("=" * 80)

output_csv = os.path.join(SAVE_DIR, "humanoid_summary_metrics.csv")
df.to_csv(output_csv, index=False)
print(f"\n集計結果をCSVに保存しました: {output_csv}")
