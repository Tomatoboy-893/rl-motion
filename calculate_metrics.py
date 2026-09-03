import os
import numpy as np

SAVE_DIR = "./npz_logs_humanoid"
PREFIX = "sac_baseline"
NUM_RUNS = 5

def load_data(key_type="return"):
    all_runs = []
    for run in range(NUM_RUNS):
        filename = f"{PREFIX}_run{run}_entropy.npz" if key_type == "entropy" else f"{PREFIX}_run{run}.npz"
        path = os.path.join(SAVE_DIR, filename)
        
        if not os.path.exists(path):
            print(f"⚠️ ファイルが見つかりません: {path}")
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

returns = load_data("return")
entropy = load_data("entropy")

if returns is not None and entropy is not None:
    # 1. 最高リターン (全シード・全ステップ中の最大値)
    max_return = np.max(returns)
    
    # 2. 最終盤（ラスト10%ステップ）のインデックス
    last_10_percent = int(returns.shape[1] * 0.9)
    
    # 最終盤の平均リターンと標準偏差
    final_return_mean = np.mean(returns[:, last_10_percent:])
    final_return_std = np.std(returns[:, last_10_percent:])
    
    # 最終盤の平均エントロピーと標準偏差
    final_entropy_mean = np.mean(entropy[:, last_10_percent:])
    final_entropy_std = np.std(entropy[:, last_10_percent:])
    
    print("==========================================")
    print("  Standard SAC (Baseline) 定量評価結果  ")
    print("==========================================")
    print(f"■ 最高リターン (Max Return): {max_return:.2f}")
    print(f"■ 収束期平均リターン (Final Return): {final_return_mean:.2f} ± {final_return_std:.2f}")
    print(f"■ 収束期平均エントロピー (Final Entropy): {final_entropy_mean:.2f} ± {final_entropy_std:.2f}")
    print("==========================================")
else:
    print("❌ データの読み込みに失敗しました。")
