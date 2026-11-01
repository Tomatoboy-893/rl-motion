import os
import glob
import numpy as np
import matplotlib.pyplot as plt

DIR_PRIOR = "./npz_logs_policy_mean_prior"
DIR_SAC = "./npz_logs_pure_sac_bs256"
SAVE_DIR = "./comparison_results"
os.makedirs(SAVE_DIR, exist_ok=True)

trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

def extract_clean_eval_data(timesteps, returns, eval_freq=5000):
    """
    毎ステップ重複記録されたデータから、5,000ステップごとのユニークな評価ポイントのみを抽出する
    """
    # 5000の倍数に最も近いインデックス、または値の変化点を抽出
    unique_ts = []
    unique_rets = []
    
    target_ts = eval_freq
    max_ts = np.max(timesteps) if len(timesteps) > 0 else 0
    
    while target_ts <= max_ts:
        # target_ts に最も近いタイムステップのインデックスを取得
        idx = np.argmin(np.abs(timesteps - target_ts))
        if idx < len(returns):
            unique_ts.append(timesteps[idx])
            unique_rets.append(returns[idx])
        target_ts += eval_freq
        
    return np.array(unique_ts), np.array(unique_rets)

def load_dataset():
    # 1. Vanilla SAC (正常に600点保存されている)
    sac_files = sorted(glob.glob(os.path.join(DIR_SAC, "pure_sac_bs256_seed*.npz")))
    sac_returns = []
    sac_ts = None
    for f in sac_files:
        data = np.load(f)
        sac_returns.append(data["returns"])
        if sac_ts is None:
            sac_ts = data["timesteps"]
    sac_returns = np.array(sac_returns)  # Shape: (5, 600)

    # 2. Policy-Mean Prior (重複データをクレンジング抽出)
    prior_files = sorted(glob.glob(os.path.join(DIR_PRIOR, "policy_mean_prior_seed*.npz")))
    prior_returns = []
    prior_ts = None
    
    for f in prior_files:
        data = np.load(f)
        raw_t = data["timesteps"]
        raw_r = data["returns"]
        
        cleaned_t, cleaned_r = extract_clean_eval_data(raw_t, raw_r, eval_freq=5000)
        prior_returns.append(cleaned_r)
        if prior_ts is None:
            prior_ts = cleaned_t

    # 最小長さに揃える
    min_len = min(sac_returns.shape[1], min(len(r) for r in prior_returns))
    
    sac_ts = sac_ts[:min_len]
    sac_returns = sac_returns[:, :min_len]
    
    prior_ts = prior_ts[:min_len]
    prior_returns = np.array([r[:min_len] for r in prior_returns])
    
    return sac_ts, sac_returns, prior_returns

def main():
    try:
        t, sac_rets, prior_rets = load_dataset()
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return

    # 平均と標準偏差
    mean_prior = np.nanmean(prior_rets, axis=0)
    std_prior = np.nanstd(prior_rets, axis=0)

    mean_sac = np.nanmean(sac_rets, axis=0)
    std_sac = np.nanstd(sac_rets, axis=0)

    # --- プロット描画 ---
    plt.figure(figsize=(10, 6))
    plt.plot(t, mean_prior, label="Policy-Mean Prior SAC (Ours)", color="#1f77b4", linewidth=2)
    plt.fill_between(t, mean_prior - std_prior, mean_prior + std_prior, color="#1f77b4", alpha=0.2)

    plt.plot(t, mean_sac, label="Vanilla SAC", color="#ff7f0e", linewidth=2)
    plt.fill_between(t, mean_sac - std_sac, mean_sac + std_sac, color="#ff7f0e", alpha=0.2)

    plt.title("Humanoid-v5: Evaluation Returns Comparison (5 Seeds)", fontsize=14)
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12, loc="lower right")
    plt.tight_layout()
    
    save_path = f"{SAVE_DIR}/returns_comparison_fixed.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    # --- 定量分析数値 ---
    idx_1m = np.searchsorted(t, 1_000_000)
    auc_prior_1m = trapezoid(mean_prior[:idx_1m], t[:idx_1m])
    auc_sac_1m = trapezoid(mean_sac[:idx_1m], t[:idx_1m])

    def count_drops(returns_arr):
        drops = 0
        for seed_idx in range(returns_arr.shape[0]):
            r_seq = returns_arr[seed_idx]
            reached = False
            for r in r_seq:
                if not np.isnan(r):
                    if r >= 4000:
                        reached = True
                    if reached and r < 3000:
                        drops += 1
        return drops

    print("==================================================")
    print("        QUANTITATIVE ANALYSIS (CLEANED)           ")
    print("==================================================")
    print(f"[1M Steps AUC (Sample Efficiency)]")
    print(f"  - Policy-Mean Prior : {auc_prior_1m:,.0f}")
    print(f"  - Vanilla SAC       : {auc_sac_1m:,.0f}")
    print(f"  - Improvement       : {((auc_prior_1m - auc_sac_1m) / auc_sac_1m) * 100:+.2f}%")
    print("--------------------------------------------------")
    print(f"[Mean Standard Deviation (Lower is More Stable)]")
    print(f"  - Policy-Mean Prior : {np.mean(std_prior):.2f}")
    print(f"  - Vanilla SAC       : {np.mean(std_sac):.2f}")
    print("--------------------------------------------------")
    print(f"[Performance Drops (Post-4000 reward < 3000 count)]")
    print(f"  - Policy-Mean Prior : {count_drops(prior_rets)} times")
    print(f"  - Vanilla SAC       : {count_drops(sac_rets)} times")
    print("==================================================")
    print(f"📁 正常に救出した比較グラフを {save_path} に保存しました。")

if __name__ == "__main__":
    main()
