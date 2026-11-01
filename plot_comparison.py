import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ディレクトリ設定
DIR_PRIOR = "./npz_logs_policy_mean_prior"
DIR_SAC = "./npz_logs_pure_sac_bs256"
SAVE_DIR = "./comparison_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# NumPy 2.0+ 互換の台形積分関数
trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

def load_data_from_dir(directory, prefix, keys):
    files = sorted(glob.glob(os.path.join(directory, f"{prefix}*.npz")))
    if not files:
        raise FileNotFoundError(f"No files found in {directory} with prefix {prefix}")
    
    data_dict = {key: [] for key in keys}
    timesteps_list = []

    for f in files:
        npz = np.load(f)
        timesteps_list.append(npz["timesteps"])
        for key in keys:
            data_dict[key].append(npz[key])
            
    # 全ファイルの中で最小のデータ長に切り揃え
    min_len = min(len(arr) for arr in timesteps_list)
    for key in keys:
        min_len = min(min_len, min(len(arr) for arr in data_dict[key]))
        
    timesteps = timesteps_list[0][:min_len]
    for key in keys:
        data_dict[key] = np.array([arr[:min_len] for arr in data_dict[key]], dtype=np.float64)
        
    return timesteps, data_dict

def analyze_and_plot():
    # データの読み込み
    t_prior, prior_data = load_data_from_dir(
        DIR_PRIOR, "policy_mean_prior_seed", ["returns", "pi_entropies"]
    )
    t_sac, sac_data = load_data_from_dir(
        DIR_SAC, "pure_sac_bs256_seed", ["returns", "pi_entropies"]
    )

    # 共通タイムステップ数の調整
    min_steps = min(len(t_prior), len(t_sac))
    t = t_prior[:min_steps]
    
    returns_prior = prior_data["returns"][:, :min_steps]
    returns_sac = sac_data["returns"][:, :min_steps]
    
    entropy_prior = prior_data["pi_entropies"][:, :min_steps]
    entropy_sac = sac_data["pi_entropies"][:, :min_steps]

    # --- 1. 評価報酬（Returns）の比較プロット ---
    plt.figure(figsize=(10, 6))
    
    mean_prior = np.nanmean(returns_prior, axis=0)
    std_prior = np.nanstd(returns_prior, axis=0)
    
    mean_sac = np.nanmean(returns_sac, axis=0)
    std_sac = np.nanstd(returns_sac, axis=0)

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
    plt.savefig(f"{SAVE_DIR}/returns_comparison.png", dpi=300)
    plt.close()

    # --- 2. エントロピー（Policy Entropy）の比較プロット ---
    plt.figure(figsize=(10, 6))
    
    mean_ent_prior = np.nanmean(entropy_prior, axis=0)
    std_ent_prior = np.nanstd(entropy_prior, axis=0)
    
    mean_ent_sac = np.nanmean(entropy_sac, axis=0)
    std_ent_sac = np.nanstd(entropy_sac, axis=0)

    plt.plot(t, mean_ent_prior, label="Policy-Mean Prior SAC", color="#1f77b4", linewidth=2)
    plt.fill_between(t, mean_ent_prior - std_ent_prior, mean_ent_prior + std_ent_prior, color="#1f77b4", alpha=0.2)

    plt.plot(t, mean_ent_sac, label="Vanilla SAC", color="#ff7f0e", linewidth=2)
    plt.fill_between(t, mean_ent_sac - std_ent_sac, mean_ent_sac + std_ent_sac, color="#ff7f0e", alpha=0.2)

    plt.title("Humanoid-v5: Policy Entropy Comparison", fontsize=14)
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Entropy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/entropy_comparison.png", dpi=300)
    plt.close()

    # --- 定量データの統計数値計算 ---
    # AUC（0〜1,000,000ステップの平均報酬エリア）
    idx_1m = np.searchsorted(t, 1_000_000)
    auc_prior_1m = trapezoid(mean_prior[:idx_1m], t[:idx_1m])
    auc_sac_1m = trapezoid(mean_sac[:idx_1m], t[:idx_1m])

    # 平均標準偏差（全体のブローの大きさ）
    mean_std_prior = np.nanmean(std_prior)
    mean_std_sac = np.nanmean(std_sac)

    # 崩壊（ドロップ）回数のカウント（一度4,000以上に到達した後、3,000未満に落とした回数）
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
    print("               QUANTITATIVE ANALYSIS              ")
    print("==================================================")
    print(f"[1M Steps AUC (Sample Efficiency)]")
    print(f"  - Policy-Mean Prior : {auc_prior_1m:,.0f}")
    print(f"  - Vanilla SAC       : {auc_sac_1m:,.0f}")
    print(f"  - Improvement       : {((auc_prior_1m - auc_sac_1m) / auc_sac_1m) * 100:+.2f}%")
    print("--------------------------------------------------")
    print(f"[Mean Standard Deviation (Lower is More Stable)]")
    print(f"  - Policy-Mean Prior : {mean_std_prior:.2f}")
    print(f"  - Vanilla SAC       : {mean_std_sac:.2f}")
    print("--------------------------------------------------")
    print(f"[Performance Drops (Post-4000 reward < 3000 count)]")
    print(f"  - Policy-Mean Prior : {count_drops(returns_prior)} times")
    print(f"  - Vanilla SAC       : {count_drops(returns_sac)} times")
    print("==================================================")
    print(f"📁 グラフと解析結果を {SAVE_DIR} に保存しました。")

if __name__ == "__main__":
    analyze_and_plot()
