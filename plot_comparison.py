import os
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "./npz_logs"

def load_entropy_avg(prefix, suffix, seeds=range(5)):
    all_entropy = []
    for s in seeds:
        # sacの場合は sac_entropy_seedX.npz、それ以外は {prefix}_seedX_{suffix}.npz
        if prefix == "sac":
            path = os.path.join(LOG_DIR, f"sac_entropy_seed{s}.npz")
        else:
            path = os.path.join(LOG_DIR, f"{prefix}_seed{s}_{suffix}.npz")
            
        if os.path.exists(path):
            data = np.load(path)
            # キー名が 'entropy' であることを想定
            all_entropy.append(data["entropy"])
        else:
            print(f"Warning: {path} が見つかりません。")
            
    if not all_entropy:
        return None
    
    min_len = min(len(e) for e in all_entropy)
    return np.array([e[:min_len] for e in all_entropy]).mean(axis=0)

def load_returns_avg(prefix, seeds=range(5)):
    all_returns = []
    timesteps = None
    for s in seeds:
        path = os.path.join(LOG_DIR, f"{prefix}_seed{s}.npz")
        if os.path.exists(path):
            data = np.load(path)
            all_returns.append(data["returns"])
            if timesteps is None and "timesteps" in data:
                timesteps = data["timesteps"]
        else:
            print(f"Warning: {path} が見つかりません。")
            
    if not all_returns:
        return None, None
        
    min_len = min(len(r) for r in all_returns)
    aligned_returns = np.array([r[:min_len] for r in all_returns])
    mean_returns = aligned_returns.mean(axis=0)
    
    # もしtimestepsが保存されていなければ、評価頻度5000ステップ刻みで自作
    if timesteps is None:
        timesteps = np.arange(1, min_len + 1) * 5000
        
    return timesteps[:min_len], mean_returns

def main():
    # ----------------------------------------------------
    # 1. ポリシーエントロピーの比較プロット (1M Steps)
    # ----------------------------------------------------
    print("エントロピーデータを読み込み中...")
    sac_ent = load_entropy_avg("sac", "entropy")
    gauss_ent = load_entropy_avg("gaussian_1.0", "entropy")
    laplace_ent = load_entropy_avg("laplace_1.0", "entropy")

    plt.figure(figsize=(10, 6))
    
    if sac_ent is not None:
        plt.plot(sac_ent, label="Standard SAC (Baseline)", color="tab:gray", alpha=0.8, linewidth=1.5)
    if gauss_ent is not None:
        plt.plot(gauss_ent, label="SAC + ADR (Gaussian, scale=1.0)", color="tab:blue", linewidth=1.5)
    if laplace_ent is not None:
        plt.plot(laplace_ent, label="SAC + ADR (Laplace, scale=1.0)", color="tab:red", linewidth=1.5)
        
    plt.title("Policy Entropy Evolution Comparison", fontsize=14, fontweight="bold")
    plt.xlabel("Gradient Steps", fontsize=12)
    plt.ylabel("Policy Entropy", fontsize=12)
    plt.legend(fontsize=11, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    entropy_path = os.path.join(LOG_DIR, "final_comparison_entropy.png")
    plt.savefig(entropy_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"🎉 エントロピー比較図を保存しました: {entropy_path}")

    # ----------------------------------------------------
    # 2. 獲得報酬（Return）の比較プロット
    # ----------------------------------------------------
    print("報酬データを読み込み中...")
    ts_sac, sac_ret = load_returns_avg("sac")
    ts_gauss, gauss_ret = load_returns_avg("gaussian_1.0")
    ts_laplace, laplace_ret = load_returns_avg("laplace_1.0")

    plt.figure(figsize=(10, 6))
    
    if sac_ret is not None:
        plt.plot(ts_sac, sac_ret, label="Standard SAC (Baseline)", color="tab:gray", alpha=0.8, linewidth=1.5)
    if gauss_ret is not None:
        plt.plot(ts_gauss, gauss_ret, label="SAC + ADR (Gaussian, scale=1.0)", color="tab:blue", linewidth=1.5)
    if laplace_ret is not None:
        plt.plot(ts_laplace, laplace_ret, label="SAC + ADR (Laplace, scale=1.0)", color="tab:red", linewidth=1.5)
        
    plt.title("Learning Curve Comparison (HalfCheetah-v5)", fontsize=14, fontweight="bold")
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Mean Episodic Return", fontsize=12)
    plt.legend(fontsize=11, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    return_path = os.path.join(LOG_DIR, "final_comparison_return.png")
    plt.savefig(return_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"🎉 報酬比較図を保存しました: {return_path}")

if __name__ == "__main__":
    main()
