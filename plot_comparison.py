import os
import numpy as np
import matplotlib.pyplot as plt

# データが格納されているディレクトリ
LOG_DIR = "./npz_logs"

def load_and_average_entropy(prefix, seeds=range(5)):
    """各シードのエントロピーデータを読み込み、最小の長さに揃えて平均を計算する関数"""
    all_entropy = []
    for s in seeds:
        path = os.path.join(LOG_DIR, f"{prefix}_entropy_seed{s}.npz")
        if os.path.exists(path):
            data = np.load(path)
            all_entropy.append(data["entropy"])
        else:
            print(f"Warning: {path} が見つかりません。")
    
    if not all_entropy:
        return None
    
    # シード間でデータの長さが微妙に違う場合を考慮して、最小の長さに揃える（アラインメント）
    min_len = min(len(e) for e in all_entropy)
    aligned_entropy = np.array([e[:min_len] for e in all_entropy])
    
    # 5シード分の平均を計算
    mean_entropy = aligned_entropy.mean(axis=0)
    return mean_entropy

def load_and_average_returns(prefix, seeds=range(5)):
    """各シードの報酬（Return）データを読み込み、最小の長さに揃えて平均を計算する関数"""
    all_returns = []
    timesteps = None
    for s in seeds:
        path = os.path.join(LOG_DIR, f"{prefix}_seed{s}.npz")
        if os.path.exists(path):
            data = np.load(path)
            all_returns.append(data["returns"])
            if timesteps is None:
                timesteps = data["timesteps"]
        else:
            print(f"Warning: {path} が見つかりません。")
            
    if not all_returns:
        return None, None
        
    min_len = min(len(r) for r in all_returns)
    aligned_returns = np.array([r[:min_len] for r in all_returns])
    mean_returns = aligned_returns.mean(axis=0)
    
    return timesteps[:min_len], mean_returns

def main():
    # ----------------------------------------------------
    # 1. ポリシーエントロピーの比較プロット
    # ----------------------------------------------------
    print("エントロピーデータを読み込み中...")
    sac_ent = load_and_average_entropy("sac")
    gauss_ent = load_and_average_entropy("sac_adr_gauss")      # ガウスベースのファイル名（接頭辞）に合わせて変更してください
    laplace_ent = load_and_average_entropy("sac_adr_laplace")  # ラプラスベースのファイル名（接頭辞）に合わせて変更してください

    plt.figure(figsize=(10, 6))
    
    if sac_ent is not None:
        plt.plot(sac_ent, label="Standard SAC (Baseline)", color="gray", alpha=0.8)
    if gauss_ent is not None:
        plt.plot(gauss_ent, label="SAC + ADR (Gaussian)", color="royalblue")
    if laplace_ent is not None:
        plt.plot(laplace_ent, label="SAC + ADR (Laplace)", color="crimson")
        
    plt.title("Policy Entropy Comparison (1,000,000 Steps)", fontsize=14)
    plt.xlabel("Gradient Steps", fontsize=12)
    plt.ylabel("Policy Entropy", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    entropy_img_path = os.path.join(LOG_DIR, "comparison_policy_entropy.png")
    plt.savefig(entropy_img_path, dpi=300)
    plt.close()
    print(f"エントロピー比較図を保存しました: {entropy_img_path}")

    # ----------------------------------------------------
    # 2. 獲得報酬（Return）の比較プロット
    # ----------------------------------------------------
    print("報酬データを読み込み中...")
    ts_sac, sac_ret = load_and_average_returns("sac")
    _, gauss_ret = load_and_average_returns("sac_adr_gauss")
    _, laplace_ret = load_and_average_returns("sac_adr_laplace")

    plt.figure(figsize=(10, 6))
    
    if sac_ret is not None and ts_sac is not None:
        plt.plot(ts_sac, sac_ret, label="Standard SAC (Baseline)", color="gray", alpha=0.8)
    if gauss_ret is not None and ts_sac is not None:
        plt.plot(ts_sac[:len(gauss_ret)], gauss_ret, label="SAC + ADR (Gaussian)", color="royalblue")
    if laplace_ret is not None and ts_sac is not None:
        plt.plot(ts_sac[:len(laplace_ret)], laplace_ret, label="SAC + ADR (Laplace)", color="crimson")
        
    plt.title("Learning Curve Comparison (1,000,000 Steps)", fontsize=14)
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Mean Episodic Return", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    return_img_path = os.path.join(LOG_DIR, "comparison_learning_curve.png")
    plt.savefig(return_img_path, dpi=300)
    plt.close()
    print(f"報酬比較図を保存しました: {return_img_path}")

if __name__ == "__main__":
    main()
