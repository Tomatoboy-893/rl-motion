import os
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "./npz_logs"

def load_entropy_avg(prefix, suffix, seeds=range(5)):
    """各モデル・各シードのエントロピーデータを読み込んで平均を計算する関数"""
    all_entropy = []
    
    # 標準SAC（ベースライン）の場合
    if prefix == "sac":
        for s in seeds:
            path = os.path.join(LOG_DIR, f"sac_entropy_seed{s}.npz")
            if os.path.exists(path):
                data = np.load(path)
                all_entropy.append(data["entropy"])
        if not all_entropy:
            return None
        min_len = min(len(e) for e in all_entropy)
        return np.array([e[:min_len] for e in all_entropy]).mean(axis=0)

    # ガウス（rho）またはラプラスの場合
    # あなたのフォルダ構造：gaussian_rho0.1_mean_std.npz などの集計済みデータ、または個別シード
    # ここでは、個別シードの _entropy.npz から平均を計算するロジックにします
    for s in seeds:
        path = os.path.join(LOG_DIR, f"{prefix}_seed{s}_{suffix}.npz")
        if os.path.exists(path):
            try:
                data = np.load(path)
                if "entropy" in data.files:
                    all_entropy.append(data["entropy"])
                else:
                    all_entropy.append(data[data.files[0]])
            except Exception as e:
                print(f"Error loading {path}: {e}")
                
    if not all_entropy:
        return None
    
    min_len = min(len(e) for e in all_entropy)
    return np.array([e[:min_len] for e in all_entropy]).mean(axis=0)

def main():
    # 比較したいパラメータのリスト
    scales = ["0.1", "0.5", "1.0", "2.0"]
    
    print("ベースライン（Standard SAC）のデータを読み込み中...")
    sac_ent = load_entropy_avg("sac", "entropy")
    if sac_ent is not None:
        print(f"-> Standard SAC 読込成功 (データ長: {len(sac_ent)})")

    for scale in scales:
        print(f"\n--- パラメータ {scale} のデータを処理中 ---")
        
        # 💡 ここがポイント！ガウスは1.0だけ「rho」がつかない名前になっているので分岐させる
        if scale == "1.0":
            gauss_prefix = "gaussian_1.0"
        else:
            gauss_prefix = f"gaussian_rho{scale}"
            
        laplace_prefix = f"laplace_{scale}"
        
        # データの読み込み
        gauss_ent = load_entropy_avg(gauss_prefix, "entropy")
        laplace_ent = load_entropy_avg(laplace_prefix, "entropy")
        
        if gauss_ent is None and laplace_ent is None:
            print(f"Warning: パラメータ {scale} に該当するガウス/ラプラスのデータが見つかりません。")
            continue
            
        # グラフプロット
        plt.figure(figsize=(10, 6))
        
        # 1. 標準SAC (灰色)
        if sac_ent is not None:
            plt.plot(sac_ent, label="Standard SAC (Baseline)", color="tab:gray", alpha=0.7, linewidth=1.5)
            
        # 2. ガウスADR (青色)
        if gauss_ent is not None:
            plt.plot(gauss_ent, label=f"SAC + ADR (Gaussian, scale={scale})", color="tab:blue", linewidth=1.5)
            print(f"-> Gaussian {scale} 読込成功 (データ長: {len(gauss_ent)})")
        else:
            print(f"-> ⚠️ Gaussian {scale} は個別シードデータが見つからないか、プレフィックスが違います")
            
        # 3. ラプラスADR (赤色)
        if laplace_ent is not None:
            plt.plot(laplace_ent, label=f"SAC + ADR (Laplace, scale={scale})", color="tab:red", linewidth=1.5)
            print(f"-> Laplace {scale} 読込成功 (データ長: {len(laplace_ent)})")
            
        plt.title(f"Policy Entropy Evolution Comparison (scale={scale})", fontsize=14, fontweight="bold")
        plt.xlabel("Gradient Steps", fontsize=12)
        plt.ylabel("Policy Entropy", fontsize=12)
        plt.legend(fontsize=11, loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.5)
        
        # 画像の保存
        save_path = os.path.join(LOG_DIR, f"final_comparison_entropy_scale_{scale}.png")
        if os.path.exists(save_path):
            os.remove(save_path)
            
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"🎉 3手法比較図（scale={scale}）を保存しました: {save_path}")

if __name__ == "__main__":
    main()
