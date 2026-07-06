import os
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "./npz_logs"

def load_entropy_avg(prefix, suffix, seeds=range(5)):
    """各モデル・各シードのエントロピーデータを読み込んで平均を計算する関数"""
    all_entropy = []
    for s in seeds:
        if prefix == "sac":
            path = os.path.join(LOG_DIR, f"sac_entropy_seed{s}.npz")
        else:
            path = os.path.join(LOG_DIR, f"{prefix}_seed{s}_{suffix}.npz")
            
        if os.path.exists(path):
            try:
                data = np.load(path)
                if "entropy" in data.files:
                    all_entropy.append(data["entropy"])
                else:
                    first_key = data.files[0]
                    all_entropy.append(data[first_key])
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
    if not all_entropy:
        return None
    
    min_len = min(len(e) for e in all_entropy)
    return np.array([e[:min_len] for e in all_entropy]).mean(axis=0)

def main():
    # 比較したいスケールのリスト
    scales = ["0.1", "0.5", "1.0", "2.0"]
    
    print("ベースライン（Standard SAC）のデータを読み込み中...")
    sac_ent = load_entropy_avg("sac", "entropy")
    if sac_ent is not None:
        print(f"-> Standard SAC 読込成功 (データ長: {len(sac_ent)})")
    else:
        print("-> ⚠️ Standard SAC のデータが見つかりません。")

    # 各スケールごとにループを回してグラフを生成
    for scale in scales:
        print(f"\n--- スケール {scale} のデータを処理中 ---")
        
        # プレフィックスを実際のファイル名（gaussian_0.1, laplace_0.1 など）に合わせる
        gauss_prefix = f"gaussian_{scale}"
        laplace_prefix = f"laplace_{scale}"
        
        gauss_ent = load_entropy_avg(gauss_prefix, "entropy")
        laplace_ent = load_entropy_avg(laplace_prefix, "entropy")
        
        # 該当するスケールのデータがどちらもない場合はスキップ
        if gauss_ent is None and laplace_ent is None:
            print(f"Warning: スケール {scale} に該当するガウス/ラプラスのデータが見つかりません。")
            continue
            
        # グラフプロットの設定
        plt.figure(figsize=(10, 6))
        
        # 共通のベースライン（標準SAC）
        if sac_ent is not None:
            plt.plot(sac_ent, label="Standard SAC (Baseline)", color="tab:gray", alpha=0.7, linewidth=1.5)
            
        # ガウスベース
        if gauss_ent is not None:
            plt.plot(gauss_ent, label=f"SAC + ADR (Gaussian, scale={scale})", color="tab:blue", linewidth=1.5)
            print(f"-> Gaussian {scale} 読込成功 (データ長: {len(gauss_ent)})")
            
        # ラプラスベース
        if laplace_ent is not None:
            plt.plot(laplace_ent, label=f"SAC + ADR (Laplace, scale={scale})", color="tab:red", linewidth=1.5)
            print(f"-> Laplace {scale} 読込成功 (データ長: {len(laplace_ent)})")
            
        plt.title(f"Policy Entropy Evolution Comparison (scale={scale})", fontsize=14, fontweight="bold")
        plt.xlabel("Gradient Steps", fontsize=12)
        plt.ylabel("Policy Entropy", fontsize=12)
        plt.legend(fontsize=11, loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.5)
        
        # スケールごとのファイル名で保存
        save_path = os.path.join(LOG_DIR, f"comparison_entropy_scale_{scale}.png")
        if os.path.exists(save_path):
            os.remove(save_path)
            
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"🎉 比較図を保存しました: {save_path}")

if __name__ == "__main__":
    main()
