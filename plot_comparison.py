import os
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "./npz_logs"

def load_entropy_avg(prefix, suffix, seeds=range(5)):
    all_entropy = []
    for s in seeds:
        # ファイル名のパターンを実際の ls の結果に完全に合わせる
        if prefix == "sac":
            path = os.path.join(LOG_DIR, f"sac_entropy_seed{s}.npz")
        else:
            # 実際のファイル名: gaussian_1.0_seed0_entropy.npz
            path = os.path.join(LOG_DIR, f"{prefix}_seed{s}_{suffix}.npz")
            
        if os.path.exists(path):
            try:
                data = np.load(path)
                # キー名が 'entropy' で入っているかチェック
                if "entropy" in data.files:
                    all_entropy.append(data["entropy"])
                else:
                    # キー名が異なる場合のフォールバック（配列が1つだけならそれを取る）
                    first_key = data.files[0]
                    all_entropy.append(data[first_key])
            except Exception as e:
                print(f"Error loading {path}: {e}")
        else:
            print(f"Warning: {path} が見つかりません。")
            
    if not all_entropy:
        return None
    
    # シード間でデータ長が数ステップズレている場合を考慮して最小サイズに揃える
    min_len = min(len(e) for e in all_entropy)
    return np.array([e[:min_len] for e in all_entropy]).mean(axis=0)

def main():
    print("各モデルのポリシーエントロピーデータを読み込み中...")
    
    sac_ent = load_entropy_avg("sac", "entropy")
    gauss_ent = load_entropy_avg("gaussian_1.0", "entropy")
    laplace_ent = load_entropy_avg("laplace_1.0", "entropy")

    # データが1つも読めていない場合は警告を出す
    if sac_ent is None and gauss_ent is None and laplace_ent is None:
        print("❌ 警告: データが1つも読み込めませんでした。ファイル名やパスを確認してください。")
        return

    # グラフのプロット設定（エントロピー専用）
    plt.figure(figsize=(10, 6))
    
    if sac_ent is not None:
        plt.plot(sac_ent, label="Standard SAC (Baseline)", color="tab:gray", alpha=0.8, linewidth=1.5)
        print(f"Standard SAC 読込成功 (データ長: {len(sac_ent)})")
    if gauss_ent is not None:
        plt.plot(gauss_ent, label="SAC + ADR (Gaussian, scale=1.0)", color="tab:blue", linewidth=1.5)
        print(f"Gaussian 1.0 読込成功 (データ長: {len(gauss_ent)})")
    if laplace_ent is not None:
        plt.plot(laplace_ent, label="SAC + ADR (Laplace, scale=1.0)", color="tab:red", linewidth=1.5)
        print(f"Laplace 1.0 読込成功 (データ長: {len(laplace_ent)})")
        
    plt.title("Policy Entropy Evolution Comparison (scale=1.0)", fontsize=14, fontweight="bold")
    plt.xlabel("Gradient Steps", fontsize=12)
    plt.ylabel("Policy Entropy", fontsize=12)
    plt.legend(fontsize=11, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # 確実に古いファイルを消して上書きを保証する
    entropy_path = os.path.join(LOG_DIR, "final_comparison_entropy.png")
    if os.path.exists(entropy_path):
        os.remove(entropy_path)
        
    plt.savefig(entropy_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\n🎉 エントロピー比較図を保存しました: {entropy_path}")

if __name__ == "__main__":
    main()
