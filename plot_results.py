
import os
import numpy as np
import matplotlib.pyplot as plt
from sac_adr_main import SACWithFixedPrior

SAVE_DIR = "./npz_logs_full_comparison"

def main():
    print("===== グラフ描画スクリプトを開始します =====")

    # 1. 5シード平均・標準偏差データのロード
    try:
        sac_data = np.load(os.path.join(SAVE_DIR, "sac_5seed_mean_std.npz"))
        adr1_data = np.load(os.path.join(SAVE_DIR, "adr_std1.0_5seed_mean_std.npz"))
        adr2_data = np.load(os.path.join(SAVE_DIR, "adr_std2.0_5seed_mean_std.npz"))
    except FileNotFoundError as e:
        print(f"❌ 必要な .npz ファイルが見つかりません: {e}")
        print("先に `sac_rho_run.py` を実行して学習・保存を完了させてください。")
        return

    # ===============================
    # ① リターンの比較グラフ (Mean ± Std)
    # ===============================
    plt.figure(figsize=(9, 6))
    
    # Standard SAC
    plt.plot(sac_data["timesteps"], sac_data["mean"], label="Standard SAC", color="tab:blue", linewidth=2)
    plt.fill_between(sac_data["timesteps"], sac_data["mean"] - sac_data["std"], sac_data["mean"] + sac_data["std"], color="tab:blue", alpha=0.2)

    # ADR (std=1.0)
    plt.plot(adr1_data["timesteps"], adr1_data["mean"], label="ADR (std=1.0)", color="tab:orange", linewidth=2)
    plt.fill_between(adr1_data["timesteps"], adr1_data["mean"] - adr1_data["std"], adr1_data["mean"] + adr1_data["std"], color="tab:orange", alpha=0.2)

    # ADR (std=2.0)
    plt.plot(adr2_data["timesteps"], adr2_data["mean"], label="ADR (std=2.0)", color="tab:green", linewidth=2)
    plt.fill_between(adr2_data["timesteps"], adr2_data["mean"] - adr2_data["std"], adr2_data["mean"] + adr2_data["std"], color="tab:green", alpha=0.2)

    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Mean Episodic Return", fontsize=12)
    plt.title("HalfCheetah-v5: Return Comparison (SAC vs ADR)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    ret_path = os.path.join(SAVE_DIR, "comparison_returns.png")
    plt.savefig(ret_path, dpi=300)
    plt.close()
    print(f"📈 保存完了: {ret_path}")

    # ===============================
    # 保存されたモデル（seed 0）のロードによる内部メトリクス描画
    # ===============================
    model_adr1_path = os.path.join(SAVE_DIR, "adr_std1.0_seed0_model.zip")
    model_adr2_path = os.path.join(SAVE_DIR, "adr_std2.0_seed0_model.zip")

    m1, m2 = None, None
    if os.path.exists(model_adr1_path):
        m1 = SACWithFixedPrior.load(model_adr1_path, print_system_info=False)
    if os.path.exists(model_adr2_path):
        m2 = SACWithFixedPrior.load(model_adr2_path, print_system_info=False)

    # ② エントロピーの比較グラフ
    plt.figure(figsize=(8, 5))
    has_entropy = False
    if m1 is not None and hasattr(m1, "pi_entropies") and len(m1.pi_entropies) > 0:
        plt.plot(m1.pi_entropies, label="ADR (std=1.0)", color="tab:orange", linewidth=1.5)
        has_entropy = True
    if m2 is not None and hasattr(m2, "pi_entropies") and len(m2.pi_entropies) > 0:
        plt.plot(m2.pi_entropies, label="ADR (std=2.0)", color="tab:green", linewidth=1.5)
        has_entropy = True
        
    if has_entropy:
        plt.title("Comparison: Pi Entropy over updates", fontsize=13)
        plt.xlabel("Updates", fontsize=11)
        plt.ylabel("Entropy", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        ent_path = os.path.join(SAVE_DIR, "comparison_entropy.png")
        plt.savefig(ent_path, dpi=300)
        plt.close()
        print(f"📈 保存完了: {ent_path}")
    else:
        plt.close()
        print("⚠️ エントロピーの記録データが見つかりませんでした（スキップします）。")

    # ③ KLダイバージェンスの比較グラフ
    plt.figure(figsize=(8, 5))
    has_kl = False
    if m1 is not None and hasattr(m1, "kl_values") and len(m1.kl_values) > 0:
        plt.plot(m1.kl_values, label="ADR (std=1.0) KL", color="tab:orange", linewidth=1.5)
        has_kl = True
    if m2 is not None and hasattr(m2, "kl_values") and len(m2.kl_values) > 0:
        plt.plot(m2.kl_values, label="ADR (std=2.0) KL", color="tab:green", linewidth=1.5)
        has_kl = True
        
    if has_kl:
        plt.title("Comparison: KL(π||ρ) over updates", fontsize=13)
        plt.xlabel("Updates", fontsize=11)
        plt.ylabel("KL Divergence", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        kl_path = os.path.join(SAVE_DIR, "comparison_kl_loss.png")
        plt.savefig(kl_path, dpi=300)
        plt.close()
        print(f"📈 保存完了: {kl_path}")
    else:
        plt.close()
        print("⚠️ KLダイバージェンスの記録データが見つかりませんでした（スキップします）。")

    print(f"🎉 すべてのグラフ描画が完了しました！保存先: {SAVE_DIR}/")

if __name__ == "__main__":
    main()
