# plot_all_metrics.py
import os
import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "./npz_logs_moving_average_prior"

def main():
    # 5シード分のモデルファイルやデータから、各指標を集計する
    # ※各 model からログを抽出して保存している前提、あるいは保存時にnpzに入れている場合の拡張用
    # 今回は各シードのモデル（adr_moving_avg_seed{seed}_model.zip）から直接復元・取得するアプローチ
    from stable_baselines3 import SAC
    from sac_adr_main import SACWithMovingAveragePrior

    seeds = range(5)
    
    all_kls = []
    all_entropies = []
    all_norms = []
    
    timesteps_metric = None

    print("Loading models and extracting metrics across all 5 seeds...")
    for seed in seeds:
        model_path = os.path.join(SAVE_DIR, f"adr_moving_avg_seed{seed}_model.zip")
        if not os.path.exists(model_path):
            print(f"[Warning] Model not found for seed {seed}: {model_path}")
            continue
        
        # モデルのロード（カスタムクラスを指定）
        model = SACWithMovingAveragePrior.load(model_path)
        
        # 1. KLダイバージェンス
        if hasattr(model, "kl_values") and len(model.kl_values) > 0:
            all_kls.append(model.kl_values)
        
        # 2. エントロピー
        if hasattr(model, "pi_entropies") and len(model.pi_entropies) > 0:
            all_entropies.append(model.pi_entropies)
            
        # 3. プライオ平均ノルム
        if hasattr(model, "prior_mean_logs") and len(model.prior_mean_logs) > 0:
            norms = [np.linalg.norm(m) for m in model.prior_mean_logs]
            all_norms.append(norms)

    # 共通のプロット関数（平均 ＋ stdバンド）
    def plot_mean_std(data_list, ylabel, title, filename, color):
        if len(data_list) == 0:
            print(f"[Skip] No data for {title}")
            return
        
        # 長さを最小のものに合わせる（万が一のズレ防止）
        min_len = min(len(d) for d in data_list)
        truncated_data = np.array([d[:min_len] for d in data_list]) # 形: (5, min_len)
        
        steps = np.arange(min_len)
        mean = truncated_data.mean(axis=0)
        std = truncated_data.std(axis=0)
        
        plt.figure(figsize=(8, 5), dpi=300)
        plt.plot(steps, mean, label=f"Mean (5 Seeds)", color=color, linewidth=2)
        plt.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2, label="± 1 std")
        
        plt.xlabel("Gradient Update Steps", fontsize=11)
        plt.ylabel(ylabel, fontsize=11)
        plt.title(title, fontsize=13)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        out_path = os.path.join(SAVE_DIR, filename)
        plt.savefig(out_path)
        plt.close()
        print(f"[Saved] {out_path}")

    # 4つの指標すべてをプロット
    # 1. 学習曲線（すでに保存されている 5seed_mean_std.npz を活用）
    data_path = os.path.join(SAVE_DIR, "adr_moving_avg_5seed_mean_std.npz")
    if os.path.exists(data_path):
        data = np.load(data_path)
        t = data["timesteps"]
        mean_ret = data["mean"]
        std_ret = data["std"]
        
        plt.figure(figsize=(8, 5), dpi=300)
        plt.plot(t, mean_ret, label="ADR (Moving Avg Prior)", color="tab:purple", linewidth=2)
        plt.fill_between(t, mean_ret - std_ret, mean_ret + std_ret, color="tab:purple", alpha=0.2, label="± 1 std")
        plt.xlabel("Timesteps", fontsize=11)
        plt.ylabel("Mean Episodic Return", fontsize=11)
        plt.title("Learning Curve: ADR (Moving Average Prior)", fontsize=13)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "moving_avg_returns_all5.png"))
        plt.close()
        print(f"[Saved] {SAVE_DIR}/moving_avg_returns_all5.png")

    # 2. プライオ平均ノルムの5シード統合プロット
    plot_mean_std(
        all_norms, 
        ylabel="L2 Norm ||μ_ρ||", 
        title="Evolution of Moving Average Prior Mean Norm (5 Seeds)", 
        filename="moving_avg_prior_mean_norm_all5.png", 
        color="tab:blue"
    )

    # 3. KLダイバージェンスの5シード統合プロット
    plot_mean_std(
        all_kls, 
        ylabel="KL Divergence KL(π||ρ)", 
        title="KL(π||ρ) over updates (5 Seeds)", 
        filename="moving_avg_kl_all5.png", 
        color="tab:green"
    )

    # 4. 方策エントロピーの5シード統合プロット
    plot_mean_std(
        all_entropies, 
        ylabel="Entropy", 
        title="Policy Entropy over updates (5 Seeds)", 
        filename="moving_avg_entropy_all5.png", 
        color="tab:orange"
    )

    print("All 4 metrics successfully plotted across all 5 seeds!")

if __name__ == "__main__":
    main()
