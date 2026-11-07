# rl-motion — SAC + ADR 研究コード

卒業論文の研究用コード。SAC に ADR（Adaptive Dilating Regularization）を組み合わせ、運動制御（Gymnasium / MuJoCo）での性能を検証する。土台は Stable-Baselines3 の SAC。

## 現在進行中

- **本実験**：遅延アンカー版 vs 素の SAC（Humanoid-v5、batch_size=64、3M steps、5シード）
  - 手法本体：`sac_delayed_anchor.py`（クラス `SACWithDelayedAnchorPrior`。事前分布＝Polyak 平均した少し前の方策）
  - 実行：`run_delayed_anchor_experiments.py`
  - 結果の図：`plot_delayed_anchor_results.py`（SAC 側の読み込みは `--sac-dir ./npz_logs_pure_sac_bs64` を明示する）
- **試し回し**：`run_warmup_trial.py`（ウォームアップ版。ブレーキを 100k→600k steps で徐々に有効化、500k steps、seed 0）

## 手法本体（モデルの定義）

| ファイル | 中身 |
|---|---|
| `sac_adr_main.py` | 先輩の ADR 実装。`SACWithFixedPrior`（固定事前分布、論文版）／`SACWithMovingAveragePrior`／`SACWithLaplacePrior` |
| `sac_models.py` | `SACWithPolicyMeanPrior`（事前分布の平均＝今の方策の平均）など、前段の自作実装 |
| `sac_delayed_anchor.py` | 現在の提案手法（遅延アンカー版） |
| `sac_baseline_final.py` | 素の SAC（bs256）の記録付きベースライン（定義＋実行） |
| `conditions.py` | FixedPrior の公平比較用の自己完結版（定義＋実行、保存先は `npz_logs_fixed_prior_fair`） |

## 実行スクリプト

現行：
- `run_delayed_anchor_experiments.py` — 本実験（遅延アンカー vs Pure SAC）
- `run_warmup_trial.py` — ウォームアップ版の試し回し

過去の実験：
- `run_bs64_experiments.py` — PolicyMeanPrior vs Pure SAC（Humanoid、bs64、3M steps、5シード）
- `run_humanoid_gaussian.py` / `gaussian_rho_sweep.py` — 先輩版（Gaussian 事前分布）の広さ rho の掃引（Humanoid）
- `laplace_sweep.py` / `unified_prior_experiment.py` — Laplace 事前分布の掃引
- `run_sac_baseline_humanoid.py` — Humanoid の SAC ベースライン
- `sac_rho_run.py` — 先輩版の HalfCheetah 実行（移動平均版の設定）
- `sac_policy_mean_prior_run.py` — PolicyMeanPrior の実行
- `sac_adr_run.py` / `sac_entropy_run.py` / `run_experiment.py` / `run_compare.py` — 初期の実行・比較用
- `compare_gaussian_laplace.py` / `compare_gaussian_sacbase.py` — 分布どうし・ベースラインとの比較用

## プロット（plot_*.py）

- `plot_delayed_anchor_results.py` — 遅延アンカー実験の結果図（現行）
- `plot_policy_mean_prior.py` — PolicyMeanPrior 実験の結果図
- その他の `plot_*.py` は各実験時点のもの

## その他

- `enjoy_humanoid.py` — 学習済みモデルを読み込んで動画化する
- `comparison.png` — 初期の比較図

## データの置き場所

- 実験結果の npz は、実行時に `npz_logs_*` フォルダへ保存される（Git 管理対象外。`.gitignore` 参照）
- 図の出力先は `plots/`

## 運用メモ

- 本実験の実行中は、コードファイルの移動・改名をしない（実行スクリプトの import が壊れるため）。フォルダ分けは実験完了後に行う。
