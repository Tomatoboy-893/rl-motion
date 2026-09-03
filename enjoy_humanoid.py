import os
import gymnasium as gym

# カスタムクラス（ADR版SAC）をインポート
from sac_adr_main import SACWithFixedPrior

# --- 設定セクション ---
ENV_ID = "Humanoid-v5"
# 可視化したいADRモデルのパス
MODEL_PATH = "./npz_logs/gaussian_rho0.05_seed0_model/best_model.zip" 
DETERMINISTIC_POLICY = True
# ---------------------

def enjoy_adr_humanoid():
    print(f"\n=== ADR Humanoid-v5 可視化を開始 ===")
    print(f"モデル: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ エラー: モデルファイルが見つかりません -> {MODEL_PATH}")
        return

    # 1. 環境の作成 (render_mode="human" でGUIウィンドウを表示)
    try:
        env = gym.make(ENV_ID, render_mode="human")
        print("✅ 環境の作成に成功しました (GUIウィンドウが開きます)")
    except Exception as e:
        print(f"❌ エラー: 環境の作成に失敗しました: {e}")
        return

    # 2. カスタムクラスでモデルをロード
    try:
        # SAC ではなく SACWithFixedPrior でロードする
        model = SACWithFixedPrior.load(MODEL_PATH, env=env)
        print(f"✅ ADRモデルのロードに成功しました！")
    except Exception as e:
        print(f"❌ エラー: モデルのロードに失敗しました: {e}")
        env.close()
        return

    # 3. 可視化ループの実行 (1エピソード)
    obs, _ = env.reset()
    total_reward = 0
    step_count = 0
    
    print("\n👀 シミュレーションを開始します (ESCキーでウィンドウを閉じると中断します)")
    
    try:
        for step in range(1000):
            # 行動選択
            action, _states = model.predict(obs, deterministic=DETERMINISTIC_POLICY)
            
            # ステップを進める
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            if terminated or truncated:
                print(f"🏁 エピソード終了 (ステップ数: {step_count})")
                break
                
    except KeyboardInterrupt:
        print("\n⏹️ ユーザーによって中断されました")
    except Exception as e:
        print(f"⚠️ 実行中にエラーが発生しました: {e}")
    
    env.close()
    print(f"📊 トータル報酬: {total_reward:.2f}")
    print("=== 可視化を終了しました ===")

if __name__ == "__main__":
    enjoy_adr_humanoid()
