import os
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from sac_adr_main import SACWithFixedPrior

MODEL_PATH = "./npz_logs/gaussian_rho0.05_seed0_model/best_model.zip"

def main():
    print(f"\n=== Humanoid-v5 可視化スクリプト ===")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ モデルファイルが見つかりません: {MODEL_PATH}")
        return

    # 1. 単一のGUI環境を作成
    try:
        env = gym.make("Humanoid-v5", render_mode="human")
        print("✅ GUI環境の作成に成功しました")
    except Exception as e:
        print(f"❌ 環境の作成に失敗しました: {e}")
        return

    # 2. モデルのロード（次元数チェックを一時的に無視して強制ロードする安全策）
    try:
        # custom_objects を用いて環境のチェックをバイパスしつつロード
        model = SACWithFixedPrior.load(
            MODEL_PATH, 
            env=env,
            custom_objects={"observation_space": env.observation_space, "action_space": env.action_space}
        )
        print("✅ モデルのロードに成功しました！")
    except Exception as e:
        print(f"❌ モデルのロードに失敗しました: {e}")
        env.close()
        return

    # 3. 可視化ループの実行
    obs, _ = env.reset()
    total_reward = 0
    
    print("\n👀 シミュレーションを開始します (ウィンドウを閉じると終了します)")
    try:
        for step in range(1000):
            # 決定論的な行動を選択
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            
            if terminated or truncated:
                print(f"🏁 エピソード終了 (ステップ数: {step + 1})")
                break
    except KeyboardInterrupt:
        print("\n⏹️ ユーザーによって中断されました")
    except Exception as e:
        print(f"⚠️ 実行中にエラーが発生しました: {e}")
    
    env.close()
    print(f"📊 トータル報酬: {total_reward:.2f}")
    print("=== 可視化を終了しました ===")

if __name__ == "__main__":
    main()
