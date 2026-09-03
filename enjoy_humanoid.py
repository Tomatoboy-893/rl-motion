import os
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from sac_adr_main import SACWithFixedPrior

MODEL_PATH = "./npz_logs/gaussian_rho0.05_seed0_model/best_model.zip"

def main():
    print(f"\n=== モデルのロードと可視化 ===")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ モデルファイルが見つかりません: {MODEL_PATH}")
        return

    # 学習時は make_vec_env("Humanoid-v5", n_envs=8) を使っていたため、
    # 観測空間の形やラッパーの構造を一致させるために vec_env を用いてテスト環境を作る
    try:
        # 評価用としても、単一環境を DummyVecEnv でラップして合わせる
        eval_env = make_vec_env("Humanoid-v5", n_envs=1, render_mode="human")
        print("✅ ベクトル環境の作成に成功しました")
    except Exception as e:
        print(f"❌ 環境の作成に失敗しました: {e}")
        return

    try:
        # SACWithFixedPrior でモデルをロード
        model = SACWithFixedPrior.load(MODEL_PATH, env=eval_env)
        print("✅ モデルのロードに成功しました！")
    except Exception as e:
        print(f"❌ モデルのロードに失敗しました: {e}")
        eval_env.close()
        return

    # 可視化ループの実行
    obs = eval_env.reset()
    total_reward = 0
    
    print("\n👀 シミュレーションを開始します...")
    try:
        for step in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward[0] # VecEnvなので配列で返る
            
            if terminated[0] or truncated[0]:
                print(f"🏁 エピソード終了 (ステップ数: {step+1})")
                break
    except KeyboardInterrupt:
        print("\n⏹️ 中断されました")
    
    eval_env.close()
    print(f"📊 トータル報酬: {total_reward:.2f}")

if __name__ == "__main__":
    main()
