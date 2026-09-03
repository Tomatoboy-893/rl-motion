import os

# --- ヘッドレス環境（サーバー）でのOpenGLエラーを防ぐための設定 ---
os.environ["MUJOCO_GL"] = "egl"
# -------------------------------------------------------------

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from sac_adr_main import SACWithFixedPrior

# モンキーパッチで初期化引数の不足を回避
_original_init = SACWithFixedPrior.__init__
def _patched_init(self, *args, beta_kl=0.01, beta_lr=1e-3, target_kl=1.0, prior_std=0.05, **kwargs):
    _original_init(self, *args, beta_kl=beta_kl, beta_lr=beta_lr, target_kl=target_kl, prior_std=prior_std, **kwargs)
SACWithFixedPrior.__init__ = _patched_init

MODEL_PATH = "gaussian_scale0.1_run0_entropy.npz"
VIDEO_DIR = "./videos"

def main():
    print(f"\n=== Humanoid 動画レンダリングスクリプト ===")
    os.makedirs(VIDEO_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ モデルファイルが見つかりません: {MODEL_PATH}")
        return

    # 1. 録画用環境の作成
    try:
        env = gym.make("Humanoid-v4", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env, 
            video_folder=VIDEO_DIR, 
            episode_trigger=lambda e: True,
            name_prefix="humanoid_eval"
        )
        print("✅ 録画用環境の作成に成功しました")
    except Exception as e:
        print(f"❌ 環境の作成に失敗しました: {e}")
        return

    # 2. モデルのロード
    try:
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

    # 3. シミュレーションの実行（動画として保存）
    obs, _ = env.reset()
    total_reward = 0
    
    print("\n🎬 シミュレーションを実行中（動画を保存しています...）")
    try:
        for step in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"🏁 エピソード終了 (ステップ数: {step + 1})")
                break
    except Exception as e:
        print(f"⚠️ 実行中にエラーが発生しました: {e}")
    
    env.close()
    print(f"📊 トータル報酬: {total_reward:.2f}")
    print(f"💾 動画が保存されました: {VIDEO_DIR}/ フォルダを確認してください！")

if __name__ == "__main__":
    main()
