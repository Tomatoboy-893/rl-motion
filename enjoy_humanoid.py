import os

# 💡 ヘッドレスサーバーでのレンダリング用設定
os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC

def main():
    model_path = "./npz_logs_humanoid/gaussian_scale2.0_seed4_model.zip"
    video_dir = "./videos_humanoid"
    
    os.makedirs(video_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"⚠️ エラー: 指定したモデルが見つかりません -> {model_path}")
        return

    print(f"🎬 モデルをロード中: {model_path}")
    
    # Humanoid-v5 環境の作成
    env = gym.make("Humanoid-v5", render_mode="rgb_array",max_episode_steps=3000)
    
    # 動画保存用ラッパーの適用
    env = RecordVideo(
        env, 
        video_folder=video_dir, 
        episode_trigger=lambda x: True,
        name_prefix="humanoid_walk"
    )
    
    model = SAC.load(model_path, env=env)
    
    num_episodes = 3
    print(f"🎥 動画の生成を開始します（計 {num_episodes} エピソード）...")
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
        print(f"Episode {ep+1} 終了 | 報酬: {total_reward:.2f}")
        
    env.close()
    print(f"🎉 動画の保存が完了しました！ 保存先フォルダ: {video_dir}/")

if __name__ == "__main__":
    main()
