import os
import gymnasium as gym
from stable_bas3 import SAC # 今回はSACと仮定します

# --- 設定セクション ---
ENV_ID = "Humanoid-v5"

# 可視化したい学習済みモデルのパスを指定してください
# 例: "./logs/best_model.zip" や "./saved_models/sac_humanoid_3M.zip"
MODEL_PATH = "path/to/your/saved/model.zip" 

# 評価時の設定: 決定論的（確率的なノイズを入れない）な動きにするか、探索を含めるか
# 通常、学習結果の純粋な性能を見る場合は True にします。
DETERMINISTIC_POLICY = True
# ---------------------

def enjoy_humanoid(model_path, env_id, deterministic=True):
    print(f"\n=== Humanoid-v5 可視化を開始 ===")
    print(f"モデル: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ エラー: モデルファイルが見つかりません -> {model_path}")
        return

    # 1. 環境の作成 (render_mode="human" でGUIウィンドウを表示)
    try:
        # Humanoid-v5 はレンダリングにMuJoCoを使用します
        env = gym.make(env_id, render_mode="human")
        print("✅ 環境の作成に成功しました (GUIウィンドウが開きます)")
    except Exception as e:
        print(f"❌ エラー: 環境の作成に失敗しました。MuJoCo環境が正しくインストールされているか確認してください。\n詳細: {e}")
        return

    # 2. 学習済みモデルのロード
    try:
        # 使用したアルゴリズムに合わせてクラスをロードしてください (今回は SAC)
        model = SAC.load(model_path, env=env)
        print(f"✅ モデルのロードに成功しました")
    except Exception as e:
        print(f"❌ エラー: モデルのロードに失敗しました。ファイル形式やアルゴリズムが一致しているか確認してください。\n詳細: {e}")
        env.close()
        return

    # 3. 可視化ループの実行 (1エピソード)
    obs, _ = env.reset()
    total_reward = 0
    step_count = 0
    
    print("\n👀 シュミレーションを開始します (ESCキーでウィンドウを閉じると中断します)")
    
    try:
        # Humanoid-v5 は最大1000ステップで1エピソードが終了します
        for step in range(1000):
            # モデルに基づいて行動を選択
            # deterministic=True にすると、学習した確率分布の平均値（最も確実な行動）を選択します
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # 環境を1ステップ進める（内部で自動的にレンダリングが行われます）
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # エピソード終了条件 (転倒など)
            if terminated or truncated:
                print(f"🏁 エピソード終了 (ステップ数: {step_count})")
                break
                
    except KeyboardInterrupt:
        print("\n⏹️ ユーザーによって中断されました")
    except Exception as e:
        print(f"⚠️ 実行中にエラーが発生しました: {e}")
    
    # 4. クリーンアップ
    print(f"📊 トータル報酬: {total_reward:.2f}")
    env.close()
    print("=== 可視化を終了しました ===")

# --- 実行 ---
if __name__ == "__main__":
    # MODEL_PATH 変数を実際のモデルファイルのパスに書き換えてから実行してください
    enjoy_humanoid(MODEL_PATH, ENV_ID, deterministic=DETERMINISTIC_POLICY)
