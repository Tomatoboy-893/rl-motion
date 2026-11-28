#
# check_buffer_fragments.py
#
# 新ステップ0の後半：ステップ3で保存した 500k 時点のリプレイバッファの中に、
# 這い歩行より良いエピソードがどれだけ入っているかを数える。
#
# 目的:
#   教授の問い（バッファを捨てずに、教師あり的に抜け出せるか）の前提確認。
#   フィルタ付き行動クローニングは、バッファの中にある断片までしか真似
#   できない。最良のエピソードが這い水準（〜2,600）止まりなら、その介入の
#   天井は最初から這い水準だと分かる。逆に良い断片が残っていれば、介入に
#   進む根拠になる。
#
# 使い方:
#   python3 check_buffer_fragments.py
#   （./npz_logs_step3_halfcheetah/state/ の2つのバッファを読む）
#
import os
import pickle

import numpy as np

STATE_DIR = "./npz_logs_step3_halfcheetah/state"
ALGOS = ["pure_sac", "fixed_prior"]
THRESHOLDS = [2600, 3000, 4000, 5000, 7000]


def episode_returns(buffer):
    """SB3 ReplayBuffer からエピソードごとのリターンを復元する。"""
    rewards = np.asarray(buffer.rewards, dtype=np.float64)
    dones = np.asarray(buffer.dones, dtype=np.float64)
    # 形は (buffer_size, n_envs)。このランは n_envs=1 前提で潰す
    rewards = rewards.reshape(rewards.shape[0], -1)[:, 0]
    dones = dones.reshape(dones.shape[0], -1)[:, 0]
    if getattr(buffer, "full", False):
        order = np.concatenate([np.arange(buffer.pos, len(rewards)),
                                np.arange(0, buffer.pos)])
    else:
        order = np.arange(0, buffer.pos)
    returns = []
    cur = 0.0
    for i in order:
        cur += rewards[i]
        if dones[i] > 0.5:
            returns.append(cur)
            cur = 0.0
    return np.array(returns)


def main():
    for algo in ALGOS:
        path = os.path.join(
            STATE_DIR, f"{algo}_halfcheetah_seed8_state_500k_replay_buffer.pkl")
        if not os.path.exists(path):
            print(f"[{algo}] バッファがありません: {path}")
            continue
        with open(path, "rb") as f:
            buffer = pickle.load(f)
        rets = episode_returns(buffer)
        print(f"\n===== {algo} (seed 8, 500k buffer) =====")
        print(f"episodes = {len(rets)}")
        if len(rets) == 0:
            print("エピソードが復元できませんでした")
            continue
        print(f"mean = {rets.mean():.1f} | median = {np.median(rets):.1f} "
              f"| max = {rets.max():.1f}")
        top = np.sort(rets)[::-1]
        print("top 10 episode returns: "
              + ", ".join(f"{v:.0f}" for v in top[:10]))
        for k, frac in [(10, "top10"), (0.01, "top1%"), (0.05, "top5%")]:
            if isinstance(k, int):
                sel = top[:k]
            else:
                sel = top[:max(1, int(len(top) * frac))]
            print(f"{frac} mean = {sel.mean():.1f} (n={len(sel)})")
        for th in THRESHOLDS:
            print(f"episodes >= {th}: {(rets >= th).sum()}")


if __name__ == "__main__":
    main()
