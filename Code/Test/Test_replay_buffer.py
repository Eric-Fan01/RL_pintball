# test.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import numpy as np

# === 加载已有的 replay buffer ===
buffer = ReplayBuffer()
buffer.load()

print(f"\n✅ Loaded {len(buffer)} trajectories from replay buffer.")

# === 打印每条 trajectory 的前几步内容 ===
for traj_id, traj in enumerate(buffer.buffer):
    print(f"\n🧾 Trajectory {traj_id + 1}: {len(traj)} steps")

    for t, (obs, action, reward, done) in enumerate(traj[:300]):  # 只看前5步
        if reward > 0:
            print(f" Step {t}: ,reward: {reward:.2f}")
        # print(f" Step {t}:")
        # print(f"   obs shape: {obs.shape}, dtype: {obs.dtype}")
        # print(f"   action: {action}")
        # print(f"   reward: {reward}")
        # print(f"   done: {done}")

    # 只可视化第一条轨迹的第一帧
    if traj_id == 0:
        obs = traj[150][0]  # 第150步的 obs
        if isinstance(obs, np.ndarray):
            print("\n🖼️ Visualizing first observation (shape CHW):", obs.shape)
            for i in range(obs.shape[0]):
                plt.subplot(1, obs.shape[0], i + 1)
                plt.imshow(obs[i], cmap='gray')
                plt.title(f"Channel {i}")
                plt.axis('off')
            plt.suptitle("First Observation in Trajectory 1")
            plt.show()
    break  # 只看一条 trajectory
