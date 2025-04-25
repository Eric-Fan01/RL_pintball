# buffer/replay_buffer.py

import random
import pickle
import os
from pathlib import Path

class ReplayBuffer:
    """
    Replay Buffer for storing trajectories.
    Each trajectory is a list of tuples (obs, action, reward, done).
    """
    def __init__(self, capacity=10000, save_path="./data/trace.pkl"):
        self.capacity = capacity
        self.buffer = []
        self.save_path = save_path

        # 自动创建目录(auto create directory)
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def add(self, trajectory):
        """Add a full trajectory: list of (obs, action, reward, done)"""
        self.buffer.append(trajectory)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        """Sample random trajectories"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def save(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.buffer, f)
        print(f"[ReplayBuffer] Saved {len(self.buffer)} trajectories to {self.save_path}")

    def load(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'rb') as f:
                self.buffer = pickle.load(f)
            print(f"[ReplayBuffer] Loaded {len(self.buffer)} trajectories from {self.save_path}")
        else:
            print(f"[ReplayBuffer] No saved buffer found at {self.save_path}")


# ========================
# ✅ 测试代码(test code)
# ========================
if __name__ == "__main__":
    from utils.make_env import make_env 
    
    buffer = ReplayBuffer(capacity=10)
    env = make_env(render=False)

    obs, info = env.reset()
    traj = []
    MAX_TRAJECTORY_LENGTH = 300

    for step in range(1000):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        traj.append((obs.transpose(2, 0, 1), action, reward, terminated))

        obs = next_obs

        if terminated or truncated or len(traj) >= MAX_TRAJECTORY_LENGTH:
            buffer.add(traj)
            print(f"[Self-Play Test] Added trajectory with {len(traj)} steps.")
            obs, info = env.reset()
            traj = []

    env.close()

    print(f"[Self-Play Test] Total collected {len(buffer)} trajectories.")

    buffer.save()

    # Reload to check
    new_buffer = ReplayBuffer()
    new_buffer.load()
    print(f"[Reload Test] Reloaded {len(new_buffer)} trajectories.")
