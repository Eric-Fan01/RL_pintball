# run.py

import torch

from utils.make_env import make_env
from utils.replay_buffer import ReplayBuffer
from utils.self_play import run_selfplay
from trainer import run_training
from muzero_network import MuZeroNet  # 你的MuZero网络
import numpy as np
# ========================
# 超参数（可以放到config.py后面更好管理）
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_space_size = 9   # VideoPinball动作数量
support_size = 21       # value支持集大小

num_epochs = 1000
selfplay_steps_per_epoch = 500
train_batch_size = 8  # train batch size shall be 32 or what, but here we use 4 for test
rollout_steps = 5
buffer_capacity = 10000
max_trajectory_len = 300

# ========================
# 初始化
# ========================
env = make_env(render=False)
network = MuZeroNet(action_space_size=action_space_size, support_size=support_size).to(device)
buffer = ReplayBuffer(capacity=buffer_capacity)

# mcts_fn 先用 random placeholder（替换）
def random_action_fn(network, obs):
    return np.random.randint(action_space_size)

# ========================
# 正式训练循环
# ========================
for epoch in range(1, num_epochs + 1):
    print(f"\n=== Epoch {epoch} ===")

    # --- Self-Play generate data ---
    run_selfplay(
        env=env,
        network=network,
        buffer=buffer,
        mcts_fn=random_action_fn,
        num_steps=selfplay_steps_per_epoch,
        max_trajectory_len=max_trajectory_len,
        device=device
    )

    # --- Trainer 从 buffer 训练 ---
    run_training(
        network=network,
        buffer=buffer,
        batch_size=train_batch_size,
        rollout_steps=rollout_steps,
        device=device,
        action_space_size=action_space_size,
        support_size=support_size
    )

    # --- 定期保存 buffer ---
    if epoch % 10 == 0:
        buffer.save()

env.close()
print("Training finished ✅")
