# run.py

import torch
import numpy as np
import datetime
import os

from utils.make_env import make_env
from utils.replay_buffer import ReplayBuffer
from utils.self_play import run_selfplay
from trainer import run_training
from muzero_network import MuZeroNet
from utils.mcts import mcts_search

# ========================
# 超参
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_space_size = 4   # VideoPinball动作数量（Breakout改成4）
support_size = 21

num_epochs = 1000
selfplay_steps_per_epoch = 500
train_batch_size = 8
rollout_steps = 5
buffer_capacity = 10000
max_trajectory_len = 300

# 创建日志目录
os.makedirs("./logs", exist_ok=True)
logfile = open("./logs/train_log.txt", "a") 
latest_checkpoint = "./checkpoints/muzero_checkpoint_latest.pth"

# ========================
# 初始化
# ========================
env = make_env(game='Breakout',render=False)
network = MuZeroNet(action_space_size=action_space_size, support_size=support_size).to(device)
buffer = ReplayBuffer(capacity=buffer_capacity)

# MCTS
def mcts_fn(network, obs):
    return mcts_search(network, obs, num_simulations=50, action_space_size=action_space_size)


for epoch in range(1, num_epochs + 1):
    print(f"\n=== Epoch {epoch} ===")

    # --- Self-Play ---
    run_selfplay(
        env=env,
        network=network,
        buffer=buffer,
        mcts_fn=mcts_fn,
        num_steps=selfplay_steps_per_epoch,
        max_trajectory_len=max_trajectory_len,
        device=device
    )

    # --- Training ---
    avg_loss = run_training(
        network=network,
        buffer=buffer,
        batch_size=train_batch_size,
        rollout_steps=rollout_steps,
        device=device,
        action_space_size=action_space_size,
        support_size=support_size
    )

    # --- Logging ---
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logfile.write(f"[{now}] Epoch {epoch} - avg_loss: {avg_loss:.6f}\n")
    logfile.flush()

    # --- 保存buffer ---
    if epoch % 10 == 0:
        buffer.save()

    # --- 保存模型 ---
        checkpoint = {
            'epoch': epoch,
            'network_state_dict': network.state_dict(),
        }
        torch.save(checkpoint, f'./checkpoints/muzero_checkpoint_epoch{epoch}.pth')
        torch.save(checkpoint, latest_checkpoint)  # 每次也保存一份最新的
        print(f"[Checkpoint] Saved at epoch {epoch} ✅")


env.close()
logfile.close()
print("Training finished ✅")
