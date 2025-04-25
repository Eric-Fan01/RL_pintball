# trainer/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def run_training(network, buffer, batch_size=32, rollout_steps=5, device="cuda", action_space_size=9, support_size=21):
    """
    Sample from replay buffer and update the network.

    Args:
        network: MuZero network (representation, dynamics, prediction)
        buffer: ReplayBuffer instance
        batch_size: number of trajectories to sample
        rollout_steps: steps to rollout into future
        device: torch device
        action_space_size: total number of actions
        support_size: value distribution output size
    """

    if len(buffer) < batch_size:
        print("[Trainer] Buffer too small, skip training.")
        return

    batch = buffer.sample(batch_size)

    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    network.train()

    total_loss = 0.0

    for traj in batch:
        if len(traj) < rollout_steps:
            continue

        # === 从轨迹中随机抽一个起点
        start_idx = np.random.randint(0, len(traj) - rollout_steps + 1)
        rollout = traj[start_idx:start_idx + rollout_steps]

        # === 解析 rollout
        obs_seq = [step[0] for step in rollout]   # (C,H,W)
        actions = [step[1] for step in rollout]
        rewards = [step[2] for step in rollout]

        obs_seq = torch.tensor(np.array(obs_seq), dtype=torch.float32, device=device) / 255.0
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        # === Step 1: Representation
        state = network.representation(obs_seq[0].unsqueeze(0))  # obs[0]

        predicted_rewards = []
        predicted_values = []
        predicted_policies = []

        # === Step 2: Dynamics and Prediction
        for t in range(rollout_steps):
            policy_logits, value = network.prediction(state)
            predicted_policies.append(policy_logits)
            predicted_values.append(value.squeeze())

            if t < rollout_steps - 1:
                # Create action plane for Dynamics
                action_plane = torch.ones((1, 1, state.shape[2], state.shape[3]), device=device) * (actions[t].item() / action_space_size)
                reward, next_state = network.dynamics(state, action_plane)
                predicted_rewards.append(reward.squeeze())
                state = next_state.detach()  # important to detach

        # # === Step 3: Loss calculation
        # loss_fn = nn.MSELoss()

        # rewards_target = rewards[:-1]  # rewards[0] ~ rewards[T-2]
        # rewards_pred = torch.stack(predicted_rewards)

        # values_target = rewards.sum().expand_as(predicted_values[0])  # 简化版，用总reward近似value
        # values_pred = torch.stack(predicted_values)

        # policy_target = actions[1:]  # 下一个动作
        # policy_pred = torch.stack(predicted_policies[:-1])

        # # reward loss
        # reward_loss = loss_fn(rewards_pred, rewards_target)

        # # value loss
        # value_loss = loss_fn(values_pred, values_target)

        # # policy loss (cross entropy)
        # policy_loss_fn = nn.CrossEntropyLoss()
        # policy_loss = policy_loss_fn(policy_pred, policy_target)

        # loss = reward_loss + value_loss + policy_loss




        loss_fn = nn.MSELoss()

        rewards_target = rewards[1:]  # ✅ 正确，rewards[1:]
        rewards_pred = torch.stack(predicted_rewards)

        values_target = rewards.sum().expand_as(predicted_values[0])
        values_pred = torch.stack(predicted_values)

        policy_target = actions[1:]  # ✅ 保持不变
        policy_pred = torch.stack(predicted_policies)  # ✅ 全部 predicted_policies，不要切[:-1]

        # reward loss
        reward_loss = loss_fn(rewards_pred, rewards_target)

        # value loss
        value_loss = loss_fn(values_pred, values_target)

        # policy loss
        policy_loss_fn = nn.CrossEntropyLoss()
        policy_loss = policy_loss_fn(policy_pred, policy_target)

        loss = reward_loss + value_loss + policy_loss



        # === Step 4: Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / batch_size
    print(f"[Trainer] Training done, avg total loss: {avg_loss:.6f}")
