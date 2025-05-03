import numpy as np
import torch

def run_selfplay(env, network, buffer, mcts_fn, num_steps=1000, max_trajectory_len=300, device="cuda"):
    """
    Run self-play to collect trajectories.

    Args:
        env: The gym environment.
        network: MuZero network (representation, dynamics, prediction)
        buffer: ReplayBuffer instance to store trajectories
        mcts_fn: function (network, obs) -> action (int)
        num_steps: total interaction steps
        max_trajectory_len: max length of one trajectory
        device: torch device
    """

    obs, info = env.reset()
    traj = []

    for step in range(num_steps):
        # === Select action by mcts_fn ===
        obs_input = torch.tensor(obs.transpose(2, 0, 1), dtype=torch.float32, device=device).unsqueeze(0) / 255.0
        action = mcts_fn(network, obs_input)

        next_obs, reward, terminated, truncated, info = env.step(action)

        traj.append((obs.transpose(2, 0, 1), action, reward, terminated))

        obs = next_obs

        if terminated or truncated or len(traj) >= max_trajectory_len:
            buffer.add(traj)
            print(f"[Self-Play] Added trajectory with {len(traj)} steps. Buffer size: {len(buffer)}")
            obs, info = env.reset()
            traj = []
