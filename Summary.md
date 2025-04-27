# CSCI1470 Final



# MuZero Project Summary

This document summarizes the full MuZero project modules we've implemented so far. It can be used for your teammate to prepare posters or documentation.

------

# Overall Flow

```plaintext
make_env -> replay_buffer -> self_play -> trainer -> update network -> repeat self_play...
```

All parts are connected and basic training can run successfully.

------

# Project Structure

```
├── data/
├── Test/
│   ├── Test_Environment.py
│   ├── Test_model.py
│   ├── Test_replay_buffer.py
│   ├── Test_success.py
├── utils/
│   ├── make_env.py
│   ├── __pycache__/
|	├── network.py
|	├── replay_buffer.py
|	├── self_play.py
├── muzero_network.py
├── trainer.py
├── run.py
├── README.md
```

------

# Module Breakdown

## utils/make_env.py

- Create VideoPinball environment.
- Apply preprocessing:
  - Grayscale conversion.
  - Resize to 84x84.
  - Frame stacking (num_stack=4).

## replay_buffer.py

- ReplayBuffer to store trajectories from self-play.
- Supports add, sample, save (to ./data/trace.pkl), and load.

## self_play.py

- Play the environment and generate (obs, action, reward) sequences.
- Currently using random actions.
- Each trajectory is capped at 300 steps.

## trainer.py

- Sample batch trajectories from ReplayBuffer.
- Perform multi-step rollout (rollout_steps=5).
- Compute losses:
  - Reward loss (MSE).
  - Value loss (MSE).
  - Policy loss (CrossEntropy).
- Each trajectory is trained independently (optimizer.step per trajectory).

## network.py

- Defines core MuZero modules:
  - Representation network (CNN + Downsampling).
  - Dynamics network (CNN + Residual block).
  - Prediction network (policy and value heads).

## muzero_network.py

- Wraps the above three networks into one MuZeroNet class:
  - `.representation(obs)`
  - `.dynamics(state, action)`
  - `.prediction(state)`

## run.py

- Overall controller:
  - Initialize environment, buffer, network.
  - Self-play to collect data.
  - Train the network from buffer.
  - Save buffer every 10 epochs.

## Test/

- Unit tests for environment setup, model, buffer operations, and full system checks.

------

# Current Training Behavior

| Observation                                                  | Explanation                                       |
| ------------------------------------------------------------ | ------------------------------------------------- |
| Loss can be computed and training runs without crashing      | System glue is correct.                           |
| Loss values oscillate heavily between very large and very small | Data from random self-play is extremely noisy.    |
| No clear downward trend in loss yet                          | Need better self-play policy, not random actions. |

------

# Next Steps (Important)

1. **Implement mcts.py (MCTS search module)**
   - Node class, PUCT formula, expand, backup.
2. **Self-play must use MCTS search to select actions**
3. **Use MCTS visit counts as policy targets**
4. **Use discounted reward backup for value target**
5. **Visualize loss curve and average reward trend**

------

# Short Summary

> **Status:** Basic MuZero system pipeline is complete and running.
>
> **Next:** Need to upgrade self-play quality (MCTS) for effective learning.

------

# Notes

> This document can be used directly for poster preparation, project presentations, or progress reports. You can add diagrams or graphs if needed to illustrate the flow visually.