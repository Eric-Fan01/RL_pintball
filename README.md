# RL_pintball

## In the Early Stage?

Because:

- Early-stage agents are very weak.
- Their actions are almost random or poorly guided by shallow search.
- In some environments (e.g., Breakout, Pinball), if you don't enforce a cutoff, the ball can keep bouncing for thousands of steps without ending.
- This causes each trajectory to be extremely long.
- The replay buffer gets filled with a few ultra-long trajectories, making training data very sparse.
- **As a result, the agent learns almost nothing!**

✅ Therefore, during the early stage of training, **an artificial `max_trajectory_len = 300` cutoff is absolutely necessary**.

---

## Remove or Relax `max_trajectory_len` Later.

Because:

- In the mid-to-late stage of training, the agent learns how to play properly.
- Its behavior becomes smarter and more stable.
- Each episode can naturally terminate (e.g., ball missed, life lost) without intervention.
- If you still force a cutoff, it might interrupt a naturally completed trajectory.
- This breaks the learning signals (rewards, done flags), harming long-term strategy learning.

✅ Therefore, in the later stage, **trajectories should end naturally (`terminated=True`)**  
✅ **and no forced `truncated=True` cutoff should be used anymore.**


---

# 📄 MuZero Lightweight Implementation: Network, MCTS, and Simplifications

## 1. Overview

This project implements a simplified version of the MuZero algorithm, focusing on reducing computational complexity while maintaining reasonable performance.  
It is designed to be trainable even on machines with **very limited resources (single GPU or CPU)**.

The core components include:

- A lightweight **Convolutional Neural Network (CNN)** for state representation
- A minimal **MuZero-style Recurrent MCTS** for planning
- A replay buffer for self-play data

---

## 2. Network Architecture (CNN)

### Representation Network

- Input: Stacked frames (shape: `[Batch, 4, 84, 84]`)
- 2D Convolutional layers followed by BatchNorm and ReLU
- A lightweight down-sampling block reduces the feature map to `[Batch, 16, 21, 21]`
- Residual blocks are used to enhance feature representation without heavy stacking.

### Dynamics Network

- Takes the internal latent state and the action (as an additional channel).
- Predicts:
  - The next latent state
  - The immediate reward

### Prediction Network

- Given a latent state, predicts:
  - A policy distribution over actions
  - A value distribution over discrete support bins (21 bins from -10 to 10)

✅ The model is **small and fast**, allowing efficient backpropagation even with a small batch size (e.g., 8).

---

## 3. MCTS Planning

- A minimal Monte Carlo Tree Search (MCTS) is used.
- **Only 50 simulations per move** (instead of 800 in original MuZero).
- Root node exploration is performed, but no Dirichlet noise or temperature-based sampling is added initially.
- Value backups are direct (without discount factors).

✅ This greatly reduces search time, making Self-Play **feasible without expensive GPUs**.

---

## 4. Simplifications Compared to Original MuZero

| Component | Original MuZero | Our Lightweight Version | Why Simplified |
|:---|:---|:---|:---|
| Number of MCTS simulations | ~800 per move | 50 per move | Much faster inference |
| Network depth | Deep ResNet | Small CNN + Residual blocks | Lower memory and computation |
| Support for reward/value | Softmax distribution decoding | Simple expectation over 21 bins | Easier to implement |
| Tree Policy Improvement | Dirichlet Noise + Temperature sampling | Simple argmax without noise | Good enough for early experiments |
| Replay buffer | Priority sampling | FIFO queue | Simpler management |

---

## 5. Motivation for Simplification

- Original MuZero requires **massive compute** (e.g., 32 TPUs).
- In real-world small projects or academic exercises, such resources are unavailable.
- Our goal:  
  **"Reproduce core MuZero behavior on a standard laptop or low-end GPU"**.

✅ With these simplifications, it becomes possible to **train functional models even with limited hardware**, while still observing meaningful self-improvement behaviors.

---

## 6. Performance

- In Breakout, the model initially plays randomly (short episodes).
- Over time, episodes become longer (agent learns to survive longer without dying).
- Average loss shows a decreasing trend, although not perfectly smooth (expected in RL settings).
- The agent is able to **stabilize gameplay** after training on a few thousand steps.

✅ **Results are acceptable given the hardware constraints.**

---

## 7. Conclusion

This lightweight MuZero variant demonstrates that:

- **Core ideas** of MuZero (planning, representation learning, value/policy prediction)  
- **Can be implemented and trained efficiently** even with significant simplifications.

While it does not achieve the superhuman performance of the full MuZero paper,  
✅ it **successfully captures the essential spirit** of MuZero:

- Learning to plan
- Learning to predict rewards and values
- Improving purely through self-play

Future improvements could include:

- Adding Dirichlet noise for better exploration
- Increasing MCTS simulation count
- Fine-tuning network depth and width

---

