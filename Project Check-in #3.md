# **Title: Learning to Play Pinball with MuZero**

---

## **Introduction**  
We aim to implement a reinforcement learning model capable of learning to play digital pinball using the MuZero algorithm. This project explores how an agent can learn an effective strategy in a complex, dynamic environment where understanding physical interactions and long-term planning are necessary for a strong result.

MuZero, developed by DeepMind, is a model-based RL algorithm that learns a policy and value function while also learning an internal model of the environment. Unlike traditional model-based approaches, MuZero does not require access to the true environment dynamics and instead learns to predict rewards, values, and policies directly from observation histories.

We selected this paper because of its breakthrough performance across a variety of environments (e.g., Atari, Go, Chess) and its potential application to real-time, physics-based games like pinball. Our problem is in the category of **Reinforcement Learning**, with a focus on decision-making in continuous and discrete action spaces.

---

## **Challenges**  
The most difficult challenge has been managing the time and compute resources required for training a MuZero agent, especially in an environment as complex and chaotic as pinball. Training even a simplified version of MuZero is computationally expensive, and long training cycles made iteration and debugging slower than expected. Moreover, integrating Monte Carlo Tree Search (MCTS) with learned dynamics and ensuring meaningful self-play data posed additional hurdles. Creating a modular and testable codebase helped alleviate some of this pain, but tuning MuZero still remains a nontrivial task.

---

## **Insights**  
At this point in the project, we have achieved several important milestones. The core MuZero system — including the environment interface, replay buffer, trainer, and neural networks — is functioning correctly. Training runs without crashing, and loss values across reward, value, and policy objectives show a clear downward trend after the first 10 epochs. By epoch 60, training loss generally stabilizes below 1.0, despite some fluctuations, which is consistent with noisy data from random self-play. These quantitative signals confirm that our implementation is learning effectively, even before incorporating MCTS.

Qualitatively, our agent is not yet playing well due to the absence of planning in self-play. However, the infrastructure is robust and ready for further optimization.

---

## **Plan**  
We are largely on track with our initial goals. The implementation of the MuZero pipeline is complete, and we have already begun experimenting with basic training loops. That said, we still need to improve our MCTS implementation to unlock the full potential of the model. Our immediate next steps include adding Dirichlet exploration noise, softmax temperature tuning, and improving the MCTS rollout strategy.

Due to the high compute demands of Pinball, we have been validating our architecture on a simpler game — Breakout — which has enabled us to test learning behavior more quickly. Once validated, we will deploy long-running training jobs for Pinball on the Oscar computing cluster.

Looking ahead, we will focus our time on fine-tuning hyperparameters, integrating MCTS, and tracking performance metrics like average episode score and duration. While we don’t anticipate major design changes, the addition of MCTS will be critical to closing the performance gap between random and learned policies.