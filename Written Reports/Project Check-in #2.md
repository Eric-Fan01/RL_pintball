# **Title: Learning to Play Pinball with MuZero**

---

## **Who**  
- Guanghe (Eric) Fan 
- Hatem Mohamed 
- Shivam Hingorani (shingor2)

---

## **Introduction**  
We aim to implement a reinforcement learning model capable of learning to play digital pinball using the MuZero algorithm. This project explores how an agent can learn an effective strategy in a complex, dynamic environment where understanding physical interactions and long-term planning are necessary for a strong result.

MuZero, developed by DeepMind, is a model-based RL algorithm that learns a policy and value function while also learning an internal model of the environment. Unlike traditional model-based approaches, MuZero does not require access to the true environment dynamics and instead learns to predict rewards, values, and policies directly from observation histories.

We selected this paper because of its breakthrough performance across a variety of environments (e.g., Atari, Go, Chess) and its potential application to real-time, physics-based games like pinball. Our problem is in the category of **Reinforcement Learning**, with a focus on decision-making in continuous and discrete action spaces.

---

## **Related Work**  
One relevant piece of work is the blog post and open-source implementation by **Juliette Noutahi**, titled *"MuZero Explained"*. It provides a high-level breakdown and a simplified re-implementation of the MuZero algorithm tailored for small-scale environments:  
🔗 https://juliette.mhorm.io/muzero  

Another helpful resource is the clean and well-documented open-source repo:  
🔗 https://github.com/werner-duvaud/muzero-general  

We are also drawing insights from the DeepMind MuZero paper:  
🔗 https://arxiv.org/abs/1911.08265  

---

## **Data**  
Since MuZero doesn’t require a fixed dataset but instead learns through interaction with the environment, data is generated dynamically through self-play. Thus, the scale of our "data" is defined by the number of episodes and timesteps run during training.  

Some preprocessing may be needed to normalize observations (e.g., screen pixels, paddle angles, ball positions) and structure the input appropriately for the network.

---

## **Methodology**  
We are implementing a scaled-down version of the MuZero architecture consisting of three key components:
1. **Representation Network** – Converts observations into hidden state.
2. **Dynamics Network** – Predicts next hidden state and reward given current state and action.
3. **Prediction Network** – Outputs policy and value based on current hidden state.

Training will follow the original MuZero procedure using Monte Carlo Tree Search (MCTS) for planning and improving action selection.

**Hardest part:** Integrating MCTS with learned dynamics and ensuring stability in environments with chaotic dynamics like pinball. Additionally, managing computational constraints when training over long episodes will be a challenge.

**Backup ideas:**  
- Replace MCTS with a simpler planning strategy for initial iterations.  
- Use a smaller network or frame-skipping to reduce training time.

---

## **Metrics**  
Success will be evaluated using the following metrics:  
- **Average Score per Episode**: Primary performance metric (reward signal).  
- **Episode Length**: Proxy for skill level (longer episodes = better survival).  
- **Learning Curve**: Rate of improvement over training time.  

We will compare these metrics to baseline RL agents (e.g., DQN or PPO without planning).  

**Goals:**  
- **Base:** Agent learns non-random behavior and achieves consistent reward.  
- **Target:** Agent outperforms a standard model-free RL agent.  
- **Stretch:** Agent can play competitively across different pinball layouts without re-training.

---

## **Ethics**  
**Why is Deep Learning a good approach to this problem?**  
Deep learning enables agents to process complex visual and temporal information needed in dynamic environments like pinball. Traditional rule-based or physics-driven systems would require significant manual engineering and wouldn't generalize across levels. MuZero’s learned world model offers a scalable and adaptive approach.

**What are the consequences of mistakes made by your algorithm?**  
In our case, the consequences are limited to gameplay performance. However, this work could extend to physical systems or robotics, where inaccuracies in learned world models could lead to dangerous behavior. Ensuring safety and interpretability in such future applications would be critical.

---

## **Division of Labor**  
- **Eric Fan**: MuZero architecture implementation, training pipeline, MCTS integration  
- **Hatem Mohamed**: Environment setup, agent-environment interface, data logging/visualization  
- **Shivam Hingorani**: Evaluation, experiments, documentation, and presentation  

We will collaborate closely during testing and debugging phases to ensure robustness across all components.
