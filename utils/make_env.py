import gymnasium as gym
import ale_py
import numpy as np
from collections import deque
import cv2

class ImagePreprocessingWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84), grayscale=True, num_stack=4):
        super().__init__(env)
        self.shape = shape
        self.grayscale = grayscale
        self.num_stack = num_stack
        self.frames = deque([], maxlen=num_stack)

        channels = 1 if grayscale else 3
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shape[0], shape[1], channels * num_stack),
            dtype=np.uint8
        )

    def observation(self, obs):
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

        obs = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)

        if self.grayscale:
            obs = np.expand_dims(obs, axis=-1)

        self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed = self.observation(obs)
        for _ in range(self.num_stack - 1):
            self.frames.append(np.zeros_like(processed[..., :1]))
        return self.observation(obs), info


def make_env(render=False, shape=(84, 84), grayscale=True, num_stack=4):
    gym.register_envs(ale_py)
    render_mode = "human" if render else "rgb_array"
    env = gym.make("ALE/VideoPinball-v5", render_mode=render_mode)
    env = ImagePreprocessingWrapper(env, shape=shape, grayscale=grayscale, num_stack=num_stack)
    return env


# Example of self-play interaction for MuZero
if __name__ == '__main__':
    env = make_env(render=False)
    replay_buffer = []

    obs, info = env.reset()

    for step in range(1000):
        # Here should be: action = mcts.select_action(obs)
        action = env.action_space.sample()

        next_obs, reward, terminated, truncated, info = env.step(action)

        # Save to replay buffer
        replay_buffer.append((obs, action, reward))

        obs = next_obs

        if step == 100:
            import matplotlib.pyplot as plt
            for i in range(obs.shape[-1]):
                plt.subplot(1, obs.shape[-1], i + 1)
                plt.imshow(obs[..., i], cmap='gray')
                plt.title(f"Frame {i+1}")
                plt.axis('off')
            plt.suptitle("Frame stack at step 100")
            plt.show()

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
