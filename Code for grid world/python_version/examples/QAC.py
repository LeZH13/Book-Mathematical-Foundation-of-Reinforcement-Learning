import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import gym
import matplotlib.pyplot as plt

ENABLE_ANIMATION = False  # Set to True to enable animation

if ENABLE_ANIMATION:
    env = gym.make('CartPole-v1', render_mode='human')
else:
    env = gym.make('CartPole-v1')

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

class Actor:
    def __init__(self, obs_dim, act_dim, hidden_size=16, lr=1e-2):
        self.w1 = np.random.randn(obs_dim, hidden_size) * 0.01
        self.w2 = np.random.randn(hidden_size, act_dim) * 0.01
        self.lr = lr

    def forward(self, obs):
        h = np.tanh(np.dot(obs, self.w1))
        logits = np.dot(h, self.w2)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs, h

    def update(self, grads):
        for param, grad in grads.items():
            setattr(self, param, getattr(self, param) + self.lr * grad)

class Critic:
    def __init__(self, obs_dim, hidden_size=16, lr=1e-2):
        self.w1 = np.random.randn(obs_dim, hidden_size) * 0.01
        self.w2 = np.random.randn(hidden_size, 1) * 0.01
        self.lr = lr

    def forward(self, obs):
        h = np.tanh(np.dot(obs, self.w1))
        value = np.dot(h, self.w2).item()
        return value, h

    def update(self, grads):
        for param, grad in grads.items():
            setattr(self, param, getattr(self, param) + self.lr * grad)

actor = Actor(obs_dim, act_dim)
critic = Critic(obs_dim)

gamma = 0.99
episode_rewards = []

for episode in range(500):
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = np.array(obs)
    done = False
    total_reward = 0

    render_this_episode = ENABLE_ANIMATION and ((episode + 1) % 100 == 0)

    while not done:
        if render_this_episode:
            env.render()
        probs, h_actor = actor.forward(obs)
        action = np.random.choice(act_dim, p=probs)
        step_result = env.step(action)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            next_obs, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, _ = step_result
        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]
        next_obs = np.array(next_obs)

        value, h_critic = critic.forward(obs)
        next_value, _ = critic.forward(next_obs)
        td_error = reward + gamma * next_value * (not done) - value

        # Actor update (policy gradient with advantage)
        dlog = -probs
        dlog[action] += 1
        actor_grads = {
            'w2': np.outer(h_actor, dlog) * td_error,
            'w1': np.outer(obs, np.dot(actor.w2, dlog) * (1 - h_actor ** 2) * td_error)
        }
        actor.update(actor_grads)

        # Critic update (TD error)
        critic_grads = {
            'w2': h_critic[:, None] * td_error,
            'w1': np.outer(obs, np.dot(critic.w2.flatten(), td_error) * (1 - h_critic ** 2))
        }
        critic.update(critic_grads)

        obs = next_obs
        total_reward += reward

    episode_rewards.append(total_reward)
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}: Total Reward = {total_reward}")

env.close()

plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('QAC (Advantage Actor-Critic): Episode Reward Progress')
plt.show()