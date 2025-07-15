import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
import gym
import matplotlib.pyplot as plt

ENABLE_ANIMATION = False  # Set to True to enable animation

if ENABLE_ANIMATION:
    env = gym.make('Pendulum-v1', render_mode='human')
else:
    env = gym.make('Pendulum-v1')

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_low = env.action_space.low
act_high = env.action_space.high

class Actor:
    def __init__(self, obs_dim, act_dim, hidden_size=32, lr=1e-3):
        self.w1 = np.random.randn(obs_dim, hidden_size) * 0.01
        self.w2 = np.random.randn(hidden_size, act_dim) * 0.01
        self.lr = lr

    def forward(self, obs):
        h = np.tanh(np.dot(obs, self.w1))
        action = np.tanh(np.dot(h, self.w2))  # Output in [-1, 1]
        return action, h

    def update(self, grad_w1, grad_w2):
        self.w1 += self.lr * grad_w1
        self.w2 += self.lr * grad_w2

class Critic:
    def __init__(self, obs_dim, act_dim, hidden_size=32, lr=1e-3):
        self.w1 = np.random.randn(obs_dim + act_dim, hidden_size) * 0.01
        self.w2 = np.random.randn(hidden_size, 1) * 0.01
        self.lr = lr

    def forward(self, obs, action):
        x = np.concatenate([obs, action])
        h = np.tanh(np.dot(x, self.w1))
        q = np.dot(h, self.w2).item()
        return q, h

    def update(self, grad_w1, grad_w2):
        self.w1 += self.lr * grad_w1
        self.w2 += self.lr * grad_w2

actor = Actor(obs_dim, act_dim)
critic = Critic(obs_dim, act_dim)

gamma = 0.99
episode_rewards = []

for episode in range(200):
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = np.array(obs)
    done = False
    total_reward = 0

    render_this_episode = ENABLE_ANIMATION and ((episode + 1) % 50 == 0)

    while not done:
        if render_this_episode:
            env.render()
        action, h_actor = actor.forward(obs)
        # Add exploration noise
        action = action + np.random.normal(0, 0.1, size=act_dim)
        action = np.clip(action, act_low, act_high)
        step_result = env.step(action)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            next_obs, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, _ = step_result
        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]
        next_obs = np.array(next_obs)

        # Critic update
        q, h_critic = critic.forward(obs, action)
        next_action, _ = actor.forward(next_obs)
        q_next, _ = critic.forward(next_obs, next_action)
        target = reward + gamma * q_next * (not done)
        td_error = target - q

        grad_w2 = h_critic[:, None] * td_error
        grad_w1 = np.outer(np.concatenate([obs, action]), np.dot(critic.w2.flatten(), td_error) * (1 - h_critic ** 2))
        critic.update(grad_w1, grad_w2)

        # Actor update (policy gradient)
        # For simplicity, use the gradient of Q w.r.t. action as the update direction
        action_for_grad, h_actor = actor.forward(obs)
        q_val, _ = critic.forward(obs, action_for_grad)
        grad_action = np.ones_like(action_for_grad)  # Placeholder for dQ/da, not exact
        grad_w2_a = h_actor[:, None] * grad_action
        grad_w1_a = np.outer(obs, np.dot(actor.w2, grad_action) * (1 - h_actor ** 2))
        actor.update(grad_w1_a, grad_w2_a)

        obs = next_obs
        total_reward += reward

    episode_rewards.append(total_reward)
    if (episode + 1) % 20 == 0:
        print(f"Episode {episode+1}: Total Reward = {total_reward}")

env.close()

plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DDPG: Episode Reward Progress')
plt.show()