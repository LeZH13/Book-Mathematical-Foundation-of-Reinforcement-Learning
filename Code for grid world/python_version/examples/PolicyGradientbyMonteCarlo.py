import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import gym
import matplotlib.pyplot as plt

ENABLE_ANIMATION = False  # Set to False to disable animation

# Create environment (CartPole-v1 is a classic control problem)
if ENABLE_ANIMATION:
    env = gym.make('CartPole-v1', render_mode='human')
else:
    env = gym.make('CartPole-v1')

# Policy: simple neural network with one hidden layer
class Policy:
    def __init__(self, obs_dim, act_dim, hidden_size=16, lr=1e-2):
        self.w1 = np.random.randn(obs_dim, hidden_size) * 0.01
        self.w2 = np.random.randn(hidden_size, act_dim) * 0.01
        self.lr = lr

    def forward(self, obs):
        h = np.tanh(np.dot(obs, self.w1))
        logits = np.dot(h, self.w2)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def update(self, grads):
        for param, grad in grads.items():
            setattr(self, param, getattr(self, param) + self.lr * grad)

def discount_rewards(rewards, gamma=0.99):
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = rewards[t] + gamma * running_add
        discounted[t] = running_add
    return discounted

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
policy = Policy(obs_dim, act_dim)

for episode in range(500):
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = np.array(obs)
    done = False
    observations, actions, rewards = [], [], []

    # Optionally render every 100 episodes for a few steps
    render_this_episode = (episode + 1) % 100 == 0

    while not done:
        if ENABLE_ANIMATION and render_this_episode:
            env.render()
        probs = policy.forward(obs)
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

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        obs = next_obs

    # Compute discounted rewards
    discounted = discount_rewards(rewards)
    discounted = (discounted - np.mean(discounted)) / (np.std(discounted) + 1e-8)

    # Policy gradient update
    grads = {'w1': np.zeros_like(policy.w1), 'w2': np.zeros_like(policy.w2)}
    for i in range(len(observations)):
        obs = observations[i]
        action = actions[i]
        probs = policy.forward(obs)
        dlog = -probs
        dlog[action] += 1
        h = np.tanh(np.dot(obs, policy.w1))
        grads['w2'] += np.outer(h, dlog) * discounted[i]
        dh = np.dot(policy.w2, dlog) * (1 - h ** 2)
        grads['w1'] += np.outer(obs, dh) * discounted[i]

    policy.update(grads)

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}: Total Reward = {sum(rewards)}")

    # Show rendered frames every 100 episodes
    # No need to show frames with matplotlib when using 'human' render mode

    # Store total reward for plotting
    if episode == 0:
        episode_rewards = []
    episode_rewards.append(sum(rewards))

env.close()

# Plot learning curve
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('REINFORCE: Episode Reward Progress')
plt.show()