import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import gymnasium as gym
import matplotlib.pyplot as plt

ENABLE_ANIMATION = True  # Set to True to enable animation

if ENABLE_ANIMATION:
    env = gym.make('CartPole-v1', render_mode='human')
else:
    env = gym.make('CartPole-v1')

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

class Policy:
    def __init__(self, obs_dim, act_dim, hidden_size=32, lr=3e-3):
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

class Value:
    def __init__(self, obs_dim, hidden_size=32, lr=3e-3):
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

policy = Policy(obs_dim, act_dim)
value_fn = Value(obs_dim)

gamma = 0.99
clip_epsilon = 0.2
episode_rewards = []
batch_size = 32

def compute_advantages(rewards, values, next_values, gamma):
    advantages = []
    for r, v, nv in zip(rewards, values, next_values):
        advantages.append(r + gamma * nv - v)
    return np.array(advantages)

best_reward = -float('inf')
best_w1 = None
best_w2 = None

for episode in range(2000):
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = np.array(obs)
    done = False
    total_reward = 0

    observations, actions, rewards, old_probs, values = [], [], [], [], []

    render_this_episode = ENABLE_ANIMATION and ((episode + 1) % 100 == 0)

    while not done:
        if render_this_episode:
            env.render()
        probs, h_policy = policy.forward(obs)
        value, h_value = value_fn.forward(obs)
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
        old_probs.append(probs[action])
        values.append(value)

        obs = next_obs
        total_reward += reward

    # Compute next state values for advantage
    next_values = values[1:] + [0]
    advantages = compute_advantages(rewards, values, next_values, gamma)
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    # PPO update (single epoch for simplicity)
    for i in range(len(observations)):
        obs = observations[i]
        action = actions[i]
        adv = advantages[i]
        old_prob = old_probs[i]
        value, h_value = value_fn.forward(obs)
        probs, h_policy = policy.forward(obs)
        prob = probs[action]

        # Policy update (clipped surrogate objective)
        ratio = prob / (old_prob + 1e-8)
        clipped_ratio = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        policy_loss = -min(ratio * adv, clipped_ratio * adv)

        dlog = -probs
        dlog[action] += 1
        policy_grads = {
            'w2': np.outer(h_policy, dlog) * policy_loss,
            'w1': np.outer(obs, np.dot(policy.w2, dlog) * (1 - h_policy ** 2) * policy_loss)
        }
        policy.update(policy_grads)

        # Value update (MSE loss)
        value_loss = (value - (rewards[i] + gamma * next_values[i])) ** 2
        value_grads = {
            'w2': h_value[:, None] * value_loss,
            'w1': np.outer(obs, np.dot(value_fn.w2.flatten(), value_loss) * (1 - h_value ** 2))
        }
        value_fn.update(value_grads)

    episode_rewards.append(total_reward)
    if total_reward > best_reward:
        best_reward = total_reward
        best_w1 = policy.w1.copy()
        best_w2 = policy.w2.copy()
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}: Total Reward = {total_reward}")

env.close()

plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('PPO: Episode Reward Progress')
plt.show()

# Deploy best policy and render in pygame
print(f"\nDeploying best policy with reward {best_reward:.2f}...")
deploy_env = gym.make('CartPole-v1', render_mode='human')
obs = deploy_env.reset()
if isinstance(obs, tuple):
    obs = obs[0]
obs = np.array(obs)
done = False

policy.w1 = best_w1
policy.w2 = best_w2

step_count = 0
max_steps = 500  # ~10 seconds at 50 FPS

while not done and step_count < max_steps:
    deploy_env.render()
    probs, _ = policy.forward(obs)
    action = np.argmax(probs)
    step_result = deploy_env.step(action)
    if isinstance(step_result, tuple) and len(step_result) == 5:
        next_obs, reward, terminated, truncated, _ = step_result
        done = terminated or truncated
    else:
        next_obs, reward, done, _ = step_result
    if isinstance(next_obs, tuple):
        next_obs = next_obs[0]
    obs = np.array(next_obs)
    step_count += 1

deploy_env.close()