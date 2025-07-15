import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import gymnasium as gym
import matplotlib.pyplot as plt

ENABLE_ANIMATION = False  # Set to True to enable animation

NUM_ENVS = 8
def make_env():
    return gym.make('Acrobot-v1')
env = gym.vector.AsyncVectorEnv([make_env for _ in range(NUM_ENVS)])

obs_dim = env.single_observation_space.shape[0]
act_dim = env.single_action_space.n

class Policy:
    def __init__(self, obs_dim, act_dim, hidden_size=64, lr=1e-3):
        self.w1 = np.random.randn(obs_dim, hidden_size) * 0.01
        self.w2 = np.random.randn(hidden_size, act_dim) * 0.01
        self.lr = lr

    def forward(self, obs):
        h = np.tanh(np.dot(obs, self.w1))
        logits = np.dot(h, self.w2)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probs, h

    def update(self, grad_w1, grad_w2):
        self.w1 += self.lr * grad_w1
        self.w2 += self.lr * grad_w2

class Value:
    def __init__(self, obs_dim, hidden_size=64, lr=1e-3):
        self.w1 = np.random.randn(obs_dim, hidden_size) * 0.01
        self.w2 = np.random.randn(hidden_size, 1) * 0.01
        self.lr = lr

    def forward(self, obs):
        h = np.tanh(np.dot(obs, self.w1))
        value = np.dot(h, self.w2)
        if value.shape[-1] == 1:
            value = value.squeeze(-1)
        if value.shape == ():
            value = float(value)
        return value, h

    def update(self, grad_w1, grad_w2):
        self.w1 += self.lr * grad_w1
        self.w2 += self.lr * grad_w2

policy = Policy(obs_dim, act_dim)
value_fn = Value(obs_dim)

gamma = 0.99
clip_epsilon = 0.2
episode_rewards = []
best_reward = -float('inf')
best_w1 = None
best_w2 = None

def compute_advantages(rewards, values, next_values, gamma):
    advantages = []
    for r, v, nv in zip(rewards, values, next_values):
        advantages.append(r + gamma * nv - v)
    return np.array(advantages)

for episode in range(1000):
    obs, _ = env.reset()
    obs = np.array(obs)
    dones = np.zeros(NUM_ENVS, dtype=bool)
    total_rewards = np.zeros(NUM_ENVS)

    observations, actions, rewards, old_probs, values = [], [], [], [], []

    while not np.all(dones):
        probs, h_policy = policy.forward(obs)
        actions_sampled = [np.random.choice(act_dim, p=probs[i]) for i in range(NUM_ENVS)]
        actions_sampled = np.array(actions_sampled)
        value, h_value = value_fn.forward(obs)

        next_obs, reward, terminated, truncated, _ = env.step(actions_sampled)
        done = np.logical_or(terminated, truncated)
        next_obs = np.array(next_obs)

        for env_idx in range(NUM_ENVS):
            if not dones[env_idx]:
                observations.append(obs[env_idx])
                actions.append(actions_sampled[env_idx])
                rewards.append(np.atleast_1d(reward[env_idx]))
                old_probs.append(np.atleast_1d(probs[env_idx, actions_sampled[env_idx]]))
                values.append(np.atleast_1d(value[env_idx]))

        obs = next_obs
        total_rewards += reward * (~dones)
        dones = np.logical_or(dones, done)

    # Flatten batch for update
    batch_obs = np.stack(observations)
    batch_action = np.array(actions)
    batch_reward = np.concatenate(rewards)
    batch_old_prob = np.concatenate(old_probs)
    batch_value = np.concatenate(values)
    next_values = np.concatenate(values[1:] + [np.zeros((NUM_ENVS,))])
    advantages = compute_advantages(batch_reward, batch_value, next_values, gamma)
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    # PPO update (single epoch for simplicity)
    for i in range(len(batch_obs)):
        obs = batch_obs[i]
        action = batch_action[i]
        adv = advantages[i]
        old_prob = batch_old_prob[i]
        value, h_value = value_fn.forward(obs)
        probs, h_policy = policy.forward(obs)
        prob = probs[action]

        ratio = prob / (old_prob + 1e-8)
        clipped_ratio = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        policy_loss = -min(ratio * adv, clipped_ratio * adv)

        dlog = -probs
        dlog[action] += 1
        grad_w2 = np.outer(h_policy, dlog) * policy_loss
        grad_w2 = np.clip(grad_w2, -1, 1)

        d_tanh = 1 - h_policy ** 2
        grad_mean = np.dot(policy.w2, dlog) * policy_loss
        grad_w1 = np.outer(obs, d_tanh * grad_mean)
        grad_w1 = np.clip(grad_w1, -1, 1)

        policy.update(grad_w1, grad_w2)

        value_target = batch_reward[i] + gamma * next_values[i]
        value_loss = (value - value_target) ** 2
        grad_w2_v = np.clip(h_value[:, None] * value_loss, -1, 1)
        grad_w1_v = np.clip(np.outer(obs, (1 - h_value ** 2) * np.dot(value_fn.w2.flatten(), value_loss)), -1, 1)
        value_fn.update(grad_w1_v, grad_w2_v)

    mean_total_reward = np.mean(total_rewards)
    episode_rewards.append(mean_total_reward)
    if mean_total_reward > best_reward:
        best_reward = mean_total_reward
        best_w1 = policy.w1.copy()
        best_w2 = policy.w2.copy()
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}: Mean Total Reward = {mean_total_reward}")

env.close()

plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('PPO: Acrobot-v1 Episode Reward Progress')
plt.show()

# Deploy best policy and render in pygame
print(f"\nDeploying best policy with reward {best_reward:.2f}...")
deploy_env = gym.make('Acrobot-v1', render_mode='human')
obs, _ = deploy_env.reset()
obs = np.array(obs)
done = False

policy.w1 = best_w1
policy.w2 = best_w2

step_count = 0
max_steps = 1000

while not done and step_count < max_steps:
    deploy_env.render()
    probs, _ = policy.forward(obs)
    action = np.argmax(probs)
    step_result = deploy_env.step(action)
    next_obs, reward, terminated, truncated, _ = step_result
    done = terminated or truncated
    next_obs = np.array(next_obs)
    obs = next_obs
    step_count += 1

deploy_env.close()