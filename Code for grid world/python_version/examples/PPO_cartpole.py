
# PPO for Mujoco InvertedPendulum-v5
# -----------------------------------
# This script implements Proximal Policy Optimization (PPO) for the Mujoco InvertedPendulum-v5 environment using PyTorch and vectorized environments for efficient training.
#
# PPO is a policy gradient method that alternates between sampling trajectories and optimizing a surrogate objective function. It uses a clipped probability ratio to ensure stable updates.
#
# PPO Objective (Clipped):
#   L^{CLIP} = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
#   where r_t(θ) = π_θ(a_t|s_t) / π_{θ_{old}}(a_t|s_t)
#   and A_t is the advantage estimate.
#
# Value Function Loss:
#   L^{VF} = (V_θ(s_t) - R_t)^2
#
# Generalized Advantage Estimation (GAE):
#   δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
#   A_t = δ_t + γ * λ * (1 - done) * A_{t+1}
#
# The algorithm alternates between collecting trajectories, computing advantages, and updating the policy and value networks.

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Speed up: use vectorized envs
def make_env():
    # Create a Mujoco InvertedPendulum-v5 environment without rendering for fast training
    return gym.make("InvertedPendulum-v5", render_mode=None, disable_env_checker=True)

envs = gym.vector.AsyncVectorEnv([make_env for _ in range(16)])  # 8 parallel envs

obs_dim = envs.single_observation_space.shape[0]  # State dimension
act_dim = envs.single_action_space.shape[0]       # Action dimension

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # Actor network outputs mean action (policy, π | gaussian distribution)
        # Q: why Tanh; 
        # A: Range(-1, 1) for training stability and gradient flow better, symmetric around 0: faster convergence, smoothness.
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        # Critic network outputs state value
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Learnable log standard deviation for Gaussian policy
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        # Returns mean action and value estimate
        return self.actor(x), self.critic(x)

def get_action_and_value(model, obs):
    # Given observation, sample action from policy and get log probability and value
    obs_tensor = torch.from_numpy(obs).float().to(device)
    mu, value = model(obs_tensor)
    std = model.log_std.exp()
    dist = torch.distributions.Normal(mu, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(-1)
    return action.cpu().detach().numpy(), log_prob.cpu().detach().numpy(), value.cpu().detach().numpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCritic(obs_dim, act_dim).to(device)  # Policy and value network
optimizer = optim.Adam(model.parameters(), lr=3e-4)  # Adam optimizer

# PPO hyperparameters
clip_eps = 0.2      # PPO clipping parameter (epsilon)
epochs = 10         # Number of PPO epochs per update
steps_per_epoch = 2048  # Number of steps to collect per update
gamma = 0.99        # Discount factor
lam = 0.95          # GAE lambda

obs = envs.reset()[0]
reward_history = []
for update in range(100):
    # --- Trajectory Collection ---
    # Collect trajectories from vectorized environments
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    for step in range(steps_per_epoch):
        action, logp, value = get_action_and_value(model, obs)
        next_obs, reward, done, trunc, info = envs.step(action)
        obs_buf.append(obs)
        act_buf.append(action)
        logp_buf.append(logp)
        rew_buf.append(reward)
        val_buf.append(value)
        done_buf.append(done)
        obs = next_obs

    # --- Advantage and Return Calculation (GAE) ---
    # Convert buffers to numpy arrays
    obs_buf = np.array(obs_buf)  # (steps_per_epoch, num_envs, obs_dim)
    act_buf = np.array(act_buf)  # (steps_per_epoch, num_envs, act_dim)
    logp_buf = np.array(logp_buf)  # (steps_per_epoch, num_envs)
    rew_buf = np.array(rew_buf)  # (steps_per_epoch, num_envs)
    val_buf = np.array(val_buf)  # (steps_per_epoch, num_envs)
    if val_buf.ndim == 3 and val_buf.shape[2] == 1:
        val_buf = val_buf.squeeze(-1)
    done_buf = np.array(done_buf)  # (steps_per_epoch, num_envs)

    # Compute GAE advantages and returns
    adv_buf = np.zeros_like(rew_buf)  # (steps_per_epoch, num_envs)
    for env_idx in range(envs.num_envs):
        lastgaelam_env = 0
        for t in reversed(range(steps_per_epoch)):
            # GAE math:
            # δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
            # A_t = δ_t + γ * λ * (1 - done) * A_{t+1}
            if t == steps_per_epoch - 1:
                nextnonterminal = 1.0 - done_buf[t, env_idx]
                nextvalue = val_buf[t, env_idx]
            else:
                nextnonterminal = 1.0 - done_buf[t+1, env_idx]
                nextvalue = val_buf[t+1, env_idx]
            delta = rew_buf[t, env_idx] + gamma * nextvalue * nextnonterminal - val_buf[t, env_idx]
            lastgaelam_env = delta + gamma * lam * nextnonterminal * lastgaelam_env
            adv_buf[t, env_idx] = lastgaelam_env
    returns = adv_buf + val_buf  # Estimated returns

    # --- Prepare Data for PPO Update ---
    # Flatten buffers for batch training
    obs_flat = torch.from_numpy(obs_buf.reshape(-1, obs_dim)).float().to(device)
    act_flat = torch.from_numpy(act_buf.reshape(-1, act_dim)).float().to(device)
    logp_flat = torch.from_numpy(logp_buf.flatten()).float().to(device)
    adv_flat = torch.from_numpy(adv_buf.flatten()).float().to(device)
    ret_flat = torch.from_numpy(returns.flatten()).float().to(device)

    # --- PPO Policy and Value Update ---
    # Optimize policy and value function using PPO clipped objective
    for _ in range(epochs):
        mu, value = model(obs_flat)
        std = model.log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        new_logp = dist.log_prob(act_flat).sum(-1)
        ratio = (new_logp - logp_flat).exp()  # Probability ratio
        # PPO surrogate objective
        surr1 = ratio * adv_flat
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_flat
        policy_loss = -torch.min(surr1, surr2).mean()
        # Value function loss
        value_loss = ((ret_flat - value.squeeze()) ** 2).mean()
        loss = policy_loss + 0.5 * value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- Track and Print Training Progress ---
    mean_reward = np.mean(rew_buf)
    print(f"Update {update}: mean reward {mean_reward}")
    reward_history.append(mean_reward)

envs.close()


# --- Visualize Training Process ---
plt.figure(figsize=(10,5))
plt.plot(reward_history, label='Mean Reward per Update')
plt.xlabel('Update')
plt.ylabel('Mean Reward')
plt.title('PPO Training Progress')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Optionally, visualize a moving average for smoother curve
window = 5
if len(reward_history) >= window:
    moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10,5))
    plt.plot(reward_history, alpha=0.3, label='Raw Mean Reward')
    plt.plot(range(window-1, len(reward_history)), moving_avg, color='red', label=f'Moving Avg (window={window})')
    plt.xlabel('Update')
    plt.ylabel('Mean Reward')
    plt.title('PPO Training Progress (Smoothed)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Visualize Trained Policy in Mujoco ---
# After training, run the policy in a rendered environment to see performance
render_env = gym.make("InvertedPendulum-v5", render_mode="human", disable_env_checker=True)
obs = render_env.reset()[0]
done = False
while not done:
    action, _, _ = get_action_and_value(model, obs)
    obs, reward, done, trunc, info = render_env.step(action)
    if done or trunc:
        break
render_env.close()