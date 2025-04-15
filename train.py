import argparse
import os
os.environ["MUJOCO_GL"] = "egl"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dm_control import suite
import imageio
import wandb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import cv2
from utils.plotting import render_side_by_side, render_episode, plot_tsne_pca

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load environment
env = suite.load("reacher", "easy")
obs_spec, act_spec = env.observation_spec(), env.action_spec()
print("Action Spec:", act_spec)
print("Observation Spec:", obs_spec)
obs_dim = sum([np.prod(v.shape) for v in obs_spec.values()])
action_dim = act_spec.shape[0]

def flatten_obs(obs_dict):
    return np.concatenate([v.ravel() for v in obs_dict.values()])

class SACPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        action = torch.tanh(action)
        return action, log_prob

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        return self.model(torch.cat([obs, action], dim=-1))

class RotationDistribution(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, obs):
        logits = self.model(obs)
        mean_angle = torch.atan2(logits[:, 0], logits[:, 1])
        return mean_angle, 0.1

    def sample_rotation(self, obs):
        mean, std = self(obs)
        angle = mean + std * torch.randn_like(mean)
        cos, sin = torch.cos(angle), torch.sin(angle)
        return torch.stack([cos, -sin, sin, cos], dim=-1).reshape(-1, 2, 2)

class ProbSymmetrizedPolicy(nn.Module):
    def __init__(self, base_policy, rotation_dist, num_samples=4):
        super().__init__()
        self.base_policy = base_policy
        self.rotation_dist = rotation_dist
        self.num_samples = num_samples

    def forward(self, obs):
        sym_actions, log_probs = [], []
        for _ in range(self.num_samples):
            rotation_matrix = self.rotation_dist.sample_rotation(obs)
            obs_rotated = self.apply_rotation(obs, rotation_matrix)
            action_rotated, log_prob = self.base_policy(obs_rotated)
            action = self.apply_rotation_inverse(action_rotated, rotation_matrix)
            sym_actions.append(action)
            log_probs.append(log_prob)

        sym_actions = torch.stack(sym_actions)
        log_probs = torch.stack(log_probs)
        return torch.mean(sym_actions, dim=0), torch.mean(log_probs, dim=0)

    @staticmethod
    def apply_rotation(obs, rotation_matrix):
        obs_rotated = obs.clone()
        obs_rotated[:, :2] = torch.einsum('bij,bj->bi', rotation_matrix, obs[:, :2])
        obs_rotated[:, 2:4] = torch.einsum('bij,bj->bi', rotation_matrix, obs[:, 2:4])
        return obs_rotated

    @staticmethod
    def apply_rotation_inverse(action, rotation_matrix):
        return torch.einsum('bij,bj->bi', rotation_matrix.transpose(1, 2), action)


def train(args):
    # Initialize components
    base_policy = SACPolicy(obs_dim, action_dim).to(device)
    qf1 = QNetwork(obs_dim, action_dim).to(device)
    qf2 = QNetwork(obs_dim, action_dim).to(device)
    target_qf1 = QNetwork(obs_dim, action_dim).to(device)
    target_qf2 = QNetwork(obs_dim, action_dim).to(device)
    rot_dist = RotationDistribution(obs_dim).to(device)
    policy = ProbSymmetrizedPolicy(base_policy, rot_dist).to(device)

    target_qf1.load_state_dict(qf1.state_dict())
    target_qf2.load_state_dict(qf2.state_dict())

    policy_optimizer = optim.Adam(base_policy.parameters(), lr=3e-4)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=3e-4)
    rot_optimizer = optim.Adam(rot_dist.parameters(), lr=3e-4)

    replay_buffer = []
    max_buffer_size = 100000
    batch_size = 256

    alpha = 0.2
    gamma = 0.99
    tau = 0.005

    num_episodes = 1000
    reward_history = []
    report_interval = 1

    wandb.init(project="PGM_symmetries", entity="panou", name=args.name)
    # os.environ["WANDB_DIR"] = "~/scratch/cache_symmetries"
    wandb_dir = os.path.expanduser("~/scratch/cache_symmetries/wandb")
    os.makedirs(wandb_dir, exist_ok=True)  # ensure it exists
    os.environ["WANDB_DIR"] = wandb_dir

    wandb.watch(policy, log_freq=report_interval)
    wandb.watch(qf1, log_freq=report_interval)
    wandb.watch(qf2, log_freq=report_interval)
    wandb.watch(target_qf1, log_freq=report_interval)
    wandb.watch(target_qf2, log_freq=report_interval)
    wandb.watch(rot_dist, log_freq=report_interval)

    for episode in range(num_episodes):
        timestep = env.reset()
        ep_reward = 0
        wandb.log({"Episode": episode})
        while not timestep.last():
            obs = torch.FloatTensor(flatten_obs(timestep.observation)).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _ = policy(obs)
            action_np = action.cpu().numpy()[0]
            timestep_next = env.step(action_np)

            reward = timestep_next.reward
            ep_reward += reward

            next_obs = torch.FloatTensor(flatten_obs(timestep_next.observation)).unsqueeze(0).to(device)
            done = timestep_next.last()
            replay_buffer.append((obs, action, reward, next_obs, done))

            if len(replay_buffer) >= batch_size:
                batch = np.random.choice(len(replay_buffer), batch_size, replace=False)
                obs_b, action_b, reward_b, next_obs_b, done_b = zip(*[replay_buffer[i] for i in batch])

                obs_b = torch.cat(obs_b)
                action_b = torch.cat(action_b)
                reward_b = torch.FloatTensor(reward_b).unsqueeze(1).to(device)
                next_obs_b = torch.cat(next_obs_b)
                done_b = torch.FloatTensor(done_b).unsqueeze(1).to(device)

                with torch.no_grad():
                    next_action, next_log_prob = policy(next_obs_b)
                    target_q = reward_b + gamma * (1 - done_b) * (
                        torch.min(target_qf1(next_obs_b, next_action), target_qf2(next_obs_b, next_action)) - alpha * next_log_prob)

                q_loss = ((qf1(obs_b, action_b) - target_q).pow(2) + (qf2(obs_b, action_b) - target_q).pow(2)).mean()
                wandb.log({"Q Loss": q_loss.item(), "Target Q": target_q.mean().item()})

                q_optimizer.zero_grad()
                q_loss.backward()
                q_optimizer.step()

                new_action, log_prob = policy(obs_b)
                policy_loss = (alpha * log_prob - torch.min(qf1(obs_b, new_action), qf2(obs_b, new_action))).mean()

                policy_optimizer.zero_grad()
                rot_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()
                rot_optimizer.step()

                wandb.log({"Policy Loss": policy_loss.item()})

                for target_param, param in zip(target_qf1.parameters(), qf1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for target_param, param in zip(target_qf2.parameters(), qf2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            timestep = timestep_next

        reward_history.append(ep_reward)
        wandb.log({"Episode Reward": ep_reward})

        if (episode + 1) % report_interval == 0:
            average_reward = np.mean(reward_history[-report_interval:])
            print(f"Episode: {episode + 1}, Average Reward (last {report_interval}): {average_reward:.2f}")
            wandb.log({"Average Episode": episode, "Average Reward": average_reward})

            with torch.no_grad():
                mean_angle, _ = rot_dist(obs_b)
                raw_action, _ = base_policy(obs_b)
                sym_action, _ = policy(obs_b)
                rot_mat = rot_dist.sample_rotation(obs_b)

            plot_tsne_pca(sym_action.cpu().numpy(), labels=np.zeros(sym_action.shape[0]), method='tsne',
                          title='TSNE_Actions_Symmetrized_policy')
            plot_tsne_pca(raw_action.cpu().numpy(), labels=np.zeros(raw_action.shape[0]), method='tsne',
                          title='TSNE_Actions_raw_policy')


            # plot_tsne_pca(raw_action.cpu().numpy(), labels=np.zeros(raw_action.shape[0]), method='pca', title='PCA_Actions')

            wandb.log({
                "Mean Rotation Angle": mean_angle.mean().item(),
                "Raw Action Mean": raw_action.mean().item(),
                "Symmetrized Action Mean": sym_action.mean().item(),
                "Action Difference Norm": (raw_action - sym_action).norm().item(),
                "Rotation Matrix First Sample": wandb.Histogram(rot_mat[0].cpu().numpy())
            })

            frames = render_side_by_side(env, base_policy, policy, device)
            video_path = f"/home/mila/p/padideh.nouri/scratch/cache_symmetries/side_by_side_episode_{episode}.mp4"
            imageio.mimsave(video_path, frames, fps=30)
            wandb.log({"Side-by-Side Behavior": wandb.Video(video_path, fps=30)})


            # TSNE or PCA visualization of observation embeddings
            plot_tsne_pca(obs_b.cpu().numpy(), labels=np.zeros(obs_b.shape[0]), method='tsne', title='TSNE_Observations')
            plot_tsne_pca(obs_b.cpu().numpy(), labels=np.zeros(obs_b.shape[0]), method='pca', title='PCA_Observations')

            plot_tsne_pca(action_b.cpu().numpy(), labels=np.zeros(action_b.shape[0]), method='tsne', title='TSNE_Actions')
            plot_tsne_pca(action_b.cpu().numpy(), labels=np.zeros(action_b.shape[0]), method='pca', title='PCA_Actions')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Probabilistic Symmetrized Policy.")
    parser.add_argument("--name", type=str, default="prob_sym_policy", help="Name of the experiment.")
    args = parser.parse_args()
    print("Starting training...")

    train(args)
    wandb.finish()
    print("Training completed.")