import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dm_control import suite
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten_obs(obs_dict):
    return np.concatenate([v.ravel() for v in obs_dict.values()])

class ReplayBuffer:
    """Same as your original ReplayBuffer."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.storage = []
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.ptr] = transition
            self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self.storage), size=batch_size)
        obs, actions, rewards, next_obs, dones = [], [], [], [], []
        for i in idxs:
            o, a, r, no, d = self.storage[i]
            obs.append(o)
            actions.append(a)
            rewards.append(r)
            next_obs.append(no)
            dones.append(d)
        return (
            torch.cat(obs).to(device),
            torch.cat(actions).to(device),
            torch.FloatTensor(rewards).unsqueeze(1).to(device),
            torch.cat(next_obs).to(device),
            torch.FloatTensor(dones).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.storage)

# ------------------------------------------------------------------------------
# Base Actor (same as your original Tanh-squashed Gaussian policy)
# ------------------------------------------------------------------------------
class BaseActor(nn.Module):
    """A standard SAC actor (tanh-squashed Gaussian)."""
    def __init__(self, obs_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_action_and_log_prob(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # reparameterization
        action = torch.tanh(z)
        # Tanh correction in log-prob:
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob

# ------------------------------------------------------------------------------
# Critic Q (same as your original)
# ------------------------------------------------------------------------------
class CriticQ(nn.Module):
    """Q-network: outputs Q(s,a) for a given state and action."""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        return self.model(torch.cat([obs, action], dim=-1))

def soft_update(source_net, target_net, tau):
    for param, target_param in zip(source_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# ------------------------------------------------------------------------------
# RotationDistribution: produces a random rotation in 2D for the Reacher domain
# ------------------------------------------------------------------------------
class RotationDistribution(nn.Module):
    """
    Returns a random angle for 2D rotation, using a small MLP to produce mean/std
    if desired. For simplicity, we’ll just return a uniform or a simple
    normal-based angle. If you'd like a learned distribution, make a small network.
    """
    def __init__(self):
        super().__init__()
        # Here we’ll keep it extremely simple and produce angles with a constant std
        # or you can make a small MLP to produce mean, log_std from obs if needed.
        # For demonstration, let's do a trivial MLP -> (mean, log_std).
        self.net = nn.Sequential(
            nn.Linear(1, 64),  # example dimension
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, obs):
        """
        obs: (batch_size, obs_dim). We could choose a smaller embedding if desired.
        Just a minimal example here: we compress obs to 1 dimension, pass to net.
        """
        # Combine entire batch into single embedding for demonstration, or
        # pick some simpler feature, etc.
        # In practice, do something more relevant to the Reacher’s geometry.
        with torch.no_grad():
            # e.g. we just sum up obs for a single scalar feature:
            scalar_feat = obs.sum(dim=1, keepdim=True)
        x = self.net(scalar_feat)
        mean_angle = torch.atan2(x[:, 0], x[:, 1])  # angle in [-pi, pi]
        # You could also produce a learned log_std and sample. For simplicity,
        # let's keep it a small constant:
        std = 0.1  # or from a trainable param
        return mean_angle, std

    def sample_rotation(self, obs):
        mean, std = self(obs)
        angle = mean + std * torch.randn_like(mean)  # shape [batch_size]
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        # Build 2x2 rotation matrix for each batch
        # shape => (batch_size, 2, 2)
        rot_mats = []
        for c, s in zip(cos, sin):
            mat = torch.tensor([[c, -s],
                                [s,  c]], device=obs.device)
            rot_mats.append(mat)
        return torch.stack(rot_mats, dim=0)

# ------------------------------------------------------------------------------
# ProbSymmetrizedPolicy: the "equivariant" actor using multiple rotation samples
# ------------------------------------------------------------------------------
class ProbSymmetrizedPolicy(nn.Module):
    def __init__(self, base_actor, rotation_dist, num_samples=4):
        super().__init__()
        self.base_actor = base_actor
        self.rotation_dist = rotation_dist
        self.num_samples = num_samples

    def forward(self, obs):
        """We won't use .forward directly, we’ll define a get_action_and_log_prob
           in the same style as your original actor."""
        raise NotImplementedError

    def get_action_and_log_prob(self, obs):
        """
        For each observation, sample a few random rotation matrices,
        rotate obs -> feed to base actor -> rotate action back.
        Combine the results (e.g. average).
        """
        # We'll store the final actions & log_probs from each sample
        all_actions = []
        all_logprobs = []

        # Sample rotation matrices
        rot_mats = self.rotation_dist.sample_rotation(obs)  # shape [B, 2, 2]

        for _ in range(self.num_samples):
            # Sample a new rotation matrix for each sample
            R = self.rotation_dist.sample_rotation(obs)  # shape [B, 2, 2]
            rotated_obs = self.apply_rotation(obs, R)    # shape same as obs

            # Feed rotated obs to the base actor
            # action for rotated obs
            action_rot, logp_rot = self.base_actor.get_action_and_log_prob(rotated_obs)
            # Now rotate the (first 2 dims) of action back:
            # (Assuming Reacher actions are 2D or more, you must define how to rotate them.)
            # If action_dim=2: we just rotate both. If you have more dims, rotate the first two,
            # keep others the same. Example below assumes act_dim=2:
            true_action = self.apply_rotation_inverse(action_rot, R)
            all_actions.append(true_action)
            all_logprobs.append(logp_rot)

        # Average
        final_actions = torch.mean(torch.stack(all_actions, dim=0), dim=0)
        final_logprobs = torch.mean(torch.stack(all_logprobs, dim=0), dim=0)
        return final_actions, final_logprobs

    @staticmethod
    def apply_rotation(obs, rotation_matrix):
        """
        Example if we want to treat the first 2 dims of obs as x,y and rotate them.
        If your Reacher observation is larger, you’d need to decide exactly what to rotate.
        For demonstration, let's assume obs.shape = [B, obs_dim], and we rotate just the first 2.
        """
        obs_rotated = obs.clone()
        # Extract x,y
        xy = obs_rotated[:, :2]  # shape [B,2]
        # matrix multiply for each batch
        # R shape [B,2,2], xy shape [B,2], result shape [B,2]
        xy_rot = torch.einsum('bij, bj->bi', rotation_matrix, xy)
        obs_rotated[:, :2] = xy_rot
        return obs_rotated

    @staticmethod
    def apply_rotation_inverse(action, rotation_matrix):
        """
        Rotate the first 2 dims of the action back by R^T.
        """
        act_out = action.clone()
        xy = act_out[:, :2]  # shape [B,2]
        # rotation_matrix^T is the inverse
        rot_T = rotation_matrix.transpose(1,2)
        xy_inv = torch.einsum('bij, bj->bi', rot_T, xy)
        act_out[:, :2] = xy_inv
        return act_out

# ------------------------------------------------------------------------------
# Main SAC training using the new ProbSymmetrizedPolicy
# ------------------------------------------------------------------------------
def train_sac(args):
    # 1) Load environment
    env = suite.load(domain_name="reacher", task_name="easy")
    action_spec = env.action_spec()
    obs_spec = env.observation_spec()

    obs_dim = sum(np.prod(v.shape) for v in obs_spec.values())
    act_dim = action_spec.shape[0]

    # 2) Initialize networks & target networks
    # Critic
    qf1 = CriticQ(obs_dim, act_dim).to(device)
    qf2 = CriticQ(obs_dim, act_dim).to(device)
    target_qf1 = CriticQ(obs_dim, act_dim).to(device)
    target_qf2 = CriticQ(obs_dim, act_dim).to(device)
    target_qf1.load_state_dict(qf1.state_dict())
    target_qf2.load_state_dict(qf2.state_dict())

    # Base Actor
    base_actor = BaseActor(obs_dim, act_dim).to(device)
    # Rotation distribution
    rot_dist = RotationDistribution().to(device)
    # Probabilistic Symmetric Actor
    actor = ProbSymmetrizedPolicy(base_actor, rot_dist, num_samples=4).to(device)

    actor_optimizer = optim.Adam(list(base_actor.parameters()) + list(rot_dist.parameters()), lr=args.lr)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.lr)

    # 3) Hyperparameters
    alpha = args.alpha
    gamma = args.gamma
    tau = args.tau
    replay_buffer = ReplayBuffer(args.buffer_size)

    if args.wandb:
        wandb.init(project="dmcontrol_sac",
                   entity="panou",
                   name=args.run_name)
        wandb.config.update(args)

    returns = []
    global_step = 0

    for episode in range(args.num_episodes):
        time_step = env.reset()
        ep_return = 0.0

        while not time_step.last():
            global_step += 1
            obs_array = flatten_obs(time_step.observation)
            obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0).to(device)

            # ACTION SELECTION
            if global_step < args.learning_starts:
                # Random action before learning starts
                action_np = np.random.uniform(action_spec.minimum,
                                              action_spec.maximum,
                                              size=action_spec.shape)
            else:
                with torch.no_grad():
                    action_t, _ = actor.get_action_and_log_prob(obs_tensor)
                action_np = action_t.cpu().numpy()[0]

            time_step_next = env.step(action_np)
            reward = time_step_next.reward if time_step_next.reward else 0.0
            ep_return += reward

            done = time_step_next.last()
            next_obs_array = flatten_obs(time_step_next.observation)
            next_obs_tensor = torch.FloatTensor(next_obs_array).unsqueeze(0).to(device)

            replay_buffer.add((
                obs_tensor,
                torch.FloatTensor(action_np).unsqueeze(0).to(device),
                reward,
                next_obs_tensor,
                float(done)
            ))

            # SAC updates
            if (len(replay_buffer) >= args.batch_size and
                global_step % args.update_every == 0 and
                global_step >= args.learning_starts):

                (obs_b, act_b, rew_b, next_obs_b, done_b) = replay_buffer.sample(args.batch_size)

                # 1) Compute target Q
                with torch.no_grad():
                    next_action, next_log_prob = actor.get_action_and_log_prob(next_obs_b)
                    target_q1_val = target_qf1(next_obs_b, next_action)
                    target_q2_val = target_qf2(next_obs_b, next_action)
                    target_q = torch.min(target_q1_val, target_q2_val) - alpha * next_log_prob
                    target_q = rew_b + gamma * (1 - done_b) * target_q

                # 2) Update Q networks
                q1_val = qf1(obs_b, act_b)
                q2_val = qf2(obs_b, act_b)
                qf1_loss = (q1_val - target_q).pow(2).mean()
                qf2_loss = (q2_val - target_q).pow(2).mean()
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # 3) Update actor
                new_action, log_prob = actor.get_action_and_log_prob(obs_b)
                q_new_action = torch.min(qf1(obs_b, new_action), qf2(obs_b, new_action))
                actor_loss = (alpha * log_prob - q_new_action).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # 4) Soft update
                soft_update(qf1, target_qf1, tau)
                soft_update(qf2, target_qf2, tau)

                # Optional logging
                if args.wandb and global_step % 1000 == 0:
                    wandb.log({
                        "actor_loss": actor_loss.item(),
                        "qf_loss": qf_loss.item(),
                        "qf1_loss": qf1_loss.item(),
                        "qf2_loss": qf2_loss.item(),
                        "global_step": global_step
                    })

            time_step = time_step_next

        returns.append(ep_return)
        if args.wandb:
            wandb.log({"episode_reward": ep_return, "episode": episode})

        if (episode+1) % args.log_interval == 0:
            avg_return = np.mean(returns[-args.log_interval:])
            print(f"Episode {episode+1}, Average Return: {avg_return:.2f}")

    if args.wandb:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="dmcontrol_symmetric_sac_run")
    parser.add_argument("--num_episodes", type=int, default=300)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--learning_starts", type=int, default=5000)
    parser.add_argument("--update_every", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging if set")
    args = parser.parse_args()

    print("Training SAC (Probabilistic Symmetric) on dm_control Reacher-easy task.")
    train_sac(args)
    print("Done.")

if __name__ == "__main__":
    main()
