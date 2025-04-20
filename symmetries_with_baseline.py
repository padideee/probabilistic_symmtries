# Probabilistic Symmetric SAC with rich visualisations for dm_control Reacher‑easy
# -----------------------------------------------------------------------------
# • Adds extensive WandB logging & matplotlib plots:
#     – episode return & success‑rate
#     – angle‑entropy and rotation‑angle histogram
#     – fingertip XY heatmap & spider plot
#     – success‑timeline raster
#     – Q‑function slice surface (optional)
# • Structured into a `VisLogger` helper to keep the training loop clean.
# -----------------------------------------------------------------------------

import os, random, math, argparse, tempfile, json
os.environ["MUJOCO_GL"] = "egl"
from dataclasses import dataclass, field
import pathlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dm_control import suite
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
import matplotlib.pyplot as plt
import imageio
import wandb

from utils.plotting import render_side_by_side  # unchanged helper supplied by user

# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_obs(obs_dict):
    return np.concatenate([v.ravel() for v in obs_dict.values()]).astype(np.float32)

def check(name, t):
    if torch.isnan(t).any():
        raise RuntimeError(f"NaNs after {name}")

# -----------------------------------------------------------------------------
# Replay Buffer (unchanged logic)
# -----------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.storage: List[Tuple[torch.Tensor, ...]] = []
        self.ptr = 0

    def add(self, transition):
        obs, act, rew, nxt, done = transition
        if not torch.is_tensor(rew):
            rew = torch.tensor([[rew]], dtype=torch.float32, device=obs.device)
        if not torch.is_tensor(done):
            done = torch.tensor([[done]], dtype=torch.float32, device=obs.device)
        packed = (obs, act, rew, nxt, done)
        if len(self.storage) < self.capacity:
            self.storage.append(packed)
        else:
            self.storage[self.ptr] = packed
            self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size):
        idx = np.random.randint(len(self.storage), size=batch_size)
        cols = list(zip(*[self.storage[i] for i in idx]))
        obs, act, rew, nxt, done = map(torch.cat, cols)
        return obs, act, rew, nxt, done

    def __len__(self):
        return len(self.storage)


# -----------------------------------------------------------------------------
# Networks (BaseActor, CriticQ) – unchanged except BaseActor.tanh‑squash correction
# -----------------------------------------------------------------------------

class BaseActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_action_and_log_prob(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        return action, log_prob.sum(dim=1, keepdim=True)


class CriticQ(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        return self.model(torch.cat([obs, action], dim=-1))


def soft_update(source, target, tau):
    for p, tp in zip(source.parameters(), target.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
        # tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


# -----------------------------------------------------------------------------
# Rotation Distributions
# -----------------------------------------------------------------------------

class RotationDistribution(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Linear(hidden_dim, 1)

    def sample_angles(self, obs, S):
        hidden = self.net(obs)
        mean = self.mean(hidden)
        raw_log_std = self.log_std(hidden)
        log_std = torch.clamp(raw_log_std, -5.0, 2.0) # TODO
        entropy_val = 0.5 * (math.log(2.0 * math.pi * math.e) + 2.0 * log_std)
        eps = torch.randn(S, obs.size(0), device=obs.device)
        angles = mean.T + torch.exp(log_std).T * eps
        return angles, entropy_val.mean()


class UniformRotationDistribution(nn.Module):
    def sample_angles(self, obs, S):
        B = obs.size(0)
        angles = (torch.rand(S, B, device=obs.device) * 2 * math.pi) - math.pi
        return angles, math.log(2.0 * math.pi)


class CanonicalRotationDistribution(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
                         nn.Linear(obs_dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, 1)

    def sample_angles(self, obs, S):
        # ignore S — we only ever use one sample
        h = self.net(obs)
        m = self.mean(h)            # (B,1)
        angles = m.T                     # (1,B)  — same as mean for each sample
        entropy = 0.0                    # we’re deterministic
        return angles, entropy

class IdentityRotationDistribution(nn.Module):
    """Always return angle=0 and entropy=0, for S=1."""
    def sample_angles(self, obs, S):
        B = obs.size(0)
        # one zero‐angle per element
        angles = torch.zeros(S, B, device=obs.device)
        entropy = 0.0
        return angles, entropy

# -----------------------------------------------------------------------------
# Probabilistic Symmetrised Policy (returns angles too)
# -----------------------------------------------------------------------------

class ProbSymmetrizedPolicy(nn.Module):
    def __init__(self, base_actor, rotation_dist, num_samples=4):
        super().__init__()
        self.base_actor, self.rotation_dist = base_actor, rotation_dist
        self.num_samples = num_samples
        self.rotate_all_dims = True
        self.rotate_12 = self.rotate_34 = self.rotate_56 = False

    def get_action_and_log_prob(self, obs, return_angles=False):
        S, B = self.num_samples, obs.size(0)
        angles, angle_entropy = self.rotation_dist.sample_angles(obs, S)
        check("rot.sample_angles", angles)
        cos, sin = torch.cos(angles), torch.sin(angles)
        check("cos/sin", cos)
        R = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)

        segs, seg_start = [], 0
        flags = [self.rotate_12, self.rotate_34, self.rotate_56]
        for flag in flags:
            seg = obs[:, seg_start:seg_start + 2]
            if self.rotate_all_dims or flag:
                seg = self._apply_R(R, seg)
            else:
                seg = seg.unsqueeze(0).expand(S, -1, -1)
            segs.append(seg)
            seg_start += 2
        obs_rep = torch.cat(segs, dim=-1)
        check("obs_rep", obs_rep)

        mean, log_std = self.base_actor(obs_rep.reshape(S * B, -1))
        check("base_actor forward", mean)
        base = Normal(mean, log_std.exp())
        dist = TransformedDistribution(base, [TanhTransform(cache_size=1)])

        a = dist.rsample()
        logp = dist.log_prob(a).sum(-1, keepdim=True)
        a = a.view(S, B, -1)
        logp = logp.view(S, B, 1)

        inv_R = R.transpose(-1, -2)
        def undo(idx, do):
            blk = a[..., idx:idx + 2]
            if do:
                blk = torch.einsum('sbji,sbj->sbi', inv_R, blk)
            return blk

        r12, r34, r56 = (self.rotate_all_dims or self.rotate_12,
                         self.rotate_all_dims or self.rotate_34,
                         self.rotate_all_dims or self.rotate_56)
        parts = [
            undo(0, r12),
            undo(2, r34) if a.size(-1) >= 4 else None,
            undo(4, r56) if a.size(-1) >= 6 else None,
        ]
        parts = [p for p in parts if p is not None]
        a_back = torch.cat(parts, dim=-1)

        act_mean = a_back.mean(0)
        logp_mean = logp.mean(0)
        if return_angles:
            return act_mean, logp_mean, angle_entropy, angles
        return act_mean, logp_mean, angle_entropy

    def _apply_R(self, R, vec):
        return torch.einsum('sbji,bj->sbi', R, vec)

# -----------------------------------------------------------------------------
# Visualisation helper
# -----------------------------------------------------------------------------

@dataclass
class VisLogger:
    args: argparse.Namespace
    act_dim: int
    episode_xy: List[np.ndarray] = field(default_factory=list)  # (T,2) list
    episode_success: List[int] = field(default_factory=list)     # 0/1 per step
    rotation_angles: List[np.ndarray] = field(default_factory=list)  # for histogram
    angle_entropy_vals: List[float] = field(default_factory=list)

    # ---------------------------------------------------------------------
    # Per‑step hooks
    # ---------------------------------------------------------------------
    def step(self, xy: np.ndarray, hit: bool, angles: np.ndarray = None, ent: float = None):
        self.episode_xy.append(xy)
        self.episode_success.append(int(hit))
        if angles is not None:
            self.rotation_angles.append(angles)
        if ent is not None:
            if isinstance(ent, torch.Tensor) and ent.device.type != "cpu":
                ent_val = ent.item()  # GPU → move to host scalar
            else:
                ent_val = float(ent) if isinstance(ent, torch.Tensor) else ent
            self.angle_entropy_vals.append(ent_val)

    # ---------------------------------------------------------------------
    # End‑episode summaries
    # ---------------------------------------------------------------------
    def end_episode(self, episode_idx: int, ep_return: float, base_actor, sym_actor, device):
        xy_arr = np.stack(self.episode_xy)
        succ_arr = np.array(self.episode_success)
        success_rate = succ_arr.mean()

        wandb.log({
            "episode/return": ep_return,
            "episode/success_rate": success_rate,
            "episode/episode": episode_idx
        })

        if episode_idx % self.args.log_interval == 0:
            self._plot_xy_traj(xy_arr, episode_idx)
            self._plot_spider(xy_arr, episode_idx)
            self._plot_success_timeline(succ_arr, episode_idx)
            self._log_angle_stats(episode_idx)
            # side‑by‑side video
            frames = render_side_by_side(base_actor, sym_actor, device, max_steps=200)
            vid_path = pathlib.Path(tempfile.gettempdir()) / f'sbs_{episode_idx}.mp4'
            imageio.mimsave(vid_path, frames, fps=30)
            wandb.log({f'video/sbs': wandb.Video(str(vid_path), fps=30)})
        # clear buffers
        self.episode_xy.clear(); self.episode_success.clear()

    # ------------------------------------------------------------------
    # Private plot helpers
    # ------------------------------------------------------------------
    def _plot_xy_traj(self, xy_arr, ep):
        fig, ax = plt.subplots()
        ax.plot(xy_arr[:, 0], xy_arr[:, 1], marker="o")
        ax.set_title(f"XY Trajectory · ep {ep}")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.axis("equal"); ax.grid(True)
        wandb.log({"vis/xy_traj": wandb.Image(fig), "episode": ep})
        plt.close(fig)

    def _plot_spider(self, xy_arr, ep):
        # polar spider plot (angle vs radius)
        radii = np.linalg.norm(xy_arr, axis=1)
        theta = np.arctan2(xy_arr[:, 1], xy_arr[:, 0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.plot(theta, radii, marker=".")
        ax.set_title(f"Spider plot · ep {ep}")
        wandb.log({"vis/spider": wandb.Image(fig), "episode": ep})
        plt.close(fig)

    def _plot_success_timeline(self, succ_arr, ep):
        """One‑row raster: dark = timestep where fingertip is inside the success band."""
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.imshow(succ_arr[None, :], aspect='auto', cmap='Greens', vmin=0, vmax=1)
        ax.set_title('Success timeline')
        ax.set_yticks([])
        ax.set_xlabel('Timestep')
        ax.set_xlim(0, len(succ_arr) - 1)
        plt.tight_layout()
        wandb.log({"vis/success_timeline": wandb.Image(fig), "episode": ep})
        plt.close(fig)

    def _log_angle_stats(self, ep):
        if not self.rotation_angles:
            return
        vals = [float(v) for v in self.angle_entropy_vals]  # all guaranteed python floats
        mean_ent = np.mean(vals)

        all_angles = np.concatenate([a.flatten() for a in self.rotation_angles])
        fig, ax = plt.subplots()
        ax.hist(all_angles, bins=30, density=True)
        ax.set_title("Rotation‑angle histogram")
        wandb.log({
            "angles/entropy_mean": mean_ent,
            "angles/hist": wandb.Image(fig),
            "episode": ep
        })
        plt.close(fig)
        self.rotation_angles.clear(); self.angle_entropy_vals.clear()


# -----------------------------------------------------------------------------
# Main training – now uses VisLogger
# -----------------------------------------------------------------------------

def train_sac(args):
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = suite.load("reacher", "easy", task_kwargs={"random": np.random.RandomState(args.seed)})
    obs_dim = sum(np.prod(v.shape) for v in env.observation_spec().values())
    act_dim = env.action_spec().shape[0]

    # networks
    qf1, qf2 = CriticQ(obs_dim, act_dim).to(device), CriticQ(obs_dim, act_dim).to(device)
    tgt_qf1, tgt_qf2 = CriticQ(obs_dim, act_dim).to(device), CriticQ(obs_dim, act_dim).to(device)
    tgt_qf1.load_state_dict(qf1.state_dict()); tgt_qf2.load_state_dict(qf2.state_dict())

    base_actor = BaseActor(obs_dim, act_dim).to(device)
    if args.no_sym:
        rot_dist = IdentityRotationDistribution()
        actor = ProbSymmetrizedPolicy(base_actor, rot_dist, num_samples=1).to(device)
    elif args.canonicalize:
        rot_dist  = CanonicalRotationDistribution(obs_dim).to(device)
        actor = ProbSymmetrizedPolicy(base_actor, rot_dist , num_samples=1).to(device)
    else:
        rot_dist = (UniformRotationDistribution() if args.rotation_dist == 'uniform'
                    else RotationDistribution(obs_dim).to(device))
        actor = ProbSymmetrizedPolicy(base_actor, rot_dist, args.num_samples).to(device)

    q_opt = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.lr)
    actor_opt = optim.Adam(list(base_actor.parameters()) + list(rot_dist.parameters()), lr=args.lr)

    replay = ReplayBuffer(args.buffer_size)
    vis = VisLogger(args, act_dim)

    wandb.init(project="dmcontrol_sac", name=args.run_name, config=args.__dict__) if args.wandb else None
    if args.wandb:
        wandb.watch(models=[base_actor, rot_dist, qf1, qf2], log="gradients",  log_freq=1000)

    global_step = 0

    for ep in range(args.num_episodes):
        ts = env.reset(); ep_ret = 0.0; step_idx = 0
        while not ts.last():
            global_step += 1; step_idx += 1
            print("Episode: ", ep, "Episode step: ", step_idx)
            obs = torch.tensor(flatten_obs(ts.observation), device=device).unsqueeze(0)
            with torch.no_grad():
                if global_step < args.learning_starts:
                    act = np.random.uniform(env.action_spec().minimum, env.action_spec().maximum, size=env.action_spec().shape)
                    angles = None; ang_ent = None
                else:
                    a_t, _, ang_ent, angles = actor.get_action_and_log_prob(obs, return_angles=True)
                    act = a_t.cpu().numpy()[0]
            ts_next = env.step(act)
            reward = ts_next.reward or 0.0; ep_ret += reward
            done = ts_next.last()
            next_obs = torch.tensor(flatten_obs(ts_next.observation), device=device).unsqueeze(0)
            replay.add((obs, torch.tensor(act, device=device).unsqueeze(0).float(), reward, next_obs, float(done)))

            # heuristic success: reward close to 1
            hit = reward > 0.9
            vis.step(obs.cpu().numpy()[0, :2], hit, angles.cpu().numpy() if angles is not None else None,
                     ang_ent if ang_ent is not None else None)

            # learning updates
            if len(replay) >= args.batch_size and global_step % args.update_every == 0 and global_step >= args.learning_starts:
                b_obs, b_act, b_rew, b_nxt, b_done = replay.sample(args.batch_size)
                with torch.no_grad():
                    nxt_a, nxt_lp, _ = actor.get_action_and_log_prob(b_nxt)
                    q_target = torch.min(tgt_qf1(b_nxt, nxt_a), tgt_qf2(b_nxt, nxt_a)) - args.alpha * nxt_lp
                    q_target = b_rew + args.gamma * (1 - b_done) * q_target
                q1, q2 = qf1(b_obs, b_act), qf2(b_obs, b_act)
                q_loss = ((q1 - q_target).pow(2).mean() + (q2 - q_target).pow(2).mean())
                q_opt.zero_grad(); q_loss.backward(); q_opt.step()

                new_a, lp, ang_ent = actor.get_action_and_log_prob(b_obs)
                q_new = torch.min(qf1(b_obs, new_a), qf2(b_obs, new_a))
                actor_loss = (args.alpha * lp - q_new).mean() - args.angle_ent_coef * ang_ent
                actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

                soft_update(qf1, tgt_qf1, args.tau); soft_update(qf2, tgt_qf2, args.tau)
                if args.wandb and global_step % args.log_interval == 0:
                    wandb.log({
                        "loss/q": q_loss.item(), "loss/actor": actor_loss.item(),
                        "angles/entropy_batch": ang_ent, "step": global_step})

            ts = ts_next
        # end episode
        if args.wandb: vis.end_episode(ep, ep_ret, base_actor, actor, device)
        else: vis.episode_xy.clear(), vis.episode_success.clear()

    if args.wandb: wandb.finish()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, default="sym_sac_vis")
    p.add_argument("--num_episodes", type=int, default=1_000)
    p.add_argument("--buffer_size", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--learning_starts", type=int, default=5_000)
    p.add_argument("--update_every", type=int, default=1)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rotation_dist", choices=["uniform", "learned"], default="learned")
    p.add_argument("--angle_ent_coef", type=float, default=0.01)
    p.add_argument('--no_sym', action='store_true', help='Use plain SAC without any rotation symmetrization')
    p.add_argument('--canonicalize', action='store_true', help='Use a deterministic canonicalizer instead of sampling')

    args = p.parse_args()

    os.environ["MUJOCO_GL"] = "egl"
    print("\n―― Training Probabilistic Symmetric SAC with visualisations ――\n")
    train_sac(args)


if __name__ == "__main__":
    cli()
