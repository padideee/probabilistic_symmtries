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

def flatten_obs(obs_dict):
    return np.concatenate([v.ravel() for v in obs_dict.values()])


# def render_side_by_side(env, base_policy, sym_policy, device, max_steps=200):
#     # Load the environment once
#     base_env = suite.load("reacher", "easy")
#
#     # Reset once and get initial state and goal
#     timestep_base = base_env.reset()
#     initial_state = base_env.physics.get_state()
#     model_xml = base_env._model_xml  # get the same underlying model
#
#     # Create a new env using the same model
#     from dm_control.suite.reacher import reacher
#     sym_env = reacher.environment(task_kwargs={"random": None})  # No new random seed
#
#     # Set sym_env to same state as base_env
#     sym_env.physics.set_state(initial_state)
#     sym_env.physics.forward()
#
#     base_frames = []
#
#     for _ in range(max_steps):
#         obs_base = torch.FloatTensor(flatten_obs(timestep_base.observation)).unsqueeze(0).to(device)
#
#         with torch.no_grad():
#             base_action, _ = base_policy(obs_base)
#             sym_action, _ = sym_policy(obs_base)
#
#         timestep_base = base_env.step(base_action.cpu().numpy()[0])
#         timestep_sym = sym_env.step(sym_action.cpu().numpy()[0])
#
#         frame_base = base_env.physics.render(camera_id=0, height=240, width=320)
#         frame_sym = sym_env.physics.render(camera_id=0, height=240, width=320)
#
#         combined = np.concatenate((frame_base, frame_sym), axis=1)
#         annotated = cv2.putText(combined.copy(), "Base Policy", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
#         annotated = cv2.putText(annotated, "Symmetrized Policy", (370, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
#
#         base_frames.append(annotated)
#
#         if timestep_base.last() or timestep_sym.last():
#             break
#
#     return base_frames


import numpy as np
import torch
import cv2
from dm_control import suite

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def set_fixed_goal_position(env, goal_xy=(0.0, 0.6)):
    """
    Hard‑code the 2‑D target position of dm_control Reacher.
    Parameters
    ----------
    env      : dm_control Environment (already reset!)
    goal_xy  : iterable of 2 floats, desired (x, y)
    """
    # 1) read full 3‑vector (x,y,z) of the 'target' geom
    target_xyz = env.physics.named.data.geom_xpos['target'].copy()

    # 2) overwrite x and y
    target_xyz[:2] = goal_xy

    # 3) write it back & forward the simulator
    env.physics.named.data.geom_xpos['target'] = target_xyz
    env.physics.forward()


def set_fixed_arm_state(env, qpos=None, qvel=None):
    if qpos is None:
        qpos = np.zeros_like(env.physics.data.qpos)
    if qvel is None:
        qvel = np.zeros_like(env.physics.data.qvel)

    env.physics.data.qpos[:] = qpos
    env.physics.data.qvel[:] = qvel
    env.physics.forward()

def in_success_band(physics):
    dist   = physics.finger_to_target_dist()                        # metres
    radii  = physics.named.model.geom_size[['target','finger'], 0].sum()
    return dist <= radii

# ---------------------------------------------------------------------
# Visual side‑by‑side roll‑out
# ---------------------------------------------------------------------
def render_side_by_side(base_policy, sym_policy, device, max_steps=200):
    base_env = suite.load("reacher", "easy")          # env for base actor
    sym_env  = suite.load("reacher", "easy")          # env for sym‑actor

    # reset and freeze both envs to identical deterministic state
    ts_base = base_env.reset()
    set_fixed_goal_position(base_env, (0.0, 0.6))
    set_fixed_arm_state(base_env)

    sym_env.physics.set_state(base_env.physics.get_state())
    sym_env.physics.forward()

    frames = []

    hits_base = 0
    hits_sym = 0

    for _ in range(max_steps):
        obs = torch.as_tensor(
            np.concatenate([v.ravel() for v in ts_base.observation.values()],
                           dtype=np.float32))[None].to(device)

        with torch.no_grad():
            a_base, _      = base_policy(obs)
            a_sym, _, _    = sym_policy.get_action_and_log_prob(obs)

        ts_base = base_env.step(a_base.cpu().numpy()[0])
        ts_sym  = sym_env.step(a_sym.cpu().numpy()[0])

        if in_success_band(base_env.physics):
            hits_base += 1
        if in_success_band(sym_env.physics):
            hits_sym += 1

        f_base = base_env.physics.render(camera_id=0, height=240, width=320)
        f_sym  = sym_env.physics.render(camera_id=0, height=240, width=320)
        canvas = np.concatenate((f_base, f_sym), axis=1)

        cv2.putText(canvas, "Base policy",       (40, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(canvas, "Symmetrized policy", (360, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(canvas, f"Base hits: {hits_base}", (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, f"Sym hits: {hits_sym}", (330, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frames.append(canvas)

        if ts_base.last() or ts_sym.last():
            break

    return frames



def render_episode(env, policy, device, max_steps=200):
    frames = []
    timestep = env.reset()
    for _ in range(max_steps):
        frame = env.physics.render(camera_id=0, height=240, width=320)
        frames.append(frame)

        obs = torch.FloatTensor(flatten_obs(timestep.observation)).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _ = policy(obs)
        timestep = env.step(action.cpu().numpy()[0])
        if timestep.last():
            break
    return frames

def plot_tsne_pca(data, labels, method='tsne', title='Embedding'):
    if data.shape[0] < 2 or np.isnan(data).any() or np.all(data == data[0]):
        print(f"[WARNING] Skipping {method} for {title} — invalid or degenerate input")
        return
    if method == 'tsne':
        reducer = TSNE(n_components=2)
    else:
        reducer = PCA(n_components=2)

    # reducer = TSNE(n_components=2) if method == 'tsne' else PCA(n_components=2)
    # breakpoint()
    reduced = reducer.fit_transform(data)
    plt.figure(figsize=(6, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', s=5)
    plt.title(f"{title}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    wandb.log({f"{method}/{title}": wandb.Image(plt)})
    plt.close()