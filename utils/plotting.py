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
#     base_env = suite.load("reacher", "easy")
#     # sym_env = suite.load("reacher", "easy")
#     sym_env = base_env.copy()
#
#     base_frames = []
#     sym_frames = []
#
#     timestep_base = base_env.reset()
#     timestep_sym = sym_env.reset()
#
#     for _ in range(max_steps):
#         obs_base = torch.FloatTensor(flatten_obs(timestep_base.observation)).unsqueeze(0).to(device)
#         obs_sym = torch.FloatTensor(flatten_obs(timestep_sym.observation)).unsqueeze(0).to(device)
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

# def render_side_by_side(env, base_policy, sym_policy, device, max_steps=200):
#     # Load environment once
#     base_env = suite.load("reacher", "easy")
#     sym_env = suite.load("reacher", "easy")
#
#     # Reset base env and copy initial state
#     timestep_base = base_env.reset()
#     initial_state = base_env.physics.get_state()
#     timestep_sym = sym_env.reset()
#     sym_env.physics.set_state(initial_state)
#     sym_env.physics.forward()  # important to apply the state!
#
#     base_frames = []
#     for _ in range(max_steps):
#         obs_base = torch.FloatTensor(flatten_obs(timestep_base.observation)).unsqueeze(0).to(device)
#
#         with torch.no_grad():
#             base_action, _ = base_policy(obs_base)
#             sym_action, _ = sym_policy(obs_base)  # use same obs
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



def render_side_by_side(env, base_policy, sym_policy, device, max_steps=200):
    # Load the environment once
    base_env = suite.load("reacher", "easy")

    # Reset once and get initial state and goal
    timestep_base = base_env.reset()
    initial_state = base_env.physics.get_state()
    model_xml = base_env._model_xml  # get the same underlying model

    # Create a new env using the same model
    from dm_control.suite.reacher import reacher
    sym_env = reacher.environment(task_kwargs={"random": None})  # No new random seed

    # Set sym_env to same state as base_env
    sym_env.physics.set_state(initial_state)
    sym_env.physics.forward()

    base_frames = []

    for _ in range(max_steps):
        obs_base = torch.FloatTensor(flatten_obs(timestep_base.observation)).unsqueeze(0).to(device)

        with torch.no_grad():
            base_action, _ = base_policy(obs_base)
            sym_action, _ = sym_policy(obs_base)

        timestep_base = base_env.step(base_action.cpu().numpy()[0])
        timestep_sym = sym_env.step(sym_action.cpu().numpy()[0])

        frame_base = base_env.physics.render(camera_id=0, height=240, width=320)
        frame_sym = sym_env.physics.render(camera_id=0, height=240, width=320)

        combined = np.concatenate((frame_base, frame_sym), axis=1)
        annotated = cv2.putText(combined.copy(), "Base Policy", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        annotated = cv2.putText(annotated, "Symmetrized Policy", (370, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        base_frames.append(annotated)

        if timestep_base.last() or timestep_sym.last():
            break

    return base_frames




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