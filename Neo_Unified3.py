#!/usr/bin/env python3
"""
Unified Temporal Predictive Sweeping System with Cognitive Atlas
Drop-in script: TPS_Neo_Unified.py

This file combines your latent-space TPS explorer and adds:
 - CognitiveAtlas (graph of landmarks)
 - Consolidation (replay -> landmarks -> edges)
 - Planning (Dijkstra) + waypoint following
 - GUI buttons: Consolidate Memory, Save Neo, Load Neo
 - Atlas visualization in the cognitive plot

Requirements:
 - torch, diffusers (for VAE), numpy, scikit-learn, matplotlib, pillow
 - Optional: a GPU for faster decoding

Usage: python TPS_Neo_Unified.py

"""

import sys
import types
import os
# --- Triton autotuner monkeypatch ---
try:
    import triton.runtime
except ImportError:
    sys.modules["triton"] = types.ModuleType("triton")
    sys.modules["triton.runtime"] = types.ModuleType("triton.runtime")
    import triton.runtime

if not hasattr(triton.runtime, "Autotuner"):
    class DummyAutotuner:
        def __init__(self, *args, **kwargs): pass
        def tune(self, *args, **kwargs): return None
    triton.runtime.Autotuner = DummyAutotuner
from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from sklearn.decomposition import PCA
import warnings
import hashlib
warnings.filterwarnings("ignore")

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Running on: {device} with dtype: {torch_dtype}")

# ----------------------------- CognitiveAtlas -----------------------------
import json
from heapq import heappush, heappop

class CognitiveAtlas:
    """
    Simple graph-based cognitive atlas.
    Nodes: {node_id: {'latent': np.array, 'visits': int, 'label': str}}
    Edges: adjacency dict {a: {b: weight, ...}, ...}
    """
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.next_id = 0

    def add_node(self, latent_vec, label=None):
        nid = str(self.next_id)
        self.nodes[nid] = {
            'latent': np.array(latent_vec).astype(np.float32).reshape(-1),
            'visits': 1,
            'label': label or nid
        }
        self.edges[nid] = {}
        self.next_id += 1
        return nid

    def merge_node(self, nid, latent_vec):
        n = self.nodes[nid]
        alpha = 1.0 / (n['visits'] + 1)
        n['latent'] = (1 - alpha) * n['latent'] + alpha * np.array(latent_vec).reshape(-1)
        n['visits'] += 1

    def add_edge(self, a, b, weight=1.0):
        if a not in self.edges: self.edges[a] = {}
        if b not in self.edges: self.edges[b] = {}
        existing = self.edges[a].get(b, None)
        if existing is None or weight < existing:
            self.edges[a][b] = float(weight)
            self.edges[b][a] = float(weight)

    def find_nearest(self, latent_vec, metric='cosine'):
        if not self.nodes:
            return None, float('inf')
        lat = np.array(latent_vec).reshape(-1)
        best_id, best_d = None, float('inf')
        for nid, nd in self.nodes.items():
            v = nd['latent']
            if metric == 'cosine':
                dot = np.dot(lat, v)
                denom = (np.linalg.norm(lat) * np.linalg.norm(v) + 1e-10)
                d = 1.0 - (dot / denom)
            else:
                d = np.linalg.norm(lat - v)
            if d < best_d:
                best_d = d
                best_id = nid
        return best_id, best_d

    def dijkstra(self, start_nid, goal_nid):
        if start_nid not in self.nodes or goal_nid not in self.nodes:
            return []
        dist = {n: float('inf') for n in self.nodes}
        prev = {}
        dist[start_nid] = 0.0
        heap = [(0.0, start_nid)]
        while heap:
            d, u = heappop(heap)
            if u == goal_nid:
                break
            if d > dist[u]:
                continue
            for v, w in self.edges.get(u, {}).items():
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heappush(heap, (nd, v))
        if goal_nid not in prev and start_nid != goal_nid:
            return []
        path = [goal_nid]
        while path[-1] != start_nid:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def to_dict(self):
        return {
            'nodes': {nid: {'latent': nd['latent'].tolist(), 'visits': nd['visits'], 'label': nd['label']} for nid, nd in self.nodes.items()},
            'edges': self.edges,
            'next_id': self.next_id
        }

    @classmethod
    def from_dict(cls, d):
        atlas = cls()
        atlas.nodes = {nid: {'latent': np.array(v['latent'], dtype=np.float32), 'visits': v['visits'], 'label': v['label']} for nid, v in d['nodes'].items()}
        atlas.edges = {k: {kk: float(vv) for kk, vv in val.items()} for k, val in d['edges'].items()}
        atlas.next_id = d.get('next_id', len(atlas.nodes))
        return atlas

# ============================================================================ #
#               ORIGINAL TPS COMPONENTS (slightly adapted)                    #
# ============================================================================ #

class SensoryEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feature_dim)
        )
        self.salience_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, image_tensor):
        features = self.encoder(image_tensor)
        feature_vector = torch.tanh(features)
        salience_score = self.salience_head(features)
        return feature_vector, salience_score

class SweepGenerator(nn.Module):
    def __init__(self, sweep_angle_degrees=30.0):
        super().__init__()
        self.sweep_angle_offset = sweep_angle_degrees
        self.current_sweep_side = 1
        self.theta_phase = 0.0
    def get_sweep_direction(self, current_heading_vector):
        self.current_sweep_side *= -1
        angle_rad = self.current_sweep_side * self.sweep_angle_offset * (math.pi / 180)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        if len(current_heading_vector.shape) == 1:
            current_heading_vector = current_heading_vector.unsqueeze(0)
        x = current_heading_vector[:, 0]
        y = current_heading_vector[:, 1]
        new_x = x * cos_a - y * sin_a
        new_y = x * sin_a + y * cos_a
        sweep_direction_vector = torch.stack([new_x, new_y], dim=1)
        return sweep_direction_vector

class SequenceProcessor(nn.Module):
    def __init__(self, latent_dim=4096, n_modules=3, scale_factor=1.5, memory_size=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_modules = n_modules
        self.scales = [scale_factor**i for i in range(n_modules)]
        self.sweep_base_length = 0.1
        self.current_state = nn.Parameter(torch.randn(1, latent_dim, dtype=torch_dtype, device=device) * 0.01, requires_grad=False)
        self.memory_size = memory_size
        self.path_memory = deque(maxlen=memory_size)
        self.state_history_2d = deque(maxlen=100)
        self.pca = None
        self.module_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim)
            ) for _ in range(n_modules)
        ])
        self.sequence_predictor = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True
        )

    def anchor_state(self, new_latent_flat, salience_score):
        with torch.no_grad():
            new_latent = new_latent_flat.detach()
            if new_latent.dtype != self.current_state.dtype:
                new_latent = new_latent.to(self.current_state.dtype)
            if new_latent.device != self.current_state.device:
                new_latent = new_latent.to(self.current_state.device)
            salience_value = salience_score.item() if torch.is_tensor(salience_score) else float(salience_score)
            self.current_state.mul_(1 - salience_value).add_(new_latent * salience_value)
        self.path_memory.append(self.current_state.detach().clone())
        self._update_2d_projection()

    def _update_2d_projection(self):
        try:
            current_state_np = self.current_state.detach().cpu().numpy()
            if len(self.path_memory) >= 3:
                recent_states = list(self.path_memory)[-50:]
                states_np = np.vstack([s.detach().cpu().numpy() for s in recent_states])
                if self.pca is None or len(self.path_memory) % 10 == 0:
                    try:
                        self.pca = PCA(n_components=2)
                        self.pca.fit(states_np)
                    except Exception as e:
                        self.pca = None
                        return
                if self.pca is not None:
                    current_2d = self.pca.transform(current_state_np)
                    current_2d = np.clip(current_2d, -10.0, 10.0)
                    self.state_history_2d.append(current_2d[0])
            else:
                self.state_history_2d.append(np.array([0.0, 0.0]))
        except Exception as e:
            print(f"_update_2d_projection error: {e}")

    def predict_sweep_trajectories(self, displacement_vector, current_for_sweep=None):
        if current_for_sweep is None:
            current_for_sweep = self.current_state
        multi_scale_sweeps = []
        for i, scale in enumerate(self.scales):
            sweep_length = self.sweep_base_length * scale
            n_steps = max(5, int(10 * scale))
            # normalize/scale displacement: displacement_vector may be [1, D]
            disp = displacement_vector
            if disp.dim() == 2 and disp.shape[0] == 1:
                disp = disp
            else:
                disp = displacement_vector.unsqueeze(0)
            # scale
            disp_norm = F.normalize(disp, p=2, dim=-1) * sweep_length
            trajectory = []
            for step in range(n_steps):
                t = (step / (n_steps - 1)) if n_steps > 1 else 0.0
                offset = disp_norm * t
                processed_offset = self.module_processors[i](offset)
                state_at_position = current_for_sweep + processed_offset
                trajectory.append(state_at_position)
            sweep_trajectory = torch.stack(trajectory, dim=0)
            multi_scale_sweeps.append(sweep_trajectory)
        return multi_scale_sweeps

class TemporalBinder(nn.Module):
    def __init__(self, latent_dim=4096):
        super().__init__()
        self.latent_dim = latent_dim
        self.mixer_network = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh()
        )
    def forward(self, vector_a, vector_b):
        mixed = vector_a * vector_b
        concatenated = torch.cat([vector_a, vector_b], dim=-1)
        bound_vector = self.mixer_network(concatenated)
        bound_vector = bound_vector + torch.tanh(mixed) * 0.1
        return bound_vector

class PhaseEncoder(nn.Module):
    def __init__(self, latent_dim=4096, n_phase_channels=7):
        super().__init__()
        self.n_phase_channels = n_phase_channels
        self.latent_dim = latent_dim
        self.to_phase = nn.Linear(latent_dim, n_phase_channels * 2)
    def forward(self, feature_vector, theta_phase):
        phase_components = self.to_phase(feature_vector)
        phase_components = phase_components.reshape(-1, self.n_phase_channels, 2)
        real = phase_components[..., 0]
        imag = phase_components[..., 1]
        phase_offset = theta_phase * 2 * math.pi
        rotated_real = real * math.cos(phase_offset) - imag * math.sin(phase_offset)
        rotated_imag = real * math.sin(phase_offset) + imag * math.cos(phase_offset)
        phase_encoded = torch.cat([rotated_real, rotated_imag], dim=-1)
        return phase_encoded.flatten(1)

class ThetaGammaOscillator:
    def __init__(self, theta_freq=8.0, gamma_freq=60.0):
        self.theta_freq = theta_freq
        self.gamma_freq = gamma_freq
        self.time_start = time.time()
    def get_phases(self):
        current_time = time.time() - self.time_start
        theta_phase = (current_time * self.theta_freq) % 1.0
        gamma_phase = (current_time * self.gamma_freq) % 1.0
        return theta_phase, gamma_phase
    def get_gamma_packet_index(self, theta_phase):
        return int(theta_phase * 7)

# ============================================================================ #
#                       TEMPORAL PREDICTIVE SWEEPING SYSTEM                   #
# ============================================================================ #

class TemporalPredictiveSweepingSystem:
    def __init__(self, vae_model):
        assert vae_model is not None, "VAE model required for latent space exploration"
        self.vae_model = vae_model
        self.scaling_factor = getattr(vae_model.config, 'scaling_factor', 0.18215)
        self.latent_channels = 4
        self.latent_h = 32
        self.latent_w = 32
        self.latent_dim = self.latent_channels * self.latent_h * self.latent_w
        # Cast all custom modules to the correct device AND data type
        self.sensory_encoder = SensoryEncoder().to(device, dtype=torch_dtype)
        self.sweep_generator = SweepGenerator()
        self.sequence_processor = SequenceProcessor(latent_dim=self.latent_dim).to(device, dtype=torch_dtype)
        self.temporal_binder = TemporalBinder(latent_dim=self.latent_dim).to(device, dtype=torch_dtype)
        self.phase_encoder = PhaseEncoder(latent_dim=self.latent_dim).to(device, dtype=torch_dtype)
        self.direction_projector = nn.Linear(2, self.latent_dim, bias=False).to(device, dtype=torch_dtype)
        nn.init.normal_(self.direction_projector.weight, mean=0.0, std=0.001)
        self.oscillator = ThetaGammaOscillator()
        self.theta_cycle_duration = 0.125
        self.last_theta_time = time.time()
        self.current_heading = torch.tensor([1.0, 0.0], device=device, dtype=torch_dtype)
        self.goal_vector = torch.zeros(1, self.latent_dim, device=device, dtype=torch_dtype)
        self.salience_history = deque(maxlen=200)
        self.theta_history = deque(maxlen=200)
        self.sweep_history = deque(maxlen=50)
        # Cognitive atlas + planning
        self.atlas = CognitiveAtlas()
        self.current_plan = []
        self.current_waypoint_idx = 0
        self.waypoint_tolerance = 0.5
        self.last_result = None

    def set_goal(self, goal_description):
        seed = int(hashlib.sha256(goal_description.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        goal_array = rng.normal(0, 0.01, size=(self.latent_dim,))
        self.goal_vector = torch.tensor(goal_array, device=device, dtype=self.current_heading.dtype).unsqueeze(0)

    def consolidate_memory(self, episode_length=200, downsample_step=5, merge_thresh=0.15):
        episode = list(self.sequence_processor.path_memory)[-episode_length:]
        if not episode:
            return {'added': 0, 'merged': 0, 'edges': 0}
        episode_np = np.vstack([s.detach().cpu().numpy().reshape(-1) for s in episode])
        samples = episode_np[::downsample_step]
        added, merged, edges = 0, 0, 0
        last_nid = None
        for s in samples:
            nid, d = self.atlas.find_nearest(s, metric='cosine')
            if nid is None or d > merge_thresh:
                new_nid = self.atlas.add_node(s)
                added += 1
                nid_to_use = new_nid
            else:
                self.atlas.merge_node(nid, s)
                merged += 1
                nid_to_use = nid
            if last_nid is not None and last_nid != nid_to_use:
                wa = self.atlas.nodes[last_nid]['latent']
                wb = self.atlas.nodes[nid_to_use]['latent']
                w = float(np.linalg.norm(wa - wb) + 1e-6)
                self.atlas.add_edge(last_nid, nid_to_use, weight=w)
                edges += 1
            last_nid = nid_to_use
        return {'added': added, 'merged': merged, 'edges': edges}

    def find_path(self, label_or_latent_a, label_or_latent_b):
        def resolve(x):
            if isinstance(x, str):
                if x in self.atlas.nodes:
                    return x
                for nid, nd in self.atlas.nodes.items():
                    if nd.get('label') == x:
                        return nid
                return None
            else:
                nid, d = self.atlas.find_nearest(np.array(x).reshape(-1), metric='cosine')
                return nid
        a_id = resolve(label_or_latent_a)
        b_id = resolve(label_or_latent_b)
        if a_id is None or b_id is None:
            return []
        return self.atlas.dijkstra(a_id, b_id)

    def follow_path(self, node_list):
        if not node_list:
            self.current_plan = []
            self.current_waypoint_idx = 0
            return
        self.current_plan = node_list
        self.current_waypoint_idx = 0

    def update(self):
        current_time = time.time()
        if current_time - self.last_theta_time < self.theta_cycle_duration:
            return None
        self.last_theta_time = current_time
        theta_phase, gamma_phase = self.oscillator.get_phases()
        gamma_packet = self.oscillator.get_gamma_packet_index(theta_phase)
        self.theta_history.append((current_time, theta_phase))
        # 1. Decode current state
        current_state = self.sequence_processor.current_state
        current_latent_reshaped = current_state.view(1, self.latent_channels, self.latent_h, self.latent_w)
        with torch.no_grad():
            # Input to VAE must be float32
            vae_input = (current_latent_reshaped / self.scaling_factor)
            # Make sure it's float32 for VAE decode
            if vae_input.dtype != torch.float32:
                vae_input = vae_input.to(torch.float32)
            current_sample_float32 = self.vae_model.decode(vae_input).sample

        # Convert VAE's float32 output to the system's dtype (float16) for custom models
        current_image_tensor = current_sample_float32.to(self.sequence_processor.current_state.dtype)

        # The numpy conversion for display can still use the float32 version
        current_image_np = ((current_sample_float32[0].permute(1, 2, 0).cpu() * 0.5 + 0.5) * 255).round().clamp(0, 255).byte().numpy()
        _, salience = self.sensory_encoder(current_image_tensor)

        salience_value = salience.item() if torch.is_tensor(salience) else float(salience)

        with torch.no_grad():
            # FIX: Cast to float32 before VAE encode to match VAE's dtype
            current_image_tensor_for_encode = current_image_tensor.to(torch.float32)
            posterior = self.vae_model.encode(current_image_tensor_for_encode).latent_dist.sample()
            new_latent = posterior * self.scaling_factor
        new_latent_flat = new_latent.view(1, -1)
        # anchor
        self.sequence_processor.anchor_state(new_latent_flat, salience_value)
        current_state = self.sequence_processor.current_state
        goal = self.goal_vector.to(current_state.device, current_state.dtype)
        bound_state = self.temporal_binder(current_state, goal)
        # planning influence
        if hasattr(self, 'current_plan') and self.current_plan:
            idx = max(0, min(self.current_waypoint_idx, len(self.current_plan)-1))
            waypoint_nid = self.current_plan[idx]
            waypoint_latent = torch.tensor(self.atlas.nodes[waypoint_nid]['latent'], device=self.sequence_processor.current_state.device, dtype=self.sequence_processor.current_state.dtype).unsqueeze(0)
            waypoint_delta = (waypoint_latent - self.sequence_processor.current_state)
            dist = torch.norm(waypoint_delta).item()
            if dist < self.waypoint_tolerance:
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx >= len(self.current_plan):
                    self.current_plan = []
                    self.current_waypoint_idx = 0
                else:
                    waypoint_nid = self.current_plan[self.current_waypoint_idx]
                    waypoint_latent = torch.tensor(self.atlas.nodes[waypoint_nid]['latent'], device=self.sequence_processor.current_state.device, dtype=self.sequence_processor.current_state.dtype).unsqueeze(0)
                    waypoint_delta = (waypoint_latent - self.sequence_processor.current_state)
            bound_state = bound_state + waypoint_delta * 0.3
        phase_encoded_state = self.phase_encoder(bound_state, theta_phase)
        sweep_dir_2d = self.sweep_generator.get_sweep_direction(self.current_heading)
        self.current_heading = sweep_dir_2d
        sweep_side = "LEFT" if self.sweep_generator.current_sweep_side < 0 else "RIGHT"
        self.sweep_history.append((current_time, sweep_side, sweep_dir_2d.cpu().numpy()))
        displacement_base = self.direction_projector(sweep_dir_2d.to(self.direction_projector.weight.device)).to(self.sequence_processor.current_state.dtype)
        predicted_trajectories = self.sequence_processor.predict_sweep_trajectories(displacement_base, current_for_sweep=bound_state)
        predicted_end = predicted_trajectories[0][-1]
        pred_latent_reshaped = predicted_end.view(1, self.latent_channels, self.latent_h, self.latent_w)
        with torch.no_grad():
            # FIX: Cast to float32 before VAE decode to match VAE's dtype
            pred_vae_input = (pred_latent_reshaped / self.scaling_factor).to(torch.float32)
            pred_sample = self.vae_model.decode(pred_vae_input).sample
        visual_prediction_np = ((pred_sample[0].permute(1, 2, 0).cpu() * 0.5 + 0.5) * 255).round().clamp(0, 255).byte().numpy()
        # action: small step toward prediction
        step_size = 0.1
        delta = (predicted_end - self.sequence_processor.current_state) * step_size
        with torch.no_grad():
            self.sequence_processor.current_state.add_(delta)
        self.sequence_processor._update_2d_projection()
        # store last_result for GUI access
        self.last_result = {
            'current_visual': current_image_np,
            'visual_prediction': visual_prediction_np,
            'trajectories': predicted_trajectories,
            'theta_phase': theta_phase,
            'gamma_packet': gamma_packet,
            'sweep_direction': sweep_dir_2d,
            'sweep_side': sweep_side,
            'phase_encoded_state': phase_encoded_state,
            'salience': salience_value,
            'current_state_2d': self.sequence_processor.state_history_2d[-1] if self.sequence_processor.state_history_2d else None
        }
        return self.last_result

# ============================================================================ #
#                            VISUALIZATION / GUI                               #
# ============================================================================ #

class TPSVisualizationDemo:
    def __init__(self, root, vae_model):
        self.root = root
        self.root.title("Temporal Predictive Sweeping System - Latent Space Explorer")
        self.root.geometry("1400x900")
        self.tps_system = TemporalPredictiveSweepingSystem(vae_model=vae_model)
        self.current_frame = None
        self.running = False
        self.setup_gui()
        self.capture_thread = threading.Thread(target=self.dummy_loop, daemon=True)
        self.capture_thread.start()
        self.process_thread = threading.Thread(target=self.process_loop, daemon=True)
        self.process_thread.start()
        self.start_animations()

    def dummy_loop(self):
        while True:
            time.sleep(0.03)

    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=5)
        control_frame.pack(fill=tk.X, pady=2)
        self.toggle_btn = ttk.Button(control_frame, text="Start Exploration", command=self.toggle_processing)
        self.toggle_btn.pack(side=tk.LEFT, padx=5)
        ttk.Label(control_frame, text="Goal:").pack(side=tk.LEFT, padx=5)
        self.goal_var = tk.StringVar(value="explore dreamlike landscapes")
        self.goal_entry = ttk.Entry(control_frame, textvariable=self.goal_var, width=40)
        self.goal_entry.pack(side=tk.LEFT, padx=5)
        self.goal_entry.bind('<Return>', self.update_goal)
        # Consolidate, Save, Load buttons
        self.consolidate_btn = ttk.Button(control_frame, text="Consolidate Memory", command=self.on_consolidate)
        self.consolidate_btn.pack(side=tk.LEFT, padx=5)
        self.save_btn = ttk.Button(control_frame, text="Save Neo", command=self.on_save_neo)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        self.load_btn = ttk.Button(control_frame, text="Load Neo", command=self.on_load_neo)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_rowconfigure(1, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)
        self.setup_sensory_panel(content_frame)
        self.setup_cognitive_panel(content_frame)
        self.setup_prediction_panel(content_frame)
        self.setup_rhythms_panel(content_frame)

    def setup_sensory_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Current Latent View & Salience", padding=5)
        panel.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.video_label = ttk.Label(panel, text="Latent View")
        self.video_label.pack(pady=5)
        self.salience_frame = ttk.Frame(panel)
        self.salience_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.salience_frame, text="Salience:").pack(side=tk.LEFT)
        self.salience_var = tk.DoubleVar()
        self.salience_progress = ttk.Progressbar(self.salience_frame, variable=self.salience_var, maximum=1.0)
        self.salience_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.salience_label = ttk.Label(self.salience_frame, text="0.000")
        self.salience_label.pack(side=tk.LEFT)

    def setup_cognitive_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Cognitive Space & Sweeps", padding=5)
        panel.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self.cog_fig = Figure(figsize=(6, 4), dpi=80)
        self.cog_ax = self.cog_fig.add_subplot(111)
        self.cog_ax.set_title("Internal State & Predictive Sweeps")
        self.cog_ax.grid(True, alpha=0.3)
        self.cog_canvas = FigureCanvasTkAgg(self.cog_fig, panel)
        self.cog_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_state_dot, = self.cog_ax.plot([], [], 'ro', markersize=10, label='Current State')
        self.path_trail, = self.cog_ax.plot([], [], 'b-', alpha=0.6, label='Path Memory')
        self.sweep_arrow = None
        self.sweep_beams = []
        self.cog_ax.legend()

    def setup_prediction_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Mind's Eye Prediction", padding=5)
        panel.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.prediction_image_label = ttk.Label(panel, text="Predicted View")
        self.prediction_image_label.pack(expand=True, fill=tk.BOTH)
        self.direction_label = ttk.Label(panel, text="Direction: --", font=("Arial", 12, "bold"))
        self.direction_label.pack(pady=5)

    def setup_rhythms_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="System Rhythms", padding=5)
        panel.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)
        self.rhythm_fig = Figure(figsize=(6, 4), dpi=80)
        self.theta_ax = self.rhythm_fig.add_subplot(211)
        self.theta_ax.set_title("Theta Rhythm (~8 Hz)")
        self.theta_ax.set_xlim(0, 2*np.pi)
        self.theta_ax.set_ylim(-1.2, 1.2)
        self.theta_ax.grid(True, alpha=0.3)
        self.salience_ax = self.rhythm_fig.add_subplot(212)
        self.salience_ax.set_title("Salience History")
        self.salience_ax.set_xlim(-10, 0)
        self.salience_ax.set_ylim(0, 1)
        self.salience_ax.grid(True, alpha=0.3)
        self.rhythm_fig.tight_layout()
        self.rhythm_canvas = FigureCanvasTkAgg(self.rhythm_fig, panel)
        self.rhythm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        theta_x = np.linspace(0, 2*np.pi, 100)
        theta_y = np.sin(theta_x)
        self.theta_line, = self.theta_ax.plot(theta_x, theta_y, 'b-', linewidth=2)
        self.theta_dot, = self.theta_ax.plot([], [], 'ro', markersize=8)
        self.salience_line, = self.salience_ax.plot([], [], 'g-', linewidth=2)
        status_frame = ttk.Frame(panel)
        status_frame.pack(fill=tk.X, pady=5)
        self.theta_status = ttk.Label(status_frame, text="Theta: --")
        self.theta_status.grid(row=0, column=0, padx=10)
        self.gamma_status = ttk.Label(status_frame, text="Gamma: --")
        self.gamma_status.grid(row=0, column=1, padx=10)
        self.sweep_status = ttk.Label(status_frame, text="Sweep: --")
        self.sweep_status.grid(row=0, column=2, padx=10)

    def start_animations(self):
        self.anim_cognitive = animation.FuncAnimation(self.cog_fig, self.update_cognitive_plot, interval=50, blit=False)
        self.anim_rhythms = animation.FuncAnimation(self.rhythm_fig, self.update_rhythms_plot, interval=50, blit=False)

    def update_cognitive_plot(self, frame):
        if not self.running or not self.tps_system.sequence_processor.state_history_2d:
            return
        if len(self.tps_system.sequence_processor.state_history_2d) > 0:
            current_pos = self.tps_system.sequence_processor.state_history_2d[-1]
            self.current_state_dot.set_data([current_pos[0]], [current_pos[1]])
            if len(self.tps_system.sequence_processor.state_history_2d) > 1:
                trail = list(self.tps_system.sequence_processor.state_history_2d)[-20:]
                trail_x = [p[0] for p in trail]
                trail_y = [p[1] for p in trail]
                self.path_trail.set_data(trail_x, trail_y)
                # autoscale bounds
                min_x, max_x = min(trail_x), max(trail_x)
                min_y, max_y = min(trail_y), max(trail_y)
                pad_x = max(0.1, (max_x - min_x) * 0.2)
                pad_y = max(0.1, (max_y - min_y) * 0.2)
                self.cog_ax.set_xlim(min_x - pad_x, max_x + pad_x)
                self.cog_ax.set_ylim(min_y - pad_y, max_y + pad_y)
            # draw atlas nodes & edges
            atlas = getattr(self.tps_system, 'atlas', None)
            pca = self.tps_system.sequence_processor.pca
            if atlas is not None and atlas.nodes and pca is not None:
                node_coords = []
                for nid, nd in atlas.nodes.items():
                    try:
                        coord = pca.transform(nd['latent'].reshape(1, -1))[0]
                        node_coords.append((nid, coord))
                    except Exception:
                        continue
                if node_coords:
                    xs = [c[1][0] for c in node_coords]
                    ys = [c[1][1] for c in node_coords]
                    self.cog_ax.scatter(xs, ys, s=40, marker='o', edgecolors='black', zorder=4)
                    for a, nbrs in atlas.edges.items():
                        a_coord = pca.transform(atlas.nodes[a]['latent'].reshape(1, -1))[0]
                        for b, w in nbrs.items():
                            b_coord = pca.transform(atlas.nodes[b]['latent'].reshape(1, -1))[0]
                            self.cog_ax.plot([a_coord[0], b_coord[0]], [a_coord[1], b_coord[1]], color='gray', alpha=0.4, linewidth=1, zorder=2)
            # Sweep arrow and beams using predicted endpoint if available
            if len(self.tps_system.sweep_history) > 0:
                _, sweep_side, sweep_dir = self.tps_system.sweep_history[-1]
                if self.sweep_arrow is not None:
                    try:
                        self.sweep_arrow.remove()
                    except Exception:
                        pass
                for beam in self.sweep_beams:
                    try:
                        beam.remove()
                    except Exception:
                        pass
                self.sweep_beams = []
                arrow_end_x, arrow_end_y = current_pos[0] + 0.2, current_pos[1]
                # try to use predicted endpoint from last_result
                last = getattr(self.tps_system, 'last_result', None)
                if last and last.get('trajectories') is not None and pca is not None:
                    try:
                        last_traj = last['trajectories'][0]
                        end_tensor = last_traj[-1].detach().cpu().numpy().reshape(1, -1)
                        end_2d = pca.transform(end_tensor)[0]
                        arrow_end_x, arrow_end_y = end_2d[0], end_2d[1]
                    except Exception:
                        pass
                self.sweep_arrow = self.cog_ax.annotate('', xy=(arrow_end_x, arrow_end_y), xytext=(current_pos[0], current_pos[1]), arrowprops=dict(arrowstyle='->', lw=2, color='red'), annotation_clip=False)
                scales = [1.0, 1.5, 2.25]
                colors = ['orange', 'yellow', 'green']
                alphas = [0.7, 0.5, 0.3]
                for i, (scale, color, alpha) in enumerate(zip(scales, colors, alphas)):
                    beam_length = 0.3 * scale
                    beam_end_x = current_pos[0] + (arrow_end_x - current_pos[0]) * (0.2 + 0.1 * i)
                    beam_end_y = current_pos[1] + (arrow_end_y - current_pos[1]) * (0.2 + 0.1 * i)
                    beam = self.cog_ax.plot([current_pos[0], beam_end_x], [current_pos[1], beam_end_y], color=color, alpha=alpha, linewidth=3+i)[0]
                    self.sweep_beams.append(beam)
        self.cog_fig.canvas.draw()
        self.cog_canvas.flush_events()

    def update_rhythms_plot(self, frame):
        if not self.running:
            return
        theta_phase, _ = self.tps_system.oscillator.get_phases()
        theta_position = theta_phase * 2 * np.pi
        theta_y = np.sin(theta_position)
        self.theta_dot.set_data([theta_position], [theta_y])
        current_time = time.time()
        if len(self.tps_system.salience_history) > 1:
            times = [t for t, _ in self.tps_system.salience_history]
            saliences = [s for _, s in self.tps_system.salience_history]
            rel_times = [(t - current_time) for t in times]
            self.salience_line.set_data(rel_times, saliences)
            if rel_times:
                self.salience_ax.set_xlim(min(rel_times), 0)
        self.rhythm_fig.canvas.draw()
        self.rhythm_canvas.flush_events()

    def toggle_processing(self):
        self.running = not self.running
        self.toggle_btn.config(text="Stop Exploration" if self.running else "Start Exploration")

    def update_goal(self, event=None):
        goal_text = self.goal_var.get()
        self.tps_system.set_goal(goal_text)

    def process_loop(self):
        while True:
            try:
                if self.running:
                    result = self.tps_system.update()
                    if result is not None:
                        self.root.after(0, lambda r=result: self.update_gui(r))
                time.sleep(0.01)
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(0.1)

    def update_gui(self, result):
        try:
            if 'current_visual' in result:
                cv = result['current_visual']
                pil = Image.fromarray(cv)
                resized = pil.resize((320, 240), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(resized)
                self.video_label.config(image=photo)
                self.video_label.image = photo
            if 'visual_prediction' in result:
                pred = result['visual_prediction']
                pil = Image.fromarray(pred)
                resized = pil.resize((320, 320), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(resized)
                self.prediction_image_label.config(image=photo)
                self.prediction_image_label.image = photo
            salience = result['salience']
            self.salience_var.set(salience)
            self.salience_label.config(text=f"{salience:.3f}")
            if salience > 0.7:
                self.video_label.config(relief="solid", borderwidth=3)
                self.root.after(200, lambda: self.video_label.config(relief="flat", borderwidth=0))
            self.theta_status.config(text=f"Theta: {result['theta_phase']:.2f}")
            self.gamma_status.config(text=f"Gamma: {result['gamma_packet']}")
            self.sweep_status.config(text=f"Sweep: {result['sweep_side']}")
            self.direction_label.config(text=f"Prediction from {result['sweep_side']} Sweep", foreground="blue" if result['sweep_side'] == "LEFT" else "red")
        except Exception as e:
            print(f"GUI update error: {e}")

    # ----------------- Consolidate / Save / Load Handlers -----------------
    def on_consolidate(self):
        def _do():
            self.running = False
            time.sleep(0.05)
            stats = self.tps_system.consolidate_memory(episode_length=400, downsample_step=6, merge_thresh=0.12)
            print(f"[Consolidate] added={stats['added']} merged={stats['merged']} edges={stats['edges']}")
            self.running = True
        threading.Thread(target=_do, daemon=True).start()

    def on_save_neo(self):
        import tkinter.filedialog as fd
        fname = fd.asksaveasfilename(defaultextension=".pt", filetypes=[("Neo Save", "*.pt")])
        if not fname:
            return
        data = {
            'atlas': self.tps_system.atlas.to_dict(),
            'current_state': self.tps_system.sequence_processor.current_state.detach().cpu(),
            'path_memory': [s.detach().cpu() for s in self.tps_system.sequence_processor.path_memory]
        }
        try:
            torch.save(data, fname)
            print(f"[Save Neo] saved to {fname}")
        except Exception as e:
            print(f"Save error: {e}")

    def on_load_neo(self):
        import tkinter.filedialog as fd
        fname = fd.askopenfilename(filetypes=[("Neo Save", "*.pt")])
        if not fname:
            return
        try:
            loaded = torch.load(fname, map_location='cpu')
            atlas_d = loaded.get('atlas', None)
            if atlas_d is not None:
                self.tps_system.atlas = CognitiveAtlas.from_dict(atlas_d)
            cs = loaded.get('current_state', None)
            if cs is not None:
                cs_t = cs.to(device=self.tps_system.sequence_processor.current_state.device, dtype=self.tps_system.sequence_processor.current_state.dtype)
                with torch.no_grad():
                    self.tps_system.sequence_processor.current_state.copy_(cs_t)
            pm = loaded.get('path_memory', [])
            self.tps_system.sequence_processor.path_memory = deque([x.to(device=self.tps_system.sequence_processor.current_state.device, dtype=self.tps_system.sequence_processor.current_state.dtype) for x in pm], maxlen=self.tps_system.sequence_processor.memory_size)
            print(f"[Load Neo] loaded {fname}")
        except Exception as e:
            print(f"Load error: {e}")

    def cleanup(self):
        self.running = False
        self.root.quit()

# ============================================================================ #
#                                    MAIN                                      #
# ============================================================================ #

def main():
    print("=" * 60)
    print("TEMPORAL PREDICTIVE SWEEPING SYSTEM - NEO UNIFIED")
    print("=" * 60)
    print(f"Running on: {device} with dtype: {torch_dtype}")
    print()
    try:
        from diffusers import AutoencoderKL
    except Exception:
        print("Diffusers library not found. Please install: pip install diffusers transformers"); return
    print("Loading pre-trained VAE...")
    try:
        MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
        vae = AutoencoderKL.from_pretrained(
            MODEL_ID, 
            subfolder="vae", 
            torch_dtype=torch.float32  # Always float32 for VAE
        ).to(device)

        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
        print("VAE loaded successfully.")
    except Exception as e:
        print(f"Could not load VAE model. Error: {e}")
        return
    root = tk.Tk()
    try:
        app = TPSVisualizationDemo(root, vae_model=vae)
        root.protocol("WM_DELETE_WINDOW", app.cleanup)
        print("Starting GUI... Click 'Start Exploration' to begin.")
        root.mainloop()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error running application: {e}")
    finally:
        if 'app' in locals():
            app.cleanup()

if __name__ == "__main__":
    main()