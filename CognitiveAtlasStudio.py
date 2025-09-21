#!/usr/bin/env python3
"""
Cognitive Atlas Studio - Interactive World Builder & Storyboard Generator
Built on Neo's cognitive architecture for autonomous creative exploration

This system allows users to:
1. Seed a creative universe with a theme
2. Watch Neo autonomously explore and map the conceptual space
3. Interact with the discovered cognitive atlas
4. Generate storyboards and animated sequences between concepts
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import json
import os
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import threading
import time
from datetime import datetime

# Import core Neo components (assumes Neo_Unified3.py exists)
try:
    from Neo_Unified3 import (
        CognitiveAtlas, 
        TemporalPredictiveSweepingSystem,
        device, torch_dtype
    )
except ImportError:
    print("Neo_Unified3.py not found. Using stub implementations.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

@dataclass
class StoryboardFrame:
    """Single frame in a storyboard sequence"""
    latent_vector: np.ndarray
    image: Optional[np.ndarray] = None
    label: str = ""
    timestamp: float = 0.0
    metadata: Dict = None

class StoryboardGenerator:
    """Generates smooth transitions between concepts in latent space"""
    
    def __init__(self, vae_model=None):
        self.vae = vae_model
        self.interpolation_methods = {
            'linear': self.linear_interpolation,
            'spherical': self.spherical_interpolation,
            'bezier': self.bezier_interpolation
        }
    
    def linear_interpolation(self, start, end, steps):
        """Simple linear interpolation in latent space"""
        alphas = np.linspace(0, 1, steps)
        return [(1-a) * start + a * end for a in alphas]
    
    def spherical_interpolation(self, start, end, steps):
        """Spherical linear interpolation (slerp) for smoother transitions"""
        start_norm = start / (np.linalg.norm(start) + 1e-10)
        end_norm = end / (np.linalg.norm(end) + 1e-10)
        
        dot = np.clip(np.dot(start_norm, end_norm), -1, 1)
        theta = np.arccos(dot)
        
        if theta < 0.01:  # Too close, use linear
            return self.linear_interpolation(start, end, steps)
        
        sin_theta = np.sin(theta)
        alphas = np.linspace(0, 1, steps)
        frames = []
        
        for a in alphas:
            s0 = np.sin((1-a) * theta) / sin_theta
            s1 = np.sin(a * theta) / sin_theta
            frames.append(s0 * start + s1 * end)
        
        return frames
    
    def bezier_interpolation(self, start, end, steps, control_points=None):
        """Bezier curve interpolation for artistic transitions"""
        if control_points is None:
            # Generate control points based on PCA of the path
            mid = (start + end) / 2
            offset = np.random.randn(*start.shape) * 0.1
            cp1 = mid + offset
            cp2 = mid - offset
            control_points = [cp1, cp2]
        
        t = np.linspace(0, 1, steps)
        frames = []
        
        for ti in t:
            # Cubic bezier
            b = ((1-ti)**3 * start + 
                 3*(1-ti)**2*ti * control_points[0] + 
                 3*(1-ti)*ti**2 * control_points[1] + 
                 ti**3 * end)
            frames.append(b)
        
        return frames
    
    def generate_transition(self, start_latent, end_latent, frames=30, 
                           method='spherical', ease_func=None):
        """Generate smooth transition between two latent vectors"""
        
        interpolator = self.interpolation_methods.get(method, self.linear_interpolation)
        latent_frames = interpolator(start_latent, end_latent, frames)
        
        # Apply easing function if provided
        if ease_func:
            latent_frames = self.apply_easing(latent_frames, ease_func)
        
        storyboard = []
        for i, latent in enumerate(latent_frames):
            frame = StoryboardFrame(
                latent_vector=latent,
                timestamp=i / 30.0,  # Assume 30fps
                metadata={'interpolation': method}
            )
            storyboard.append(frame)
        
        return storyboard
    
    def apply_easing(self, frames, ease_type='ease_in_out'):
        """Apply easing functions for more natural motion"""
        n = len(frames)
        
        if ease_type == 'ease_in':
            weights = [t**2 for t in np.linspace(0, 1, n)]
        elif ease_type == 'ease_out':
            weights = [1 - (1-t)**2 for t in np.linspace(0, 1, n)]
        elif ease_type == 'ease_in_out':
            weights = [(1 - np.cos(t * np.pi)) / 2 for t in np.linspace(0, 1, n)]
        else:
            return frames
        
        # Apply weights
        start, end = frames[0], frames[-1]
        return [start + w * (end - start) for w in weights]

class WorldBuilder:
    """Main cognitive world-building engine"""
    
    def __init__(self, neo_system, vae_model):
        self.neo = neo_system
        self.vae = vae_model
        self.storyboard_gen = StoryboardGenerator(vae_model)
        self.worlds = {}  # Dictionary of saved worlds
        self.current_world = None
        
    def seed_world(self, theme_prompt, exploration_time=60):
        """Initialize Neo with a theme and let it explore"""
        # Create new world
        world_id = f"{theme_prompt}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_world = {
            'id': world_id,
            'theme': theme_prompt,
            'atlas': CognitiveAtlas(),
            'exploration_history': [],
            'storyboards': []
        }
        
        # Set Neo's goal based on theme
        self.neo.set_goal(theme_prompt)
        
        # Let Neo explore autonomously
        start_time = time.time()
        while time.time() - start_time < exploration_time:
            result = self.neo.update()
            if result:
                # Record exploration
                self.current_world['exploration_history'].append({
                    'timestamp': time.time() - start_time,
                    'state': self.neo.sequence_processor.current_state.cpu().numpy()
                })
                
                # Periodically consolidate memories into atlas
                if len(self.current_world['exploration_history']) % 50 == 0:
                    self.consolidate_discoveries()
        
        # Final consolidation
        self.consolidate_discoveries()
        self.worlds[world_id] = self.current_world
        
        return self.current_world
    
    def consolidate_discoveries(self):
        """Convert Neo's path memory into cognitive atlas landmarks"""
        if not self.current_world:
            return
        
        # Use Neo's consolidation but store in our world's atlas
        stats = self.neo.consolidate_memory()
        
        # Copy Neo's atlas to our world (keeping them separate)
        self.current_world['atlas'] = CognitiveAtlas.from_dict(
            self.neo.atlas.to_dict()
        )
        
        return stats
    
    def generate_landmark_image(self, landmark_id, use_diffusion=True):
        """Generate high-res image from a landmark"""
        if not self.current_world:
            return None
        
        atlas = self.current_world['atlas']
        if landmark_id not in atlas.nodes:
            return None
        
        if use_diffusion and hasattr(self, 'pipeline'):
            # Use Stable Diffusion pipeline for coherent images
            # Start with a noise image
            init_image = Image.new('RGB', (512, 512), color='gray')
            
            # Use the theme as prompt with some variation
            prompt = f"{self.current_world['theme']}, highly detailed, photorealistic"
            
            # Generate using img2img with high strength
            with torch.no_grad():
                image = self.pipeline(
                    prompt=prompt,
                    image=init_image,
                    strength=0.95,  # High strength for more variation
                    guidance_scale=7.5,
                    num_inference_steps=50
                ).images[0]
            
            # Convert PIL to numpy
            image_np = np.array(image)
            return image_np
        else:
            # Fallback to raw VAE decode (will be noisy)
            latent = atlas.nodes[landmark_id]['latent']
            latent_tensor = torch.tensor(latent, device=device, dtype=torch.float32)
            
            with torch.no_grad():
                latent_4d = latent_tensor.view(1, 4, 32, 32)
                image_tensor = self.vae.decode(latent_4d / 0.18215).sample
            
        # Convert to numpy
        image = image_tensor[0].cpu().permute(1, 2, 0).numpy()
        image = ((image * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        
        return image
    
    def create_storyboard(self, start_landmark, end_landmark, 
                         frames=90, method='spherical'):
        """Create storyboard between two landmarks"""
        if not self.current_world:
            return None
        
        atlas = self.current_world['atlas']
        
        # Find path through atlas
        path = atlas.dijkstra(start_landmark, end_landmark)
        if not path:
            return None
        
        # Generate frames for each segment
        full_storyboard = []
        frames_per_segment = max(frames // len(path), 10)
        
        for i in range(len(path) - 1):
            start_latent = atlas.nodes[path[i]]['latent']
            end_latent = atlas.nodes[path[i+1]]['latent']
            
            segment = self.storyboard_gen.generate_transition(
                start_latent, end_latent, 
                frames_per_segment, method
            )
            
            # Add landmark labels
            segment[0].label = atlas.nodes[path[i]].get('label', path[i])
            segment[-1].label = atlas.nodes[path[i+1]].get('label', path[i+1])
            
            full_storyboard.extend(segment)
        
        # Store in world
        storyboard_data = {
            'id': f"story_{len(self.current_world['storyboards'])}",
            'start': start_landmark,
            'end': end_landmark,
            'frames': full_storyboard,
            'path': path
        }
        
        self.current_world['storyboards'].append(storyboard_data)
        
        return storyboard_data
    
    def export_to_video(self, storyboard, output_path, fps=30, use_diffusion=True):
        """Export storyboard to video file with option for coherent images"""
        if not storyboard or not storyboard['frames']:
            return False
        
        # Generate images for all frames
        frames_with_images = []
        
        if use_diffusion and hasattr(self, 'pipeline'):
            # Use Stable Diffusion for coherent video
            prompt = f"{self.current_world['theme']}, cinematic, detailed"
            
            for i, frame in enumerate(storyboard['frames']):
                if frame.image is None:
                    # Create seed image from latent
                    latent_tensor = torch.tensor(
                        frame.latent_vector, 
                        device=device, 
                        dtype=torch.float32
                    ).view(1, 4, 32, 32)
                    
                    with torch.no_grad():
                        # First decode to get a base image
                        base_img_tensor = self.vae.decode(latent_tensor / 0.18215).sample
                        base_img = base_img_tensor[0].cpu().permute(1, 2, 0).numpy()
                        base_img = ((base_img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                        
                        # Convert to PIL for SD pipeline
                        pil_img = Image.fromarray(base_img).resize((512, 512))
                        
                        # Generate coherent image using SD
                        # Use lower strength to maintain continuity
                        coherent_img = self.pipeline(
                            prompt=prompt,
                            image=pil_img,
                            strength=0.7,  # Lower strength for smoother transitions
                            guidance_scale=7.5,
                            num_inference_steps=20  # Fewer steps for speed
                        ).images[0]
                        
                        # Convert back to numpy
                        frame.image = np.array(coherent_img.resize((512, 512)))
                
                frames_with_images.append(frame.image)
                
        else:
            # Fallback to raw VAE decode
            for frame in storyboard['frames']:
                if frame.image is None:
                    latent_tensor = torch.tensor(
                        frame.latent_vector, 
                        device=device, 
                        dtype=torch.float32
                    ).view(1, 4, 32, 32)
                    
                    with torch.no_grad():
                        img_tensor = self.vae.decode(latent_tensor / 0.18215).sample
                
                img = img_tensor[0].cpu().permute(1, 2, 0).numpy()
                img = ((img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                frame.image = img
            
            frames_with_images.append(frame.image)
        
        # Write video
        h, w = frames_with_images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for img in frames_with_images:
            # Convert RGB to BGR for OpenCV
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        
        writer.release()
        return True

class CognitiveAtlasStudioUI:
    """Interactive GUI for the Cognitive Atlas Studio"""
    
    def __init__(self, root, neo_system, vae_model):
        self.root = root
        self.root.title("Cognitive Atlas Studio - World Builder")
        self.root.geometry("1400x900")
        
        self.world_builder = WorldBuilder(neo_system, vae_model)
        self.setup_ui()
        
    def setup_ui(self):
        """Create the user interface"""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # === Control Panel ===
        control_frame = ttk.LabelFrame(main_frame, text="World Creation", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Theme input
        ttk.Label(control_frame, text="World Theme:").grid(row=0, column=0, sticky=tk.W)
        self.theme_var = tk.StringVar(value="Gothic Sci-Fi Alien Jungle")
        theme_entry = ttk.Entry(control_frame, textvariable=self.theme_var, width=40)
        theme_entry.grid(row=0, column=1, padx=5)
        
        # Exploration time
        ttk.Label(control_frame, text="Exploration Time (sec):").grid(row=0, column=2, padx=(20,5))
        self.explore_time_var = tk.IntVar(value=60)
        ttk.Spinbox(control_frame, from_=10, to=300, textvariable=self.explore_time_var, 
                    width=10).grid(row=0, column=3)
        
        # Buttons
        self.seed_btn = ttk.Button(control_frame, text="🌱 Seed World", 
                                   command=self.seed_world)
        self.seed_btn.grid(row=0, column=4, padx=10)
        
        self.explore_btn = ttk.Button(control_frame, text="🔍 Start Exploration", 
                                      command=self.start_exploration, state=tk.DISABLED)
        self.explore_btn.grid(row=0, column=5, padx=5)
        
        # === Atlas Visualization ===
        atlas_frame = ttk.LabelFrame(main_frame, text="Cognitive Atlas", padding="10")
        atlas_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        
        # Create matplotlib figure for network visualization
        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, atlas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # === Landmark Details ===
        detail_frame = ttk.LabelFrame(main_frame, text="Landmark Explorer", padding="10")
        detail_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        
        # Selected landmark info
        self.landmark_info = tk.Text(detail_frame, height=5, width=40, wrap=tk.WORD)
        self.landmark_info.pack(fill=tk.X, pady=5)
        
        # Landmark image display
        self.landmark_image_label = ttk.Label(detail_frame)
        self.landmark_image_label.pack(pady=5)
        
        # Generate image button
        self.gen_image_btn = ttk.Button(detail_frame, text="Generate Image", 
                                        command=self.generate_landmark_image,
                                        state=tk.DISABLED)
        self.gen_image_btn.pack(pady=5)
        
        # === Storyboard Panel ===
        story_frame = ttk.LabelFrame(main_frame, text="Storyboard Creator", padding="10")
        story_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Path selection
        ttk.Label(story_frame, text="From:").grid(row=0, column=0)
        self.from_landmark_var = tk.StringVar()
        self.from_combo = ttk.Combobox(story_frame, textvariable=self.from_landmark_var, 
                                       width=20, state="readonly")
        self.from_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(story_frame, text="To:").grid(row=0, column=2, padx=(20,5))
        self.to_landmark_var = tk.StringVar()
        self.to_combo = ttk.Combobox(story_frame, textvariable=self.to_landmark_var, 
                                     width=20, state="readonly")
        self.to_combo.grid(row=0, column=3, padx=5)
        
        # Interpolation settings
        ttk.Label(story_frame, text="Method:").grid(row=0, column=4, padx=(20,5))
        self.interp_var = tk.StringVar(value="spherical")
        interp_combo = ttk.Combobox(story_frame, textvariable=self.interp_var,
                                    values=["linear", "spherical", "bezier"],
                                    width=10, state="readonly")
        interp_combo.grid(row=0, column=5, padx=5)
        
        ttk.Label(story_frame, text="Frames:").grid(row=0, column=6, padx=(20,5))
        self.frames_var = tk.IntVar(value=90)
        ttk.Spinbox(story_frame, from_=30, to=300, textvariable=self.frames_var,
                   width=8).grid(row=0, column=7, padx=5)
        
        # Create storyboard button
        self.create_story_btn = ttk.Button(story_frame, text="🎬 Create Storyboard",
                                          command=self.create_storyboard,
                                          state=tk.DISABLED)
        self.create_story_btn.grid(row=0, column=8, padx=20)
        
        # Export button
        self.export_btn = ttk.Button(story_frame, text="💾 Export Video",
                                     command=self.export_video,
                                     state=tk.DISABLED)
        self.export_btn.grid(row=0, column=9, padx=5)
        
        # === Status Bar ===
        self.status_var = tk.StringVar(value="Ready to create worlds...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
    
    def seed_world(self):
        """Initialize world with theme"""
        theme = self.theme_var.get()
        if not theme:
            messagebox.showwarning("No Theme", "Please enter a world theme")
            return
        
        self.status_var.set(f"Seeding world: {theme}")
        self.explore_btn.config(state=tk.NORMAL)
        
    def start_exploration(self):
        """Start Neo's autonomous exploration"""
        def explore():
            self.progress.start()
            self.explore_btn.config(state=tk.DISABLED)
            self.status_var.set("Neo is exploring the conceptual space...")
            
            # Run exploration
            world = self.world_builder.seed_world(
                self.theme_var.get(),
                self.explore_time_var.get()
            )
            
            # Update UI with results
            self.root.after(0, self.update_atlas_view, world)
            
            self.progress.stop()
            self.status_var.set(f"Exploration complete. Discovered {len(world['atlas'].nodes)} landmarks")
            self.create_story_btn.config(state=tk.NORMAL)
            self.gen_image_btn.config(state=tk.NORMAL)
        
        # Run in thread
        threading.Thread(target=explore, daemon=True).start()
    
    def update_atlas_view(self, world):
        """Update the atlas visualization"""
        self.ax.clear()
        
        atlas = world['atlas']
        if not atlas.nodes:
            self.ax.text(0.5, 0.5, 'No landmarks discovered yet', 
                        ha='center', va='center')
            self.canvas.draw()
            return
        
        # Create networkx graph
        G = nx.Graph()
        for node_id, node_data in atlas.nodes.items():
            G.add_node(node_id, label=node_data.get('label', node_id))
        
        for source, targets in atlas.edges.items():
            for target, weight in targets.items():
                G.add_edge(source, target, weight=weight)
        
        # Draw network
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=self.ax, alpha=0.3)
        
        # Draw nodes
        node_colors = [atlas.nodes[n]['visits'] for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=self.ax, 
                              node_color=node_colors,
                              cmap='viridis',
                              node_size=300,
                              alpha=0.9)
        
        # Draw labels
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, ax=self.ax, font_size=8)
        
        self.ax.set_title(f"Cognitive Atlas: {world['theme']}")
        self.ax.axis('off')
        self.canvas.draw()
        
        # Update landmark combos
        landmark_ids = list(atlas.nodes.keys())
        self.from_combo['values'] = landmark_ids
        self.to_combo['values'] = landmark_ids
        
    def generate_landmark_image(self):
        """Generate and display image for selected landmark"""
        # This would be connected to landmark selection in the graph
        pass
    
    def create_storyboard(self):
        """Create storyboard between selected landmarks"""
        start = self.from_landmark_var.get()
        end = self.to_landmark_var.get()
        
        if not start or not end:
            messagebox.showwarning("Selection Required", 
                                  "Please select start and end landmarks")
            return
        
        self.progress.start()
        self.status_var.set(f"Creating storyboard from {start} to {end}...")
        
        def generate():
            storyboard = self.world_builder.create_storyboard(
                start, end,
                self.frames_var.get(),
                self.interp_var.get()
            )
            
            if storyboard:
                self.current_storyboard = storyboard
                self.export_btn.config(state=tk.NORMAL)
                self.status_var.set(f"Storyboard created: {len(storyboard['frames'])} frames")
            else:
                self.status_var.set("Failed to create storyboard")
            
            self.progress.stop()
        
        threading.Thread(target=generate, daemon=True).start()
    
    def export_video(self):
        """Export current storyboard to video file"""
        if not hasattr(self, 'current_storyboard'):
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")]
        )
        
        if filename:
            self.progress.start()
            self.status_var.set("Exporting video...")
            
            def export():
                success = self.world_builder.export_to_video(
                    self.current_storyboard, 
                    filename,
                    30  # fps
                )
                
                if success:
                    self.status_var.set(f"Video exported: {filename}")
                else:
                    self.status_var.set("Export failed")
                
                self.progress.stop()
            
            threading.Thread(target=export, daemon=True).start()

def main():
    """Launch the Cognitive Atlas Studio"""
    
    # Load models
    try:
        from diffusers import AutoencoderKL, StableDiffusionImg2ImgPipeline
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            subfolder="vae",
            torch_dtype=torch.float32  # Always use float32 for VAE
        ).to(device)
        vae.eval()
        
        # Load full Stable Diffusion pipeline for coherent image generation
        print("Loading Stable Diffusion pipeline for coherent images...")
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
    except Exception as e:
        print(f"Could not load models: {e}")
        return
    
    # Create Neo system with pipeline
    neo = TemporalPredictiveSweepingSystem(vae)
    
    # Create world builder with pipeline
    world_builder = WorldBuilder(neo, vae)
    world_builder.pipeline = pipeline  # Add SD pipeline for coherent images
    
    # Launch UI
    root = tk.Tk()
    app = CognitiveAtlasStudioUI(root, neo, vae)
    app.world_builder = world_builder  # Use our enhanced world builder
    
    print("=" * 60)
    print("COGNITIVE ATLAS STUDIO")
    print("Interactive World Builder & Storyboard Generator")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Enter a world theme (e.g., 'Gothic Sci-Fi Alien Jungle')")
    print("2. Click 'Seed World' then 'Start Exploration'")
    print("3. Watch Neo build its cognitive map")
    print("4. Select landmarks to generate images or create storyboards")
    print("5. Export storyboards as videos")
    print()
    
    root.mainloop()
    
if __name__ == "__main__":
    main()
