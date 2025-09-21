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

import gc
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, Scale, HORIZONTAL, messagebox
import threading
import time
import queue
from dataclasses import dataclass
from typing import Optional, Tuple

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16  # Always use fp16 for memory
print(f"Running on: {device} with dtype: {torch_dtype}")

# Memory management
if device == "cuda":
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    # Limit memory growth
    torch.cuda.set_per_process_memory_fraction(0.85)

# ============================================================================ #
#                          CORE STABILIZATION COMPONENTS                       #
# ============================================================================ #

class MoireField(nn.Module):
    """Generates moiré patterns for movement detection and phase-based gating"""
    def __init__(self, base_frequency=8.0, field_size=32):  # Reduced from 64
        super().__init__()
        self.base_freq = base_frequency
        self.field_size = field_size
        
        # Create reference moiré pattern (in fp16 for memory)
        x = torch.linspace(-1, 1, field_size, dtype=torch_dtype)
        y = torch.linspace(-1, 1, field_size, dtype=torch_dtype)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Two slightly different frequencies create moiré
        self.register_buffer('pattern1', torch.sin(base_frequency * np.pi * xx))
        self.register_buffer('pattern2', torch.sin((base_frequency + 0.5) * np.pi * yy))
        
    def compute_phase_shift(self, current_frame, previous_frame):
        """Compute phase shift between frames using moiré interference"""
        if previous_frame is None:
            return torch.zeros((1, 1, self.field_size, self.field_size), device=device, dtype=torch_dtype)
        
        with torch.cuda.amp.autocast():
            # Downsample frames to field size
            current_small = F.interpolate(current_frame, size=(self.field_size, self.field_size), mode='bilinear')
            previous_small = F.interpolate(previous_frame, size=(self.field_size, self.field_size), mode='bilinear')
            
            # Convert to grayscale for phase detection
            current_gray = current_small.mean(dim=1, keepdim=True)
            previous_gray = previous_small.mean(dim=1, keepdim=True)
            
            # Compute local phase shifts using moiré interference
            diff = current_gray - previous_gray
            
            # Modulate with moiré patterns
            phase_x = diff * self.pattern1.unsqueeze(0).unsqueeze(0)
            phase_y = diff * self.pattern2.unsqueeze(0).unsqueeze(0)
            
            # Combined phase shift (magnitude indicates movement)
            phase_shift = torch.sqrt(phase_x**2 + phase_y**2)
            
        return phase_shift

class HolographicSlowField(nn.Module):
    """Maintains a slowly-evolving latent field with wave dynamics"""
    def __init__(self, dimensions=(32, 32), channels=4):  # Reduced from 64x64
        super().__init__()
        self.dimensions = dimensions
        self.channels = channels
        
        # Learnable damping for each channel (in fp16)
        self.damping = nn.Parameter(torch.full((1, channels, 1, 1), 0.02, dtype=torch_dtype))
        
        # Frequency domain operations for smooth evolution
        k_freq = [torch.fft.fftfreq(n, d=1/n, dtype=torch.float32) for n in dimensions]  # FFT needs float32
        k_grid = torch.meshgrid(*k_freq, indexing='ij')
        self.register_buffer('k2', sum(k**2 for k in k_grid))
        
    def evolve(self, field_state, steps=1):
        """Evolve field using spectral methods for stability"""
        with torch.cuda.amp.autocast(enabled=False):  # FFT needs float32
            field_state_f32 = field_state.float()
            field_fft = torch.fft.fft2(field_state_f32)
            
            # Frequency-domain damping (preserves low frequencies)
            decay = torch.exp(-self.k2.unsqueeze(0).unsqueeze(0) * F.softplus(self.damping.float()))
            
            for _ in range(steps):
                field_fft = field_fft * decay
                
            result = torch.fft.ifft2(field_fft).real
            
        return result.to(torch_dtype)

@dataclass
class StabilizationState:
    """Maintains the state for temporal stabilization"""
    slow_field: Optional[torch.Tensor] = None
    latent_anchor: Optional[torch.Tensor] = None
    previous_frame: Optional[torch.Tensor] = None
    previous_latent: Optional[torch.Tensor] = None
    frame_count: int = 0

class StabilizedDiffusionPipeline:
    """Main stabilization pipeline combining moiré fields and slow fields"""
    
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1", resolution=384):
        print("Loading Stable Diffusion components...")
        
        self.resolution = resolution
        self.latent_size = resolution // 8  # VAE downscales by 8x
        
        # Load pipeline components with memory optimizations
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
            variant="fp16" if device == "cuda" else None,
        ).to(device)
        
        # Enable memory efficient attention
        if hasattr(self.pipe.unet, 'enable_attention_slicing'):
            self.pipe.unet.enable_attention_slicing(1)
        if hasattr(self.pipe.unet, 'enable_vae_slicing'):
            self.pipe.vae.enable_slicing()
        if hasattr(self.pipe.unet, 'enable_xformers_memory_efficient_attention'):
            try:
                self.pipe.unet.enable_xformers_memory_efficient_attention()
                print("Using xformers memory efficient attention")
            except:
                pass
        
        # Direct access to VAE for latent manipulation
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler
        
        # Stabilization components (reduced size)
        self.moire_field = MoireField(base_frequency=8.0, field_size=32).to(device)
        self.slow_field = HolographicSlowField(
            dimensions=(self.latent_size, self.latent_size), 
            channels=4
        ).to(device)
        
        # State tracking
        self.state = StabilizationState()
        
        # Hyperparameters
        self.ema_alpha = 0.92  # Slow field update rate
        self.anchor_strength = 0.6  # How much to use slow field
        self.moire_sensitivity = 0.15  # Movement detection sensitivity
        
        # Clear cache after loading
        if device == "cuda":
            torch.cuda.empty_cache()
            
        print(f"Pipeline ready! Using resolution: {resolution}x{resolution}")
        
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode PIL image to VAE latent"""
        # Preprocess
        image = image.convert("RGB").resize((self.resolution, self.resolution), Image.LANCZOS)
        image_array = np.array(image).astype(np.float32) / 127.5 - 1.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device, torch_dtype)
        
        # Encode with autocast
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                latent = self.vae.encode(image_tensor).latent_dist.sample()
                latent = latent * self.vae.config.scaling_factor
                
        # Clear intermediate tensors
        del image_tensor
        if device == "cuda":
            torch.cuda.empty_cache()
            
        return latent
    
    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """Decode VAE latent to PIL image"""
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                latent = latent / self.vae.config.scaling_factor
                image = self.vae.decode(latent).sample
                
        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        image = (image * 255).astype(np.uint8)
        
        return Image.fromarray(image)
    
    def compute_movement_gate(self, current_tensor: torch.Tensor) -> torch.Tensor:
        """Compute gating signal from moiré phase shifts"""
        phase_shift = self.moire_field.compute_phase_shift(
            current_tensor, 
            self.state.previous_frame
        )
        
        # Convert phase shift to gating signal (less movement = stronger gate)
        movement_energy = phase_shift.mean(dim=[2, 3], keepdim=True)
        gate = torch.exp(-movement_energy * self.moire_sensitivity)
        
        return gate, phase_shift
    
    def update_slow_field(self, current_latent: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Update slow field with gated blending and wave evolution"""
        if self.state.slow_field is None:
            self.state.slow_field = current_latent.clone()
        else:
            # Gated update: less movement = more influence from slow field
            self.state.slow_field = (
                self.state.slow_field * (self.ema_alpha + (1 - self.ema_alpha) * gate) +
                current_latent * (1 - self.ema_alpha) * (1 - gate)
            )
            
            # Evolve using wave dynamics for smoothness
            self.state.slow_field = self.slow_field.evolve(self.state.slow_field, steps=2)
        
        return self.state.slow_field
    
    @torch.no_grad()
    def generate_stabilized(
        self, 
        image: Image.Image, 
        prompt: str, 
        strength: float = 0.6,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 10,  # Reduced from 20
        dream_mode: bool = False
    ) -> Tuple[Image.Image, dict]:
        """Main generation with stabilization"""
        
        # Clear cache before generation
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Prepare image tensor
        image_array = np.array(image.convert("RGB").resize((self.resolution, self.resolution))).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device, torch_dtype)
        
        # Encode current frame
        current_latent = self.encode_image(image)
        
        # Compute movement-based gating
        gate, phase_shift = self.compute_movement_gate(image_tensor)
        
        # Update slow field
        slow_field_current = self.update_slow_field(current_latent, gate)
        
        # Create initial latent based on dream mode
        if dream_mode:
            init_latent = slow_field_current
        else:
            # Blend current encoding with slow field based on anchor_strength (now controllable)
            init_latent = (
                current_latent * (1 - self.anchor_strength) +
                slow_field_current * self.anchor_strength
            )
        
        # Add controlled noise for diffusion
        noise_level = strength * 0.3  # Less noise = more stability
        noise = torch.randn_like(init_latent) * noise_level
        init_latent = init_latent + noise
        
        # Run img2img with custom latent
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        
        # Encode prompt (with truncation for memory)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,  # Limit token length
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.cuda.amp.autocast():
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(device))[0]
        
        # Prepare latents with proper noise scheduling
        latents = init_latent
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps-1)
        timesteps = self.scheduler.timesteps[-init_timestep:] if init_timestep > 0 else self.scheduler.timesteps
        
        # Denoising loop with anchor constraint
        for i, t in enumerate(timesteps):
            with torch.cuda.amp.autocast():
                # Predict noise
                latent_model_input = latents
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                # Compute denoised latent
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                
                # Soft constraint: pull slightly toward slow field anchor (now scaled by anchor_strength)
                anchor_pull = self.anchor_strength * 0.05 * (1 - float(t) / 1000)  # Stronger pull as we denoise
                latents = latents + anchor_pull * (slow_field_current - latents)
        
        # Decode final latent
        output_image = self.decode_latent(latents)
        
        # Update state for next frame
        self.state.previous_frame = image_tensor
        self.state.previous_latent = latents
        self.state.frame_count += 1
        
        # Prepare debug info
        debug_info = {
            'gate_strength': gate.mean().item(),
            'phase_shift_mag': phase_shift.mean().item(),
            'slow_field_std': slow_field_current.std().item(),
            'frame_count': self.state.frame_count
        }
        
        # Clean up
        del image_tensor, current_latent, text_embeddings, noise_pred
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return output_image, debug_info

# ============================================================================ #
#                                   GUI APPLICATION                            #
# ============================================================================ #

class StabilizedVideoFilterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Moiré-Stabilized Diffusion Video Filter")
        self.root.geometry("1200x800")
        
        # Resolution options
        self.resolution_options = [256, 384, 512]
        self.current_resolution = 384  # Default
        
        # Initialize pipeline (will be created after resolution selection)
        self.pipeline = None
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open webcam")
            self.root.destroy()
            return
            
        # Processing state
        self.processing = False
        self.current_frame = None
        self.result_queue = queue.Queue()
        
        # Setup GUI
        self.setup_gui()
        
        # Initialize pipeline with selected resolution
        self.init_pipeline()
        
        # Start video capture thread
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_loop, daemon=True)
        self.process_thread.start()
        
        # Start GUI update loop
        self.update_gui()
        
    def init_pipeline(self):
        """Initialize or reinitialize pipeline with current resolution"""
        if self.pipeline is not None:
            # Clean up old pipeline
            del self.pipeline
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
        
        self.pipeline = StabilizedDiffusionPipeline(resolution=self.current_resolution)
        self.status_label.config(text=f"Pipeline loaded at {self.current_resolution}x{self.current_resolution}")
        
    def setup_gui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Resolution selector
        ttk.Label(control_frame, text="Resolution:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.resolution_var = tk.IntVar(value=384)
        resolution_menu = ttk.Combobox(control_frame, textvariable=self.resolution_var, 
                                       values=self.resolution_options, state="readonly", width=10)
        resolution_menu.grid(row=0, column=1, sticky=tk.W, padx=5)
        resolution_menu.bind("<<ComboboxSelected>>", self.on_resolution_change)
        
        # Prompt entry
        ttk.Label(control_frame, text="Prompt:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.prompt_var = tk.StringVar(value="cinematic portrait, dramatic lighting")
        prompt_entry = ttk.Entry(control_frame, textvariable=self.prompt_var, width=40)
        prompt_entry.grid(row=1, column=1, columnspan=3, sticky=(tk.W, tk.E), padx=5)
        
        # Strength slider
        ttk.Label(control_frame, text="Strength:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.strength_var = tk.DoubleVar(value=0.6)
        strength_scale = Scale(control_frame, from_=0.001, to=1.0, resolution=0.05,
                               variable=self.strength_var, orient=HORIZONTAL, length=200)
        strength_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)
        self.strength_label = ttk.Label(control_frame, text="0.60")
        self.strength_label.grid(row=2, column=2, padx=5)
        strength_scale.configure(command=self.update_strength_label)
        
        # Anchor strength slider
        ttk.Label(control_frame, text="Anchor:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.anchor_var = tk.DoubleVar(value=0.6)
        anchor_scale = Scale(control_frame, from_=0.0, to=1.0, resolution=0.05,
                            variable=self.anchor_var, orient=HORIZONTAL, length=200)
        anchor_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5)
        self.anchor_label = ttk.Label(control_frame, text="0.60")
        self.anchor_label.grid(row=3, column=2, padx=5)
        anchor_scale.configure(command=self.update_anchor_label)
        
        # EMA alpha slider
        ttk.Label(control_frame, text="Smoothing:").grid(row=4, column=0, sticky=tk.W, padx=5)
        self.ema_var = tk.DoubleVar(value=0.92)
        ema_scale = Scale(control_frame, from_=0.8, to=0.99, resolution=0.01,
                         variable=self.ema_var, orient=HORIZONTAL, length=200)
        ema_scale.grid(row=4, column=1, sticky=(tk.W, tk.E), padx=5)
        self.ema_label = ttk.Label(control_frame, text="0.92")
        self.ema_label.grid(row=4, column=2, padx=5)
        ema_scale.configure(command=self.update_ema_label)
        
        # Inference steps
        ttk.Label(control_frame, text="Steps:").grid(row=5, column=0, sticky=tk.W, padx=5)
        self.steps_var = tk.IntVar(value=10)
        steps_scale = Scale(control_frame, from_=5, to=20, resolution=1,
                           variable=self.steps_var, orient=HORIZONTAL, length=200)
        steps_scale.grid(row=5, column=1, sticky=(tk.W, tk.E), padx=5)
        self.steps_label = ttk.Label(control_frame, text="10")
        self.steps_label.grid(row=5, column=2, padx=5)
        steps_scale.configure(command=self.update_steps_label)
        
        # Dream mode checkbox
        self.dream_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Dream Mode (Ignore Webcam)", variable=self.dream_mode_var).grid(row=6, column=1, padx=5, sticky='w')
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=4, rowspan=7, padx=20)
        
        self.process_btn = ttk.Button(button_frame, text="Start Processing",
                                     command=self.toggle_processing)
        self.process_btn.pack(pady=3)
        
        self.reset_btn = ttk.Button(button_frame, text="Reset State",
                                   command=self.reset_state)
        self.reset_btn.pack(pady=3)
        
        self.clear_cache_btn = ttk.Button(button_frame, text="Clear VRAM",
                                         command=self.clear_cache)
        self.clear_cache_btn.pack(pady=3)
        
        self.save_state_btn = ttk.Button(button_frame, text="Save State",
                                        command=self.save_state)
        self.save_state_btn.pack(pady=3)
        
        self.load_state_btn = ttk.Button(button_frame, text="Load State",
                                        command=self.load_state)
        self.load_state_btn.pack(pady=3)
        
        # Video displays
        video_frame = ttk.Frame(main_frame)
        video_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Webcam feed
        webcam_frame = ttk.LabelFrame(video_frame, text="Webcam Input", padding="5")
        webcam_frame.grid(row=0, column=0, padx=10)
        self.webcam_label = ttk.Label(webcam_frame)
        self.webcam_label.pack()
        
        # Generated output
        output_frame = ttk.LabelFrame(video_frame, text="Stabilized Output", padding="5")
        output_frame.grid(row=0, column=1, padx=10)
        self.output_label = ttk.Label(output_frame)
        self.output_label.pack()
        
        # Debug information
        debug_frame = ttk.LabelFrame(main_frame, text="Stabilization Metrics", padding="10")
        debug_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.gate_label = ttk.Label(debug_frame, text="Gate: --")
        self.gate_label.grid(row=0, column=0, padx=10)
        
        self.phase_label = ttk.Label(debug_frame, text="Phase: --")
        self.phase_label.grid(row=0, column=1, padx=10)
        
        self.field_label = ttk.Label(debug_frame, text="Field σ: --")
        self.field_label.grid(row=0, column=2, padx=10)
        
        self.fps_label = ttk.Label(debug_frame, text="FPS: --")
        self.fps_label.grid(row=0, column=3, padx=10)
        
        # VRAM usage
        self.vram_label = ttk.Label(debug_frame, text="VRAM: --")
        self.vram_label.grid(row=0, column=4, padx=10)
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
    def on_resolution_change(self, event=None):
        """Handle resolution change"""
        new_res = self.resolution_var.get()
        if new_res != self.current_resolution:
            self.current_resolution = new_res
            self.processing = False
            self.process_btn.config(text="Start Processing")
            self.status_label.config(text=f"Reloading pipeline at {new_res}x{new_res}...")
            self.root.after(100, self.init_pipeline)
            
    def update_strength_label(self, value):
        self.strength_label.config(text=f"{float(value):.2f}")
        
    def update_anchor_label(self, value):
        self.anchor_label.config(text=f"{float(value):.2f}")
        if self.pipeline:
            self.pipeline.anchor_strength = float(value)
        
    def update_ema_label(self, value):
        self.ema_label.config(text=f"{float(value):.2f}")
        if self.pipeline:
            self.pipeline.ema_alpha = float(value)
            
    def update_steps_label(self, value):
        self.steps_label.config(text=str(int(value)))
        
    def toggle_processing(self):
        self.processing = not self.processing
        if self.processing:
            self.process_btn.config(text="Stop Processing")
            self.status_label.config(text="Processing...")
        else:
            self.process_btn.config(text="Start Processing")
            self.status_label.config(text="Stopped")
            
    def reset_state(self):
        if self.pipeline:
            self.pipeline.state = StabilizationState()
        self.status_label.config(text="State reset")
    
    def save_state(self):
        """Save the slow field and pipeline state"""
        if self.pipeline and self.pipeline.state.slow_field is not None:
            import pickle
            from tkinter import filedialog
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if filename:
                state_dict = {
                    'slow_field': self.pipeline.state.slow_field.cpu().numpy(),
                    'frame_count': self.pipeline.state.frame_count,
                    'resolution': self.current_resolution,
                    'prompt': self.prompt_var.get()
                }
                with open(filename, 'wb') as f:
                    pickle.dump(state_dict, f)
                self.status_label.config(text=f"State saved to {filename.split('/')[-1]}")
        else:
            self.status_label.config(text="No state to save yet")
    
    def load_state(self):
        """Load a saved slow field state"""
        if self.pipeline:
            import pickle
            from tkinter import filedialog
            
            filename = filedialog.askopenfilename(
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'rb') as f:
                    state_dict = pickle.load(f)
                
                # Check resolution compatibility
                if state_dict['resolution'] != self.current_resolution:
                    response = tk.messagebox.askyesno(
                        "Resolution Mismatch",
                        f"Saved state is for {state_dict['resolution']}x{state_dict['resolution']}, "
                        f"current is {self.current_resolution}x{self.current_resolution}. "
                        f"Switch to saved resolution?"
                    )
                    if response:
                        self.resolution_var.set(state_dict['resolution'])
                        self.on_resolution_change()
                        # Schedule the actual load after pipeline reinit
                        self.root.after(1000, lambda: self._complete_load(state_dict))
                        return
                    else:
                        self.status_label.config(text="Load cancelled - resolution mismatch")
                        return
                
                self._complete_load(state_dict)
    
    def _complete_load(self, state_dict):
        """Complete the state loading after pipeline is ready"""
        self.pipeline.state.slow_field = torch.from_numpy(
            state_dict['slow_field']
        ).to(device, torch_dtype)
        self.pipeline.state.frame_count = state_dict['frame_count']
        self.prompt_var.set(state_dict.get('prompt', ''))
        self.status_label.config(text=f"State loaded (frame {state_dict['frame_count']})")
        
    def clear_cache(self):
        """Clear CUDA cache"""
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            self.status_label.config(text="VRAM cleared")
        
    def capture_loop(self):
        """Continuous webcam capture"""
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
            time.sleep(0.03)  # ~30 FPS
            
    def process_loop(self):
        """Processing thread for generation"""
        last_time = time.time()
        
        while True:
            if self.processing and self.current_frame is not None and self.pipeline is not None:
                try:
                    # Convert frame to PIL
                    frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Generate with stabilization
                    result_image, debug_info = self.pipeline.generate_stabilized(
                        pil_image,
                        self.prompt_var.get(),
                        strength=self.strength_var.get(),
                        num_inference_steps=self.steps_var.get(),
                        dream_mode=self.dream_mode_var.get()
                    )
                    
                    # Calculate FPS
                    current_time = time.time()
                    fps = 1.0 / (current_time - last_time)
                    last_time = current_time
                    debug_info['fps'] = fps
                    
                    # Get VRAM usage
                    if device == "cuda":
                        vram_used = torch.cuda.memory_allocated() / 1024**3
                        vram_reserved = torch.cuda.memory_reserved() / 1024**3
                        debug_info['vram'] = (vram_used, vram_reserved)
                    
                    # Queue result
                    self.result_queue.put((result_image, debug_info))
                    
                except Exception as e:
                    print(f"Processing error: {e}")
                    self.status_label.config(text=f"Error: {str(e)[:50]}")
                    if "out of memory" in str(e).lower():
                        self.clear_cache()
            else:
                time.sleep(0.1)
                
    def update_gui(self):
        """Update GUI with latest frames"""
        # Update webcam display
        if self.current_frame is not None:
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize((400, 300), Image.LANCZOS)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            self.webcam_label.config(image=frame_tk)
            self.webcam_label.image = frame_tk
            
        # Update generated display
        try:
            while not self.result_queue.empty():
                result_image, debug_info = self.result_queue.get_nowait()
                
                # Display result
                result_image = result_image.resize((400, 300), Image.LANCZOS)
                result_tk = ImageTk.PhotoImage(result_image)
                self.output_label.config(image=result_tk)
                self.output_label.image = result_tk
                
                # Update debug info
                self.gate_label.config(text=f"Gate: {debug_info['gate_strength']:.3f}")
                self.phase_label.config(text=f"Phase: {debug_info['phase_shift_mag']:.3f}")
                self.field_label.config(text=f"Field σ: {debug_info['slow_field_std']:.3f}")
                self.fps_label.config(text=f"FPS: {debug_info['fps']:.1f}")
                
                # Update VRAM usage if available
                if 'vram' in debug_info:
                    vram_used, vram_reserved = debug_info['vram']
                    self.vram_label.config(text=f"VRAM: {vram_used:.1f}/{vram_reserved:.1f} GB")
                
        except queue.Empty:
            pass
            
        # Schedule next update
        self.root.after(30, self.update_gui)
        
    def cleanup(self):
        """Cleanup on close"""
        self.processing = False
        if self.cap.isOpened():
            self.cap.release()
        if self.pipeline is not None:
            del self.pipeline
        if device == "cuda":
            torch.cuda.empty_cache()
        self.root.quit()

# ============================================================================ #
#                                      MAIN                                    #
# ============================================================================ #

if __name__ == "__main__":
    root = tk.Tk()
    app = StabilizedVideoFilterGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()