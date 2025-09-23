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
torch_dtype = torch.float16
print(f"Running on: {device} with dtype: {torch_dtype}")

# Memory management
if device == "cuda":
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.85)

# ============================================================================ #
#                  CORE STABILIZATION COMPONENTS (MATRIX 8 UPDATE)             #
# ============================================================================ #

class MoireField(nn.Module):
    """Generates moiré patterns for movement detection and phase-based gating"""
    def __init__(self, base_frequency=8.0, field_size=32):
        super().__init__()
        self.base_freq = base_frequency
        self.field_size = field_size
        x = torch.linspace(-1, 1, field_size, dtype=torch_dtype)
        y = torch.linspace(-1, 1, field_size, dtype=torch_dtype)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('pattern1', torch.sin(base_frequency * np.pi * xx))
        self.register_buffer('pattern2', torch.sin((base_frequency + 0.5) * np.pi * yy))

    def compute_phase_shift(self, current_frame, previous_frame):
        if previous_frame is None:
            return torch.zeros((1, 1, self.field_size, self.field_size), device=device, dtype=torch_dtype)
        with torch.amp.autocast('cuda'):
            current_small = F.interpolate(current_frame, size=(self.field_size, self.field_size), mode='bilinear')
            previous_small = F.interpolate(previous_frame, size=(self.field_size, self.field_size), mode='bilinear')
            current_gray = current_small.mean(dim=1, keepdim=True).to(torch_dtype)
            previous_gray = previous_small.mean(dim=1, keepdim=True).to(torch_dtype)
            diff = current_gray - previous_gray
            phase_x = diff * self.pattern1.unsqueeze(0).unsqueeze(0)
            phase_y = diff * self.pattern2.unsqueeze(0).unsqueeze(0)
            phase_shift = torch.sqrt(phase_x**2 + phase_y**2)
        return phase_shift.to(torch_dtype)

class HolographicSlowField(nn.Module):
    """Maintains a slowly-evolving latent field with controllable frequency gating."""
    def __init__(self, dimensions=(32, 32), channels=4):
        super().__init__()
        self.dimensions = dimensions
        self.channels = channels

        k_freq = [torch.fft.fftfreq(n, d=1/n, dtype=torch.float32) for n in dimensions]
        k_grid = torch.meshgrid(*k_freq, indexing='ij')
        k2_tensor = sum(k**2 for k in k_grid)
        self.register_buffer('k2', k2_tensor / k2_tensor.max())

    def evolve(self, field_state, steps=1, low_pass_damping=5.0, high_pass_gain=0.0):
        with torch.no_grad():  # Disable autocast for FFT stability
            field_state_f32 = field_state.float()
            field_fft = torch.fft.fft2(field_state_f32)
            
            low_pass_decay = torch.exp(-self.k2.unsqueeze(0).unsqueeze(0) * low_pass_damping)
            high_pass_decay = 1.0 - torch.exp(-(1.0 - self.k2.unsqueeze(0).unsqueeze(0)) * high_pass_gain * 20.0)
            final_decay = low_pass_decay * high_pass_decay

            for _ in range(steps):
                field_fft = field_fft * final_decay

            result = torch.fft.ifft2(field_fft).real
        return result.to(torch_dtype)

@dataclass
class StabilizationState:
    slow_field: Optional[torch.Tensor] = None
    latent_anchor: Optional[torch.Tensor] = None
    previous_frame: Optional[torch.Tensor] = None
    previous_latent: Optional[torch.Tensor] = None
    frame_count: int = 0

class StabilizedDiffusionPipeline:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1", resolution=384):
        print("Loading Stable Diffusion components...")
        self.resolution = resolution
        self.latent_size = resolution // 8

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, dtype=torch_dtype, safety_checker=None,
            requires_safety_checker=False, variant="fp16" if device == "cuda" else None
        ).to(device)

        if hasattr(self.pipe.unet, 'enable_attention_slicing'): self.pipe.unet.enable_attention_slicing(1)
        if hasattr(self.pipe.unet, 'enable_vae_slicing'): self.pipe.vae.enable_slicing()
        if hasattr(self.pipe.unet, 'enable_xformers_memory_efficient_attention'):
            try:
                self.pipe.unet.enable_xformers_memory_efficient_attention()
                print("Using xformers memory efficient attention")
            except: pass

        self.vae, self.text_encoder, self.tokenizer, self.unet, self.scheduler = \
            self.pipe.vae, self.pipe.text_encoder, self.pipe.tokenizer, self.pipe.unet, self.pipe.scheduler

        self.moire_field = MoireField(field_size=32).to(device)
        self.slow_field_model = HolographicSlowField(dimensions=(self.latent_size, self.latent_size), channels=4).to(device)
        self.state = StabilizationState()
        self.ema_alpha = 0.92
        self.anchor_strength = 0.6
        self.moire_sensitivity = 0.15

        if device == "cuda": torch.cuda.empty_cache()
        print(f"Pipeline ready! Using resolution: {resolution}x{resolution}")

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((self.resolution, self.resolution), Image.LANCZOS)
        image_array = np.array(image).astype(np.float32) / 127.5 - 1.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device, torch_dtype)
        with torch.amp.autocast('cuda'), torch.no_grad():
            latent = self.vae.encode(image_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        del image_tensor
        return latent.to(torch_dtype)

    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        with torch.amp.autocast('cuda'), torch.no_grad():
            latent = latent / self.vae.config.scaling_factor
            image = self.vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()[0]
        return Image.fromarray((image * 255).astype(np.uint8))

    def compute_movement_gate(self, current_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        phase_shift = self.moire_field.compute_phase_shift(current_tensor, self.state.previous_frame)
        movement_energy = phase_shift.mean(dim=[2, 3], keepdim=True).to(torch_dtype)
        gate = torch.exp(-movement_energy * self.moire_sensitivity).to(torch_dtype)
        return gate, phase_shift

    def update_slow_field(self, current_latent, gate, low_pass_damping, high_pass_gain):
        if self.state.slow_field is None:
            self.state.slow_field = current_latent.clone()
        else:
            blend_factor = (self.ema_alpha + (1 - self.ema_alpha) * gate).to(torch_dtype)
            self.state.slow_field = torch.lerp(
                current_latent.to(torch_dtype),
                self.state.slow_field.to(torch_dtype),
                blend_factor
            )
            self.state.slow_field = self.slow_field_model.evolve(
                self.state.slow_field, steps=2,
                low_pass_damping=low_pass_damping,
                high_pass_gain=high_pass_gain
            )
        return self.state.slow_field

    @torch.no_grad()
    def generate_stabilized(self, image, prompt, strength, guidance_scale, num_inference_steps, dream_mode, low_pass_damping, high_pass_gain):
        if device == "cuda": torch.cuda.empty_cache()
        
        image_array = np.array(image.convert("RGB").resize((self.resolution, self.resolution))).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device, torch_dtype)
        
        current_latent = self.encode_image(image)
        gate, phase_shift = self.compute_movement_gate(image_tensor)
        slow_field_current = self.update_slow_field(current_latent, gate, low_pass_damping, high_pass_gain)

        init_latent = slow_field_current if dream_mode else torch.lerp(current_latent, slow_field_current, self.anchor_strength)
        
        noise = torch.randn_like(init_latent) * (strength * 0.3)
        init_latent += noise

        self.pipe.scheduler.set_timesteps(num_inference_steps)
        text_inputs = self.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        with torch.amp.autocast('cuda'):
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(device))[0]

        latents = init_latent
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps-1)
        timesteps = self.scheduler.timesteps[-init_timestep:] if init_timestep > 0 else self.scheduler.timesteps
        
        for t in timesteps:
            with torch.amp.autocast('cuda'):
                noise_pred = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                anchor_pull = self.anchor_strength * 0.05 * (1 - float(t) / 1000)
                latents += anchor_pull * (slow_field_current - latents)

        output_image = self.decode_latent(latents)
        self.state.previous_frame, self.state.previous_latent = image_tensor, latents
        self.state.frame_count += 1

        debug_info = {'gate': gate.mean().item(), 'phase': phase_shift.mean().item(), 'field_std': slow_field_current.std().item()}
        del image_tensor, current_latent, text_embeddings, noise_pred
        return output_image, debug_info

# ============================================================================ #
#                               GUI APPLICATION                                #
# ============================================================================ #

class StabilizedVideoFilterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Matrix8: Frequency-Gated Diffusion Filter")
        self.root.geometry("1200x900")
        
        self.pipeline = None
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam Error", "Cannot open webcam")
            self.root.destroy()
            return
            
        self.processing, self.current_frame = False, None
        self.result_queue = queue.Queue()
        self.current_resolution = 384
        
        self.setup_gui()
        self.init_pipeline()
        
        threading.Thread(target=self.capture_loop, daemon=True).start()
        threading.Thread(target=self.process_loop, daemon=True).start()
        self.update_gui()
        
    def init_pipeline(self):
        if self.pipeline: del self.pipeline; gc.collect(); torch.cuda.empty_cache()
        self.pipeline = StabilizedDiffusionPipeline(resolution=self.current_resolution)
        self.status_label.config(text=f"Pipeline loaded at {self.current_resolution}x{self.current_resolution}")

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        r = 0
        ttk.Label(control_frame, text="Resolution:").grid(row=r, column=0, sticky=tk.W, padx=5)
        self.resolution_var = tk.IntVar(value=self.current_resolution)
        resolution_menu = ttk.Combobox(control_frame, textvariable=self.resolution_var, values=[256, 384, 512], state="readonly", width=10)
        resolution_menu.grid(row=r, column=1, sticky=tk.W, padx=5)
        resolution_menu.bind("<<ComboboxSelected>>", self.on_resolution_change)

        r += 1
        ttk.Label(control_frame, text="Prompt:").grid(row=r, column=0, sticky=tk.W, padx=5)
        self.prompt_var = tk.StringVar(value="cinematic portrait, dramatic lighting")
        ttk.Entry(control_frame, textvariable=self.prompt_var, width=40).grid(row=r, column=1, columnspan=3, sticky=(tk.W, tk.E), padx=5)

        def add_slider(row, text, var, v_from, v_to, v_res, v_len, label_widget, update_func):
            ttk.Label(control_frame, text=text).grid(row=row, column=0, sticky=tk.W, padx=5)
            s = Scale(control_frame, from_=v_from, to=v_to, resolution=v_res, variable=var, orient=HORIZONTAL, length=v_len)
            s.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
            label_widget.grid(row=row, column=2, padx=5)
            s.configure(command=update_func)
            update_func(var.get())

        r += 1
        self.strength_var = tk.DoubleVar(value=0.6); self.strength_label = ttk.Label(control_frame)
        add_slider(r, "Strength:", self.strength_var, 0.001, 1.0, 0.05, 200, self.strength_label, lambda v: self.strength_label.config(text=f"{float(v):.2f}"))

        r += 1
        self.anchor_var = tk.DoubleVar(value=0.6); self.anchor_label = ttk.Label(control_frame)
        add_slider(r, "Anchor:", self.anchor_var, 0.0, 1.0, 0.05, 200, self.anchor_label, self.update_anchor_label)

        r += 1
        self.ema_var = tk.DoubleVar(value=0.92); self.ema_label = ttk.Label(control_frame)
        add_slider(r, "Smoothing:", self.ema_var, 0.8, 0.99, 0.01, 200, self.ema_label, self.update_ema_label)

        r += 1
        self.gist_var = tk.DoubleVar(value=5.0); self.gist_label = ttk.Label(control_frame)
        add_slider(r, "Gist (Low-Pass):", self.gist_var, 0.0, 20.0, 0.1, 200, self.gist_label, lambda v: self.gist_label.config(text=f"{float(v):.1f}"))

        r += 1
        self.detail_var = tk.DoubleVar(value=0.0); self.detail_label = ttk.Label(control_frame)
        add_slider(r, "Detail (High-Pass):", self.detail_var, 0.0, 1.0, 0.01, 200, self.detail_label, lambda v: self.detail_label.config(text=f"{float(v):.2f}"))

        r += 1
        self.steps_var = tk.IntVar(value=10); self.steps_label = ttk.Label(control_frame)
        add_slider(r, "Steps:", self.steps_var, 5, 20, 1, 200, self.steps_label, lambda v: self.steps_label.config(text=str(int(float(v)))))

        r += 1
        self.dream_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Dream Mode (Ignore Webcam)", variable=self.dream_mode_var).grid(row=r, column=1, padx=5, sticky='w')

        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=4, rowspan=r+1, padx=20)
        self.process_btn = ttk.Button(button_frame, text="Start Processing", command=self.toggle_processing)
        self.process_btn.pack(pady=3)
        ttk.Button(button_frame, text="Reset State", command=self.reset_state).pack(pady=3)
        ttk.Button(button_frame, text="Clear VRAM", command=self.clear_cache).pack(pady=3)

        video_frame = ttk.Frame(main_frame)
        video_frame.grid(row=1, column=0, columnspan=2, pady=10)
        self.webcam_label = ttk.Label(ttk.LabelFrame(video_frame, text="Webcam Input", padding="5")); self.webcam_label.pack(); self.webcam_label.master.grid(row=0,column=0,padx=10)
        self.output_label = ttk.Label(ttk.LabelFrame(video_frame, text="Stabilized Output", padding="5")); self.output_label.pack(); self.output_label.master.grid(row=0,column=1,padx=10)
        
        debug_frame = ttk.LabelFrame(main_frame, text="Stabilization Metrics", padding="10")
        debug_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.gate_label = ttk.Label(debug_frame, text="Gate: --"); self.gate_label.grid(row=0, column=0, padx=10)
        self.phase_label = ttk.Label(debug_frame, text="Phase: --"); self.phase_label.grid(row=0, column=1, padx=10)
        self.field_label = ttk.Label(debug_frame, text="Field σ: --"); self.field_label.grid(row=0, column=2, padx=10)
        self.fps_label = ttk.Label(debug_frame, text="FPS: --"); self.fps_label.grid(row=0, column=3, padx=10)
        self.vram_label = ttk.Label(debug_frame, text="VRAM: --"); self.vram_label.grid(row=0, column=4, padx=10)

        self.status_label = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

    def on_resolution_change(self, event=None):
        new_res = self.resolution_var.get()
        if new_res != self.current_resolution:
            self.current_resolution, self.processing = new_res, False
            self.process_btn.config(text="Start Processing")
            self.status_label.config(text=f"Reloading pipeline at {new_res}x{new_res}...")
            self.root.after(100, self.init_pipeline)
            
    def update_anchor_label(self, value):
        self.anchor_label.config(text=f"{float(value):.2f}")
        if self.pipeline: self.pipeline.anchor_strength = float(value)
        
    def update_ema_label(self, value):
        self.ema_label.config(text=f"{float(value):.2f}")
        if self.pipeline: self.pipeline.ema_alpha = float(value)

    def toggle_processing(self):
        self.processing = not self.processing
        self.process_btn.config(text="Stop Processing" if self.processing else "Start Processing")
        self.status_label.config(text="Processing..." if self.processing else "Stopped")
            
    def reset_state(self):
        if self.pipeline: self.pipeline.state = StabilizationState()
        self.status_label.config(text="State reset")
    
    def clear_cache(self):
        if device == "cuda": gc.collect(); torch.cuda.empty_cache()
        self.status_label.config(text="VRAM cleared")
        
    def capture_loop(self):
        while True:
            ret, frame = self.cap.read()
            if ret: self.current_frame = frame
            time.sleep(0.03)
            
    def process_loop(self):
        last_time = time.time()
        while True:
            if self.processing and self.current_frame is not None and self.pipeline:
                try:
                    pil_image = Image.fromarray(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB))
                    result_image, debug_info = self.pipeline.generate_stabilized(
                        pil_image, self.prompt_var.get(), strength=self.strength_var.get(),
                        guidance_scale=7.5, num_inference_steps=self.steps_var.get(),
                        dream_mode=self.dream_mode_var.get(),
                        low_pass_damping=self.gist_var.get(),
                        high_pass_gain=self.detail_var.get()
                    )
                    
                    current_time = time.time()
                    debug_info['fps'] = 1.0 / (current_time - last_time)
                    last_time = current_time
                    if device == "cuda":
                        debug_info['vram'] = (torch.cuda.memory_allocated()/1e9, torch.cuda.memory_reserved()/1e9)
                    self.result_queue.put((result_image, debug_info))
                except Exception as e:
                    print(f"Processing error: {e}")
                    self.status_label.config(text=f"Error: {str(e)[:50]}")
                    if "out of memory" in str(e).lower(): self.clear_cache()
            else:
                time.sleep(0.1)
                
    def update_gui(self):
        if self.current_frame is not None:
            frame_pil = Image.fromarray(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)).resize((400, 300), Image.LANCZOS)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            self.webcam_label.config(image=frame_tk); self.webcam_label.image = frame_tk
            
        try:
            while not self.result_queue.empty():
                result_image, debug_info = self.result_queue.get_nowait()
                result_tk = ImageTk.PhotoImage(result_image.resize((400, 300), Image.LANCZOS))
                self.output_label.config(image=result_tk); self.output_label.image = result_tk
                self.gate_label.config(text=f"Gate: {debug_info['gate']:.3f}")
                self.phase_label.config(text=f"Phase: {debug_info['phase']:.3f}")
                self.field_label.config(text=f"Field σ: {debug_info['field_std']:.3f}")
                self.fps_label.config(text=f"FPS: {debug_info['fps']:.1f}")
                if 'vram' in debug_info:
                    self.vram_label.config(text=f"VRAM: {debug_info['vram'][0]:.1f}/{debug_info['vram'][1]:.1f} GB")
        except queue.Empty: pass
        self.root.after(30, self.update_gui)
        
    def cleanup(self):
        self.processing = False
        if self.cap.isOpened(): self.cap.release()
        if self.pipeline: del self.pipeline
        if device == "cuda": torch.cuda.empty_cache()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = StabilizedVideoFilterGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()