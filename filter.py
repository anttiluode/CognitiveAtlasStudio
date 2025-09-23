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
from diffusers import AutoencoderKL
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

# ============================================================================ #
#                  CORE STABILIZATION COMPONENTS (MATRIX 9)                    #
# ============================================================================ #

class MoireField(nn.Module):
    def __init__(self, base_frequency=8.0, field_size=32):
        super().__init__()
        x = torch.linspace(-1, 1, field_size, dtype=torch_dtype, device=device)
        y = torch.linspace(-1, 1, field_size, dtype=torch_dtype, device=device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('pattern1', torch.sin(base_frequency * np.pi * xx))
        self.register_buffer('pattern2', torch.sin((base_frequency + 0.5) * np.pi * yy))

    def compute_phase_shift(self, current_frame, previous_frame):
        if previous_frame is None: return torch.zeros((1, 1, 32, 32), device=device, dtype=torch_dtype)
        with torch.cuda.amp.autocast():
            current_small = F.interpolate(current_frame, size=(32, 32), mode='bilinear')
            previous_small = F.interpolate(previous_frame, size=(32, 32), mode='bilinear')
            diff = current_small.mean(dim=1, keepdim=True) - previous_small.mean(dim=1, keepdim=True)
            phase_shift = torch.sqrt((diff * self.pattern1)**2 + (diff * self.pattern2)**2)
        return phase_shift

class HolographicSlowField(nn.Module):
    def __init__(self, dimensions=(64, 64), channels=4):
        super().__init__()
        k_freq = [torch.fft.fftfreq(n, d=1/n, dtype=torch.float32, device=device) for n in dimensions]
        k_grid = torch.meshgrid(*k_freq, indexing='ij')
        k2_tensor = sum(k**2 for k in k_grid)
        self.register_buffer('k2', k2_tensor / k2_tensor.max())

    def evolve(self, field_state, steps=1, low_pass_damping=5.0, high_pass_gain=0.0):
        with torch.cuda.amp.autocast(enabled=False):
            field_fft = torch.fft.fft2(field_state.float())
            low_pass_filter = torch.exp(-self.k2 * low_pass_damping)
            high_pass_filter = 1.0 - torch.exp(-self.k2 * high_pass_gain)
            final_decay = low_pass_filter * high_pass_filter
            for _ in range(steps): field_fft = field_fft * final_decay
            result = torch.fft.ifft2(field_fft).real
        return result.to(torch_dtype)

@dataclass
class StabilizationState:
    slow_field: Optional[torch.Tensor] = None
    previous_frame: Optional[torch.Tensor] = None

class FastFilterPipeline:
    def __init__(self, resolution=512):
        print("Loading VAE component...")
        self.resolution = resolution
        self.latent_size = resolution // 8
        
        # --- OPTIMIZATION: ONLY LOAD THE VAE ---
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-1", 
            subfolder="vae", 
            torch_dtype=torch_dtype
        ).to(device)
        
        self.moire_field = MoireField().to(device)
        self.slow_field_model = HolographicSlowField(dimensions=(self.latent_size, self.latent_size)).to(device)
        self.state = StabilizationState()
        self.ema_alpha = 0.92
        
        if device == "cuda": torch.cuda.empty_cache()
        print(f"Fast Filter Pipeline ready at {resolution}x{resolution}")

    @torch.no_grad()
    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(image_tensor).latent_dist.sample() * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        latent = latent / self.vae.config.scaling_factor
        image = self.vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()[0]
        return Image.fromarray((image * 255).astype(np.uint8))

    def process_frame(self, image: Image.Image, smoothing: float, low_pass: float, high_pass: float):
        self.ema_alpha = smoothing
        
        image_array = np.array(image.convert("RGB").resize((self.resolution, self.resolution))).astype(np.float32) / 127.5 - 1.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device, torch_dtype)

        with torch.cuda.amp.autocast():
            current_latent = self.encode_image(image_tensor)
            phase_shift = self.moire_field.compute_phase_shift(image_tensor, self.state.previous_frame)
            gate = torch.exp(-phase_shift.mean() * 1.5) # Simplified gating

            if self.state.slow_field is None:
                self.state.slow_field = current_latent.clone()
            else:
                blend_factor = self.ema_alpha + (1 - self.ema_alpha) * gate
                slow_field_f32 = torch.lerp(current_latent.float(), self.state.slow_field.float(), blend_factor)
                self.state.slow_field = slow_field_f32.to(torch_dtype)

            evolved_field = self.slow_field_model.evolve(
                self.state.slow_field, steps=2, 
                low_pass_damping=low_pass, high_pass_gain=high_pass
            )
            self.state.slow_field = evolved_field

        output_image = self.decode_latent(evolved_field)
        self.state.previous_frame = image_tensor
        
        debug_info = {'gate': gate.item(), 'phase': phase_shift.mean().item(), 'field_std': evolved_field.std().item()}
        del image_tensor, current_latent
        return output_image, debug_info

class FastFilterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Matrix9: Fast Latent Space Filter")
        self.root.geometry("1024x768")
        
        self.pipeline = None
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam Error", "Cannot open webcam"); self.root.destroy(); return
            
        self.processing = True
        self.current_frame = None
        self.result_queue = queue.Queue()
        self.current_resolution = 512
        
        self.setup_gui()
        self.init_pipeline()
        
        threading.Thread(target=self.capture_loop, daemon=True).start()
        threading.Thread(target=self.process_loop, daemon=True).start()
        self.update_gui()
        
    def init_pipeline(self):
        if self.pipeline: del self.pipeline; gc.collect(); torch.cuda.empty_cache()
        self.pipeline = FastFilterPipeline(resolution=self.current_resolution)
        self.status_label.config(text=f"Pipeline loaded at {self.current_resolution}x{self.current_resolution}")

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1); self.root.rowconfigure(0, weight=1)

        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky="ew", pady=5)
        main_frame.columnconfigure(0, weight=1)

        def add_slider(parent, row, text, var, v_from, v_to, v_res):
            label = ttk.Label(parent, text=text, width=15)
            label.grid(row=row, column=0, sticky="w", padx=5)
            scale = Scale(parent, from_=v_from, to=v_to, resolution=v_res, variable=var, orient=HORIZONTAL, length=250)
            scale.grid(row=row, column=1, sticky="ew", padx=5)
            value_label = ttk.Label(parent, text=f"{var.get():.2f}", width=5)
            value_label.grid(row=row, column=2, padx=5)
            scale.configure(command=lambda v: value_label.config(text=f"{float(v):.2f}"))
            parent.columnconfigure(1, weight=1)

        r=0
        ttk.Label(control_frame, text="Resolution:", width=15).grid(row=r, column=0, sticky="w", padx=5)
        self.resolution_var = tk.IntVar(value=self.current_resolution)
        resolution_menu = ttk.Combobox(control_frame, textvariable=self.resolution_var, values=[256, 384, 512], state="readonly")
        resolution_menu.grid(row=r, column=1, sticky="w", padx=5)
        resolution_menu.bind("<<ComboboxSelected>>", lambda e: self.init_pipeline())

        r+=1
        self.smoothing_var = tk.DoubleVar(value=0.80)
        add_slider(control_frame, r, "Smoothing:", self.smoothing_var, 0.0, 0.99, 0.01)

        r+=1
        self.gist_var = tk.DoubleVar(value=5.0)
        add_slider(control_frame, r, "Gist (Low-Pass):", self.gist_var, 0.0, 20.0, 0.1)

        r+=1
        self.detail_var = tk.DoubleVar(value=0.0)
        add_slider(control_frame, r, "Detail (High-Pass):", self.detail_var, 0.0, 50.0, 0.5)

        video_frame = ttk.Frame(main_frame)
        video_frame.grid(row=1, column=0, pady=10)
        self.webcam_label = ttk.Label(ttk.LabelFrame(video_frame, text="Webcam Input", padding=5)); self.webcam_label.pack(); self.webcam_label.master.grid(row=0, column=0, padx=10)
        self.output_label = ttk.Label(ttk.LabelFrame(video_frame, text="Smoothed Output", padding=5)); self.output_label.pack(); self.output_label.master.grid(row=0, column=1, padx=10)
        
        debug_frame = ttk.LabelFrame(main_frame, text="Metrics", padding="10")
        debug_frame.grid(row=2, column=0, sticky="ew", pady=5)
        self.fps_label = ttk.Label(debug_frame, text="FPS: --"); self.fps_label.pack(side="left", padx=10)
        self.vram_label = ttk.Label(debug_frame, text="VRAM: --"); self.vram_label.pack(side="left", padx=10)
        
        self.status_label = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.grid(row=3, column=0, sticky="ew", pady=5)

    def capture_loop(self):
        while True:
            ret, frame = self.cap.read()
            if ret: self.current_frame = frame
            time.sleep(1/60) # Capture at high FPS
            
    def process_loop(self):
        last_time = time.time()
        while True:
            if self.processing and self.current_frame is not None and self.pipeline:
                try:
                    pil_image = Image.fromarray(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB))
                    result_image, debug_info = self.pipeline.process_frame(
                        pil_image, self.smoothing_var.get(), self.gist_var.get(), self.detail_var.get()
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
            else:
                time.sleep(0.01)
                
    def update_gui(self):
        if self.current_frame is not None:
            img = Image.fromarray(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)).resize((400, 300), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.webcam_label.imgtk = imgtk
            self.webcam_label.config(image=imgtk)
            
        try:
            result_image, debug_info = self.result_queue.get_nowait()
            img = result_image.resize((400, 300), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.output_label.imgtk = imgtk
            self.output_label.config(image=imgtk)
            self.fps_label.config(text=f"FPS: {debug_info['fps']:.1f}")
            if 'vram' in debug_info:
                self.vram_label.config(text=f"VRAM: {debug_info['vram'][0]:.1f}/{debug_info['vram'][1]:.1f} GB")
        except queue.Empty: pass
        self.root.after(16, self.update_gui)
        
    def cleanup(self):
        self.processing = False
        if self.cap.isOpened(): self.cap.release()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = FastFilterGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()