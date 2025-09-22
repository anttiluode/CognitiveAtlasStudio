import os
import sys
# --- Triton autotuner monkeypatch for compatibility ---
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
import types
import threading
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

# --- Environment Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Running on device: {device} with dtype: {torch_dtype}")

# ============================================================================ #
#         SECTION 1: CORE ARCHITECTURAL COMPONENTS (4-CHANNEL REVISION)        #
# ============================================================================ #

class HolographicField(nn.Module):
    """A field that evolves based on wave dynamics. Now handles multi-channel fields."""
    def __init__(self, dimensions=(64, 64), num_channels=1):
        super().__init__()
        self.dimensions = dimensions
        # Damping map is now channel-specific for richer dynamics
        self.damping_map = nn.Parameter(torch.full((1, num_channels, *dimensions), 0.02, dtype=torch.float32))
        
        k_freq = [torch.fft.fftfreq(n, d=1 / n) for n in dimensions]
        k_grid = torch.meshgrid(*k_freq, indexing='ij')
        k2 = sum(k ** 2 for k in k_grid)
        self.register_buffer('k2', k2)

    def evolve(self, field_state, steps=1):
        """Evolve the field state for a number of steps using spectral methods."""
        field_fft = torch.fft.fft2(field_state)
        # Apply channel-specific damping
        decay = torch.exp(-self.k2.unsqueeze(0).unsqueeze(0) * F.softplus(self.damping_map))
        for _ in range(steps):
            field_fft *= decay
        return torch.fft.ifft2(field_fft).real

class SensoryEncoder(nn.Module):
    """The 'Eye' and 'V1'. Encodes images to a single-channel fast field."""
    def __init__(self, field_dims=(64, 64)):
        super().__init__()
        self.field = HolographicField(field_dims, num_channels=1) # Fast field is 1-channel
        self.image_to_drive = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.AdaptiveAvgPool2d(field_dims)
        )
        self.gamma_freq = 7.5
        self.receptive_threshold = 0.0

    def get_gamma_phase(self):
        return (time.time() * self.gamma_freq * 2 * np.pi) % (2 * np.pi)

    def is_receptive_phase(self, phase):
        return np.cos(phase) > self.receptive_threshold

    def forward(self, image_tensor):
        drive_pattern = self.image_to_drive(image_tensor)
        fast_pattern = self.field.evolve(drive_pattern, steps=5)
        phase = self.get_gamma_phase()
        receptive = self.is_receptive_phase(phase)
        return fast_pattern, phase, receptive

class AttentionalFieldComputer(nn.Module):
    """The complete, grounded perception-prediction-intention loop with a 4-channel conceptual field."""
    def __init__(self, text_encoder, vae):
        super().__init__()
        self.fast_field_dims = (64, 64)
        self.slow_field_dims = (64, 64) # Match VAE latent dimensions
        self.latent_channels = 4       # Match VAE latent channels

        # --- Sub-systems ---
        self.sensory_encoder = SensoryEncoder(self.fast_field_dims)
        self.conceptual_field = HolographicField(self.slow_field_dims, num_channels=self.latent_channels)
        self.vae = vae
        self.text_encoder = text_encoder

        # --- Pathways ---
        self.promoter = nn.Sequential(
            nn.Upsample(size=self.slow_field_dims, mode='bicubic', align_corners=False),
            nn.Conv2d(1, self.latent_channels, kernel_size=7, padding=3), nn.Tanh()
        )
        
        text_embedding_dim = self.text_encoder.config.hidden_size
        slow_field_flat_dim = np.prod(self.slow_field_dims) * self.latent_channels
        self.text_to_field_projector = nn.Sequential(
            nn.Linear(text_embedding_dim, slow_field_flat_dim),
            nn.Tanh()
        )

    @torch.no_grad()
    def forward(self, webcam_frame, text_embedding, slow_field_state, prompt_strength=0.5):
        # --- PATH 1: SENSORY PERCEPTION ---
        fast_pattern, phase, receptive = self.sensory_encoder(webcam_frame)
        
        # --- PATH 2: INTENTIONAL GUIDANCE ---
        sentence_embedding = text_embedding.mean(dim=1).to(torch.float32)
        goal_field_flat = self.text_to_field_projector(sentence_embedding)
        goal_field = goal_field_flat.view(1, self.latent_channels, *self.slow_field_dims)

        # --- INTEGRATION & CONCEPTUALIZATION ---
        new_slow_field_state = slow_field_state
        if receptive:
            promoted_sensory = self.promoter(fast_pattern)
            # Combine the persistent state, new sensory info, and the intentional goal
            new_slow_field_state = new_slow_field_state + promoted_sensory + (goal_field * prompt_strength)
        
        evolved_slow_field = self.conceptual_field.evolve(new_slow_field_state, steps=10)
        
        # --- PREDICTION / "MIND'S EYE" ---
        latent_for_decoder = evolved_slow_field / self.vae.config.scaling_factor
        predicted_percept = self.vae.decode(latent_for_decoder.to(torch_dtype)).sample
        
        return evolved_slow_field, predicted_percept, fast_pattern, receptive, phase

# ============================================================================ #
#         SECTION 2: GUI APPLICATION                                           #
# ============================================================================ #

class LiveDemoApp:
    def __init__(self, root, model, tokenizer):
        self.root = root
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        
        self.slow_field_state = torch.zeros(1, self.model.latent_channels, *self.model.slow_field_dims, device=device)
        self.root.title("Attentional Field Computer - Live Demo")
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam Error", "Cannot open webcam."); self.root.destroy(); return
            
        self.transform = T.Compose([
            T.ToTensor(), 
            T.Resize((512, 512), antialias=True),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.setup_gui()
        self.update_loop()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        vis_frame = ttk.Frame(main_frame)
        vis_frame.pack(fill=tk.BOTH, expand=True)
        for i in range(2):
            vis_frame.grid_rowconfigure(i, weight=1)
            vis_frame.grid_columnconfigure(i, weight=1)

        ttk.Label(top_frame, text="Text Prompt (Intention):").pack(side=tk.LEFT, padx=5)
        self.prompt_var = tk.StringVar(value="cinematic, epic, hyperrealistic, portrait of a man")
        self.prompt_entry = ttk.Entry(top_frame, textvariable=self.prompt_var, width=50)
        self.prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(top_frame, text="Prompt Strength:").pack(side=tk.LEFT, padx=5)
        self.strength_var = tk.DoubleVar(value=0.5)
        ttk.Scale(top_frame, from_=0.0, to=1.0, variable=self.strength_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, padx=5)

        frame1 = self.create_display_frame(vis_frame, "1. Live Input & 'Thalamic Gate'", 0, 0)
        self.input_label = ttk.Label(frame1)
        self.input_label.pack(pady=5, expand=True)
        self.gate_label = ttk.Label(frame1, text="GATE: ...", font=("Helvetica", 16, "bold"))
        self.gate_label.pack(pady=10)

        frame2 = self.create_display_frame(vis_frame, "2. Fast Field (Sensory Pattern)", 1, 0)
        self.fig_fast, self.ax_fast = plt.subplots(figsize=(3, 3))
        self.canvas_fast = self.add_plot_to_frame(self.fig_fast, frame2)

        frame3 = self.create_display_frame(vis_frame, "3. Slow Field (Conceptual Attractor)", 0, 1)
        self.fig_slow, self.ax_slow = plt.subplots(figsize=(3, 3))
        self.canvas_slow = self.add_plot_to_frame(self.fig_slow, frame3)

        frame4 = self.create_display_frame(vis_frame, "4. Prediction (Mind's Eye)", 1, 1)
        self.prediction_label = ttk.Label(frame4)
        self.prediction_label.pack(pady=5, expand=True)

    def create_display_frame(self, parent, title, r, c):
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.grid(row=r, column=c, sticky="nsew", padx=10, pady=10)
        return frame

    def add_plot_to_frame(self, fig, frame):
        fig.tight_layout(pad=0.5)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        return canvas

    def update_plot(self, ax, canvas, data, cmap='viridis'):
        ax.clear()
        ax.imshow(data, cmap=cmap)
        ax.axis('off')
        canvas.draw()

    @torch.no_grad()
    def update_loop(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)

            img_tk = ImageTk.PhotoImage(img_pil.resize((320, 240)))
            self.input_label.config(image=img_tk)
            self.input_label.image = img_tk

            input_tensor = self.transform(img_pil).unsqueeze(0).to(device)
            
            prompt = self.prompt_var.get()
            text_inputs = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = self.model.text_encoder(text_inputs.input_ids.to(device))[0]
            
            prompt_strength = self.strength_var.get()

            new_slow, prediction, fast_pattern, receptive, phase = self.model(
                input_tensor, text_embeddings, self.slow_field_state, prompt_strength
            )
            self.slow_field_state = new_slow.detach()

            gate_text = f"GATE: {'OPEN' if receptive else 'CLOSED'} (Phase: {np.degrees(phase):.1f}°)"
            self.gate_label.config(text=gate_text, foreground="green" if receptive else "red")
            
            self.update_plot(self.ax_fast, self.canvas_fast, fast_pattern.cpu().squeeze().numpy(), cmap='inferno')
            self.update_plot(self.ax_slow, self.canvas_slow, self.slow_field_state.cpu().squeeze(0)[0].numpy(), cmap='magma')
            
            pred_np = prediction.cpu().squeeze().permute(1, 2, 0).numpy()
            pred_np = (pred_np * 0.5 + 0.5)
            pred_img = Image.fromarray((np.clip(pred_np, 0, 1) * 255).astype(np.uint8))
            pred_tk = ImageTk.PhotoImage(pred_img.resize((256, 256)))
            self.prediction_label.config(image=pred_tk)
            self.prediction_label.image = pred_tk
            
        self.root.after(33, self.update_loop)

    def on_closing(self):
        if self.cap.isOpened(): self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    try:
        print("Loading pre-trained models (Tokenizer, Text Encoder, VAE)...")
        MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
        
        tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch_dtype)
        vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch_dtype)

        text_encoder.to(device).eval()
        vae.to(device).eval()
        for param in text_encoder.parameters(): param.requires_grad = False
        for param in vae.parameters(): param.requires_grad = False
        
        afc_model = AttentionalFieldComputer(text_encoder, vae).to(device)
        
        print("Model ready. Launching GUI...")

        root = tk.Tk()
        app = LiveDemoApp(root, afc_model, tokenizer)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()