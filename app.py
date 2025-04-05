import gradio as gr
import torch
import os
import sys
import numpy as np
from pathlib import Path
import json
import time

# Set up error handling for imports
try:
    from omegaconf import OmegaConf
except ImportError:
    print("Error: omegaconf not installed. Installing now...")
    os.system("pip install omegaconf")
    from omegaconf import OmegaConf

try:
    from PIL import Image
except ImportError:
    print("Error: PIL not installed. Installing now...")
    os.system("pip install pillow")
    from PIL import Image

try:
    from einops import rearrange
except ImportError:
    print("Error: einops not installed. Installing now...")
    os.system("pip install einops")
    from einops import rearrange

try:
    from pytorch_lightning import seed_everything
except ImportError:
    print("Error: pytorch_lightning not installed. Installing now...")
    os.system("pip install pytorch-lightning==1.4.2")
    from pytorch_lightning import seed_everything

try:
    from torch import autocast
except ImportError:
    # Create a fallback if autocast is not available
    from contextlib import nullcontext as autocast

from contextlib import nullcontext

# Set up paths
REPO_ROOT = Path(__file__).parent
MODELS_PATH = REPO_ROOT / "models"
CONFIG_PATH = REPO_ROOT / "configs"
OUTPUT_PATH = REPO_ROOT / "outputs"

# Create directories
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(CONFIG_PATH / "stable-diffusion", exist_ok=True)
os.makedirs(OUTPUT_PATH / "gradio-samples", exist_ok=True)

# Path to default config
DEFAULT_CONFIG_PATH = CONFIG_PATH / "stable-diffusion/v1-inference.yaml"

# Special handling for ldm utilities - we might need to install the package first
LDM_AVAILABLE = False
try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler
    from ldm.models.diffusion.plms import PLMSSampler
    LDM_AVAILABLE = True
except ImportError:
    print("Warning: ldm package not installed or not found in path.")
    print("Trying to install package from current directory...")
    install_result = os.system("pip install -e .")
    
    try:
        from ldm.util import instantiate_from_config
        from ldm.models.diffusion.ddim import DDIMSampler
        from ldm.models.diffusion.plms import PLMSSampler
        LDM_AVAILABLE = True
        print("Successfully installed ldm package.")
    except ImportError:
        print("Error: Failed to install ldm package.")
        print("The app may not function correctly.")

# Create default config if it doesn't exist
if not os.path.exists(DEFAULT_CONFIG_PATH):
    print(f"Config file not found at {DEFAULT_CONFIG_PATH}, creating default config")
    # Simple default config - adapt as needed
    default_config = {
        "model": {
            "target": "ldm.models.diffusion.ddpm.LatentDiffusion",
            "params": {
                "linear_start": 0.00085,
                "linear_end": 0.0120,
                "num_timesteps_cond": 1,
                "log_every_t": 200,
                "timesteps": 1000,
                "first_stage_key": "jpg",
                "cond_stage_key": "txt",
                "image_size": 64,
                "channels": 4,
                "cond_stage_trainable": False,
                "conditioning_key": "crossattn",
                "monitor": "val/loss_simple_ema",
                "scale_factor": 0.18215,
                "use_ema": False
            }
        }
    }
    os.makedirs(os.path.dirname(DEFAULT_CONFIG_PATH), exist_ok=True)
    with open(DEFAULT_CONFIG_PATH, 'w') as f:
        OmegaConf.save(OmegaConf.create(default_config), f)

def load_model_from_config(config, ckpt):
    if not LDM_AVAILABLE:
        print("Error: ldm package not available. Cannot load model.")
        return None
        
    print(f"Loading model from {ckpt}")
    try:
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            print("missing keys:")
            print(m)
        if len(u) > 0:
            print("unexpected keys:")
            print(u)
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            model.cuda()
            print("Using GPU for inference")
        else:
            print("CUDA not available, using CPU for inference (this will be very slow)")
        
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def generate_image(prompt, steps=50, use_plms=True, n_samples=1, guidance_scale=7.5):
    if not LDM_AVAILABLE:
        return Image.new("RGB", (512, 512)), "Error: ldm package not available. Cannot generate images."
        
    # Check if model path exists, if not inform user
    model_path = "models/ldm/stable-diffusion-v1/model.ckpt"
    if not os.path.exists(model_path):
        return Image.new("RGB", (512, 512)), f"Error: Model not found at {model_path}. Please download the model weights first."
    
    # Load config
    try:
        config = OmegaConf.load(DEFAULT_CONFIG_PATH)
    except Exception as e:
        return Image.new("RGB", (512, 512)), f"Error loading config: {e}"
    
    # Load model
    model = load_model_from_config(config, model_path)
    if model is None:
        return Image.new("RGB", (512, 512)), "Error: Failed to load model."
    
    # Set sampler
    try:
        sampler = PLMSSampler(model) if use_plms else DDIMSampler(model)
    except Exception as e:
        return Image.new("RGB", (512, 512)), f"Error creating sampler: {e}"
    
    # Generate random seed
    seed = torch.randint(0, 2**32, (1,)).item()
    
    # Set seed for reproducibility
    seed_everything(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs("outputs/gradio-samples", exist_ok=True)
    
    # Prepare batch data
    batch_size = n_samples
    prompt_batch = batch_size * [prompt]
    
    # Set up parameters
    H = 512  # Height
    W = 512  # Width
    C = 4    # Channels
    f = 8    # Downsampling factor
    
    # Determine shape
    shape = [C, H // f, W // f]
    
    # Determine device and precision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision_scope = autocast if torch.cuda.is_available() else nullcontext
    
    try:
        with torch.no_grad():
            with precision_scope(device):
                # Encode prompt
                cond = model.get_learned_conditioning(prompt_batch)
                
                # Random initial noise
                uc = None
                if guidance_scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])
                    
                # Get noise shape
                samples_ddim, _ = sampler.sample(
                    S=steps,
                    conditioning=cond,
                    batch_size=batch_size,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=guidance_scale,
                    unconditional_conditioning=uc,
                    eta=0.0
                )
                
                # Decode samples
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                
                # Convert to image
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                
                # Use first generated image if multiple
                x_sample = x_samples_ddim[0]
                
                # Convert to PIL image
                x_sample = (255. * x_sample).astype(np.uint8)
                img = Image.fromarray(x_sample)
                
                # Save image
                img_path = f"outputs/gradio-samples/{seed}.png"
                img.save(img_path)
                
                device_info = "GPU" if torch.cuda.is_available() else "CPU"
                return img, f"Seed: {seed} | Generated using: {device_info}"
    except Exception as e:
        print(f"Error during image generation: {e}")
        return Image.new("RGB", (512, 512)), f"Error generating image: {str(e)}"

def health_check():
    """Simple health check endpoint for Render"""
    status_info = {
        "status": "ok",
        "timestamp": time.time(),
        "environment": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "ldm_available": LDM_AVAILABLE
        }
    }
    
    # Format as JSON for API responses
    return status_info

# Create the Gradio interface if we can
try:
    # Create the Gradio interface
    interface = gr.Interface(
        fn=generate_image,
        inputs=[
            gr.Textbox(label="Prompt", placeholder="Enter your text prompt here..."),
            gr.Slider(minimum=20, maximum=150, value=50, step=1, label="Steps"),
            gr.Checkbox(label="Use PLMS Sampler", value=True),
            gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Samples"),
            gr.Slider(minimum=1.0, maximum=15.0, value=7.5, step=0.5, label="Guidance Scale")
        ],
        outputs=[
            gr.Image(type="pil", label="Generated Image"),
            gr.Textbox(label="Info")
        ],
        title="Stable Diffusion Text-to-Image",
        description="Generate images from text using Stable Diffusion. Enter a prompt and click submit to generate an image.",
        examples=[
            ["A painting of a cat in the style of Van Gogh"],
            ["A photograph of an astronaut riding a horse on Mars"],
            ["A fantasy landscape with mountains and a castle, trending on artstation"],
            ["Photorealistic portrait of a cyberpunk character with neon lights"]
        ]
    )

    # Add health check endpoint
    interface.add_api_route("/health", health_check, methods=["GET"])

    if __name__ == "__main__":
        print("Starting Stable Diffusion Web UI")
        print("Note: Make sure you have downloaded the model weights first")
        
        # Get the port from environment variable for Render deployment
        port = int(os.environ.get("PORT", 7860))
        
        print(f"Launching server on port {port}")
        
        # Use 0.0.0.0 to bind to all interfaces
        interface.launch(server_name="0.0.0.0", server_port=port, share=False)
except Exception as e:
    print(f"Error setting up Gradio interface: {e}")
    
    # Create a simple HTTP server as a fallback
    if __name__ == "__main__":
        import http.server
        import socketserver
        
        class HealthHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(health_check()).encode())
                else:
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(b"Stable Diffusion API is running. The UI failed to load.")
        
        port = int(os.environ.get("PORT", 7860))
        with socketserver.TCPServer(("0.0.0.0", port), HealthHandler) as httpd:
            print(f"Serving emergency fallback server at port {port}")
            httpd.serve_forever()