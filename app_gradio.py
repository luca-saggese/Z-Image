"""Gradio interface for Z-Image generation."""

import os
import time
import warnings

import gradio as gr
import torch

warnings.filterwarnings("ignore")

from src.utils import AttentionBackend, ensure_model_weights, load_from_local_dir, set_attention_backend
from src.zimage import generate


# Global variables for model caching
components = None
device = None


def initialize_model():
    """Initialize and cache the model components."""
    global components, device
    
    if components is not None:
        return  # Already initialized
    
    print("Loading Z-Image model...")
    
    # Device selection priority: cuda -> tpu -> mps -> cpu
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: cuda (GPU: {torch.cuda.get_device_name(0)})")
    else:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            print("Using device: tpu")
        except (ImportError, RuntimeError):
            if torch.backends.mps.is_available():
                device = "mps"
                print("Using device: mps")
            else:
                device = "cpu"
                print("Using device: cpu")
    
    # Load model weights
    model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)
    
    # Load components
    dtype = torch.bfloat16
    components = load_from_local_dir(model_path, device=device, dtype=dtype, compile=False)
    
    # Set attention backend
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    print(f"Attention backend: {attn_backend}")
    print("Model loaded successfully!")


def generate_image(
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 8,
    guidance_scale: float = 0.0,
    seed: int = 42,
):
    """Generate an image from a text prompt."""
    
    # Initialize model if not already done
    if components is None:
        initialize_model()
    
    if not prompt.strip():
        return None, "Please enter a prompt"
    
    try:
        start_time = time.time()
        
        # Generate image
        images = generate(
            prompt=prompt,
            **components,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device).manual_seed(seed),
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Return image and generation info
        status_msg = f"✅ Image generated in {generation_time:.2f} seconds"
        return images[0], status_msg
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="Z-Image Generator",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🎨 Z-Image Generator
            Generate high-quality images from text prompts using Z-Image-Turbo
            
            **Model**: Z-Image-Turbo (Sub-second generation, 8 steps)
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input prompt
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your image description here...",
                    lines=4,
                    value="A beautiful landscape with mountains at sunset",
                )
                
                with gr.Row():
                    with gr.Column():
                        height = gr.Slider(
                            label="Height",
                            minimum=512,
                            maximum=1024,
                            step=128,
                            value=1024,
                        )
                    with gr.Column():
                        width = gr.Slider(
                            label="Width",
                            minimum=512,
                            maximum=1024,
                            step=128,
                            value=1024,
                        )
                
                with gr.Row():
                    with gr.Column():
                        num_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=8,
                        )
                    with gr.Column():
                        guidance_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=0.0,
                            maximum=7.5,
                            step=0.5,
                            value=0.0,
                        )
                
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=2**31 - 1,
                    step=1,
                    value=42,
                )
                
                generate_btn = gr.Button(
                    "🚀 Generate Image",
                    scale=1,
                    size="lg",
                    variant="primary",
                )
            
            with gr.Column(scale=1):
                # Output image
                output_image = gr.Image(
                    label="Generated Image",
                    type="pil",
                )
                
                # Status message
                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Ready to generate",
                )
        
        # Connect generate button
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, height, width, num_steps, guidance_scale, seed],
            outputs=[output_image, status],
        )
        
        # Add examples
        gr.Examples(
            examples=[
                ["A serene Japanese garden with a koi pond and cherry blossom trees", 1024, 1024, 8, 0.0],
                ["A steampunk airship flying over clouds at sunset", 1024, 1024, 8, 0.0],
                ["A cozy coffee shop with warm lighting and reading people", 1024, 1024, 8, 0.0],
                ["A fantasy dragon sitting on a mountain peak, detailed scales", 1024, 1024, 8, 0.0],
            ],
            inputs=[prompt, height, width, num_steps, guidance_scale],
            label="Example Prompts",
        )
        
        # Initialize model on load
        demo.load(fn=initialize_model)
    
    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
