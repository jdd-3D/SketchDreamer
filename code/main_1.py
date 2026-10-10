# stage 1
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image

controlnet = ControlNetModel.from_pretrained(
    "xinsir/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)



pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    add_watermarker=False,
)

pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter-plus_sdxl_vit-h.bin",
    image_encoder_folder="models/image_encoder"
)# models/image_encoder

scale1 = {
    "down": {"block_2": [0.0, 1.0]}, # layout
    "up": {"block_0": [0.0, 1.0, 0.0]}, # style
}
pipe.set_ip_adapter_scale([scale1])


pipe.enable_model_cpu_offload()

input_image = load_image("img/cat.jpeg")
style_image = load_image("img/or.png")

def get_canny_image(image, low_threshold=50, high_threshold=150):
    image_np = np.array(image)
    if image_np.shape[-1] == 4:
        image_np = image_np[:, :, :3]
    if image_np.dtype != np.uint8:
        image_np = (image_np * 255).astype(np.uint8)

    import cv2
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(image_gray, low_threshold, high_threshold)
    return Image.fromarray(canny_image).convert("RGB")

control_image = get_canny_image(input_image)

prompt = ""
negative_prompt = "ugly, blurry, low quality, deformed"

generator = torch.Generator(device="cuda").manual_seed(-1)

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=input_image,
    control_image=control_image,
    strength=1.0,
    num_inference_steps=50,
    generator=generator,
    guidance_scale=7.0,
    controlnet_conditioning_scale=0.5,
    ip_adapter_image=[style_image]
)

output.images[0].save("output_image.png")
