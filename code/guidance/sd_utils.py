from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
)
from typing import List
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import load_image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
# from guidance.control_sd import refine_imgs
# #####from guidance.inpaint import refine_imgs
from control_lora_v3.refine import refine_image
# from control_lora_v3.refine_sd import refine_image
from diffusers import ControlNetModel
import cv2
from PIL import Image
import numpy as np
# from diffusers.models.embeddings import ImageProjection
from ip_adapter import IPAdapter, StableDiffusionImg2ImgPipeline
# from diffusers.pipelines.controlnet.pipeline_controlnet
from tqdm.auto import tqdm
from torchvision.transforms import ToTensor, ToPILImage
from typing import List, Optional, Union
import math
from PIL import Image, ImageFilter, ImageEnhance
import lpips


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def largest_factor_near_sqrt(n: int) -> int:
    """
    Finds the largest factor of n that is closest to the square root of n.

    Args:
        n (int): The integer for which to find the largest factor near its square root.

    Returns:
        int: The largest factor of n that is closest to the square root of n.
    """
    sqrt_n = int(math.sqrt(n))  # Get the integer part of the square root

    # First, check if the square root itself is a factor
    if sqrt_n * sqrt_n == n:
        return sqrt_n

    # Otherwise, find the largest factor by iterating from sqrt_n downwards
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return i

    # If n is 1, return 1
    return 1

def make_image_grid(
    images: List[Image.Image],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    resize: Optional[int] = None,
) -> Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    if rows is None and cols is not None:
        assert len(images) % cols == 0
        rows = len(images) // cols
    elif cols is None and rows is not None:
        assert len(images) % rows == 0
        cols = len(images) // rows
    elif rows is None and cols is None:
        rows = largest_factor_near_sqrt(len(images))
        cols = len(images) // rows

    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


class StableDiffusion(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        sd_version="1.5",
        hf_key=None,
        t_range=[0.02, 0.98],
    ):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif self.sd_version == "2.1":
            model_key = "/root/autodl-tmp/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06"
        elif self.sd_version == "2.0":
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == "1.5":
            model_key = "/root/autodl-tmp/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
        else:
            raise ValueError(
                f"Stable-diffusion version {self.sd_version} not supported."
            )

        self.dtype = torch.float16 if fp16 else torch.float32
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )



        vae = AutoencoderKL.from_pretrained("/root/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots/31f26fdeee1355a5c34592e401dd41e45d25a493").to(dtype=torch.float16)

        # Create model
        # controlnet=ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny",torch_dtype=torch.float16,cache_dir="/root/autodl-tmp").to(dtype=torch.float16)
        # controlnet=ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_scribble",torch_dtype=torch.float16,cache_dir="/root/autodl-tmp").to(dtype=torch.float16)
        controlnet=ControlNetModel.from_pretrained("/root/autodl-tmp/models--lllyasviel--control_v11p_sd15_normalbae/snapshots/cb7296e6587a219068e9d65864e38729cd862aa8",torch_dtype=torch.float16,cache_dir="/root/autodl-tmp").to(dtype=torch.float16)
        # controlnet=ControlNetModel.from_pretrained("frankjoshua/control_v11f1p_sd15_depth",torch_dtype=torch.float16,cache_dir="/root/autodl-tmp").to(dtype=torch.float16)
        
        
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_key, 
            # torch_dtype=self.dtype,\
            torch_dtype=torch.float16, 
            controlnet=controlnet,
            # scheduler=noise_scheduler,
            vae=vae,
            safety_checker=None
        )
        
        self.lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
        self.lpips_fn.eval()

        for p in self.lpips_fn.parameters():
            p.requires_grad = False

        
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_key,
            torch_dtype=torch.float16,
            # scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        )

        # self.pipe = StableDiffusionPipeline.from_pretrained(
        #     model_key,
        #     torch_dtype=torch.float16,
        #     # scheduler=noise_scheduler,
        #     vae=vae,
        #     feature_extractor=None,
        #     safety_checker=None
        # )


        if vram_O:
            self.pipe.enable_sequential_cpu_offload()
            self.pipe.enable_vae_slicing()
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
       
            self.pipe.to(device)
            pipe.to(device)
       
        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet

        # self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

       
       
        # self.pipe.set_progress_bar_config(disable=True)
        # self.controlnet = pipe.controlnet
        
        self.pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_xformers_memory_efficient_attention()
     

        self.pipe.load_ip_adapter(
    ["/root/autodl-tmp/models--h94--IP-Adapter/snapshots/018e402774aeeddd60609b4ecdb7e298259dc729" ],
    subfolder=["models"],
    weight_name=[ "ip-adapter_sd15.bin"],  #ip-adapter-plus_sd15.bin         ip-adapter_sd15.bin
    image_encoder_folder="image_encoder"  # 使用多个图像编码器文件夹
)

        self.pipe.set_ip_adapter_scale(1.0)
       
        image_encoder_path="/root/autodl-tmp/models--h94--IP-Adapter/snapshots/018e402774aeeddd60609b4ecdb7e298259dc729/models/image_encoder"
        ip_ckpt = "/root/autodl-tmp/models--h94--IP-Adapter/snapshots/018e402774aeeddd60609b4ecdb7e298259dc729/models/ip-adapter_sd15.bin"
        self.ip_model = IPAdapter(pipe , image_encoder_path, ip_ckpt, device)
        # self.image_encoder =pipe.image_encoder
        # self.feature_extractor=pipe.feature_extractor



        # self.scheduler = DDIMScheduler.from_pretrained(
        #     model_key, subfolder="scheduler" , torch_dtype=self.dtype
        # )

        # self.scheduler = noise_scheduler
        

       

        # del pipe

        # self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        # self.min_step = int(self.num_train_timesteps * t_range[0])
        # self.max_step = int(self.num_train_timesteps * t_range[1])
        # self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience
       
        self.embeddings = {}

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds

        # directional embeddings
        for d in ['front', 'side', 'back']:
            embeds = self.encode_text([f'{p}, {d} view' for p in prompts])
            # print([f'{p}, {d} view' for p in prompts])
            self.embeddings[d] = embeds
        # print("self.embeddings",self.embeddings)
    
    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings
    

    
 
        
    
    # def get_image_embeds(self, pil_image):
    #     to_pil =transforms.ToPILImage()
        

    #     if isinstance(pil_image, Image.Image):
    #         pil_image = [pil_image]
    #     else:
    #         pil_image = to_pil(pil_image.squeeze(0))
    #         # print("pil_image****************************************************",pil_image)
        
    #     clip_image = self.feature_extractor(pil_image, return_tensors="pt").pixel_values
    #     clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
    #     print("clip_image_embeds",clip_image_embeds.shape)
    #     image_prompt_embeds = self.image_proj_model(clip_image_embeds)
    #     print("image_prompt_embeds",image_prompt_embeds.shape)
    #     uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
    #     return image_prompt_embeds, uncond_image_prompt_embeds

    
    
    def refine2(self, path = None, real_renders=None, guidance_scale=5, steps=1, strength=0.5,direction="",valid_mask2=0,edge_images=None):
        transform = transforms.ToPILImage()
        pred_rgb = transform(real_renders.squeeze(0)).convert('RGB')
        refine_img = refine_image(path = path,direction=direction,input = pred_rgb,strength=strength,steps=steps,edge_images=edge_images)     
        # refine_img = transforms.ToTensor()(refine_img).to(self.device)
        return refine_img


    # Sample function (regular DDIM)
    @torch.no_grad() #no face and mouth and eyes on back, 
    def sample(self, prompt="back view, best quality, high quality", start_step=0, start_latents=None,
            guidance_scale=3.5, num_inference_steps=20,
            num_images_per_prompt=1, do_classifier_free_guidance=True,
            negative_prompt='extra face, face on back, distorted body, extra head, malformed body, unnatural anatomy, duplicate face, wrong body structure',image_embeds=None,control_img=None,la=None):
        
        device=self.device
        # print("*2",prompt)
        # Encode prompt
        if isinstance(prompt, str):
            text_embeddings = self.pipe._encode_prompt(
                    prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
        else:
            text_embeddings = prompt

        # Set num inference steps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)

        # Create a random starting point if we don't have one already
        if start_latents is None:
            start_latents = torch.randn(1, 4, 64, 64, device=device)
            start_latents *= self.pipe.scheduler.init_noise_sigma

        latents = start_latents.clone()
        latents = start_latents #+ 0.08*la
        # print(la.shape,start_latents.shape) torch.Size([1, 4, 64, 512])
        self.pipe.set_progress_bar_config(disable=True)

        for i in tqdm(range(start_step, num_inference_steps)):
        
            t = self.pipe.scheduler.timesteps[i]

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            # prepare_ip_adapter_image_embeds
            added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if image_embeds is not None
            else None
        )
            # print(image_embeds.shape)
        #     image_embeds2 = image_embeds
        #     image_embeds2[0] = image_embeds[0]#*0.5
        #     added_cond_kwargs2 = (
        #     {"image_embeds": image_embeds2}
        #     if image_embeds is not None
        #     else None
        # )
            # self.pipe.set_ip_adapter_scale(1)
            
            down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
            latent_model_input.to(self.dtype),
            t,
            encoder_hidden_states=text_embeddings.to(self.dtype),
            controlnet_cond=control_img.to(self.dtype),
            conditioning_scale=1.0,
            return_dict=False,
            )




            # Predict the noise residual

            skip=None
            noise_pred,skip = self.pipe.unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=text_embeddings,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample
                )
            noise_pred,_ = self.pipe.unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=text_embeddings,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                skip = skip
                )
            noise_pred=noise_pred.sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


            # Normally we'd rely on the scheduler to handle the update step:
            # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

            # Instead, let's do it ourselves:
            prev_t = max(1, t.item() - (1000//num_inference_steps)) # t-1
            alpha_t = self.pipe.scheduler.alphas_cumprod[t.item()]
            alpha_t_prev = self.pipe.scheduler.alphas_cumprod[prev_t]
            predicted_x0 = (latents - (1-alpha_t).sqrt()*noise_pred) / alpha_t.sqrt()
            direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*noise_pred
            latents = alpha_t_prev.sqrt()*predicted_x0 + direction_pointing_to_xt

        # Post-processing
        images = self.pipe.decode_latents(latents)
        images = self.pipe.numpy_to_pil(images)

        return images


    ## Inversion
    @torch.no_grad()#back view, no face on back #no face and mouth and eyes on back,
    def invert(self,start_latents, prompt="back view, best quality, high quality", guidance_scale=3.5, num_inference_steps=80,
            num_images_per_prompt=1, do_classifier_free_guidance=True,
            negative_prompt='extra face, face on back, distorted body, extra head, malformed body, unnatural anatomy, duplicate face, wrong body structure',image_embeds=None,control_img=None):
        device=self.device
        # print("****guidance_scale*****",prompt,negative_prompt,guidance_scale)#monochrome, lowres, bad anatomy, worst quality, low quality
   
        # Encode prompt
        # print("*1",prompt)
        if isinstance(prompt, str):
            text_embeddings = self.pipe._encode_prompt(
                    prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
        else:
            text_embeddings = prompt
        # print(text_embeddings.shape)   torch.Size([2, 77, 768])

        # Latents are now the specified start latents
        latents = start_latents.clone()

        # We'll keep a list of the inverted latents as the process goes on
        intermediate_latents = []

        # Set num inference steps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        # self.pipe.set_progress_bar_config(disable=True)

        # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
        timesteps = reversed(self.pipe.scheduler.timesteps)

        for i in tqdm(range(1, num_inference_steps), total=num_inference_steps-1):

            # We'll skip the final iteration
            if i >= num_inference_steps - 1: continue
            # print(timesteps)

            t = timesteps[i]

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            # print("text_embeddings",text_embeddings.shape)
            # print(latent_model_input.shape)
            # print(t)
            # self.pipe.set_ip_adapter_scale(0)
            added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if image_embeds is not None
            else None)

            # print("*****",len(image_embeds),image_embeds[0].shape,"*****") torch.Size([2, 1, 257, 1280])
        


            # control_img = control_img
            # print(control_img.shape) ([2, 3, 512, 512])

            down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
            latent_model_input.to(self.dtype),
            t,
            encoder_hidden_states=text_embeddings.to(self.dtype),
            controlnet_cond=control_img.to(self.dtype),
            conditioning_scale=1.0,
            return_dict=False,
            )

            # Predict the noise residual
            noise_pred,_ = self.unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=text_embeddings,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample)
            noise_pred=noise_pred.sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            current_t = max(0, t.item() - (1000//num_inference_steps)) #t
            next_t = t # min(999, t.item() + (1000//num_inference_steps)) # t+1
            alpha_t = self.pipe.scheduler.alphas_cumprod[current_t]
            alpha_t_next = self.pipe.scheduler.alphas_cumprod[next_t]

            # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
            latents = (latents - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred


            # Store
            intermediate_latents.append(latents)
                
        return torch.cat(intermediate_latents)
    

    def get_prompt(
            self,
            refer_img,
            num_samples,
            prompt=None,
            negative_prompt=None,
            image_prompt_delta=None,
            ):

        if isinstance(refer_img, Image.Image):
            num_prompts = 1
        else:
            num_prompts = len(refer_img)
        
        if prompt is None:
            prompt = "back view, no face on back view, best quality, high quality"
        else:
            prompt = prompt + "back view, no face on back view, best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "extra face, face on back, distorted body, extra head, malformed body, unnatural anatomy, duplicate face, wrong body structure,monochrome, lowres, bad anatomy, worst quality, low quality"
            
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.ip_model.get_image_embeds(refer_img)
        if image_prompt_delta != None:
            image_prompt_embeds = image_prompt_embeds + image_prompt_delta
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds = self.ip_model.pipe._encode_prompt(
                prompt, device=self.device, num_images_per_prompt=num_samples, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
            negative_prompt_embeds_, prompt_embeds_ = prompt_embeds.chunk(2)
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
        
        prompt_embeds = self.ip_model.pipe._encode_prompt(
            prompt=None,
            device=self.device,
            num_images_per_prompt=num_samples,
            do_classifier_free_guidance=True,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        return prompt_embeds
    
    def edit(self, input_image, input_image_prompt, edit_prompt, num_steps=100, start_step=30, guidance_scale=3.5,image_embeds=None,control_image=None):
        # print(input_image.shape)
        with torch.no_grad(): 
            latent = self.pipe.vae.encode(input_image.unsqueeze(0)*2-1)

        l = 0.18215 * latent.latent_dist.sample()
        
        control_img = control_image.unsqueeze(0).repeat(2, 1, 1, 1).to(self.device)

        #input_image_prompt,  edit_prompt,
        inverted_latents = self.invert(l,  num_inference_steps=num_steps,image_embeds=image_embeds,control_img=control_img)
        final_im = self.sample( prompt=edit_prompt, start_latents=inverted_latents[-(start_step+1)][None], la=l,
                        start_step=start_step, num_inference_steps=num_steps, guidance_scale=guidance_scale,image_embeds=image_embeds,control_img=control_img)[0]
        return final_im
    
    def high_pass_filter2(self,style_image):
        high_pass_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        high_pass_kernel = high_pass_kernel.repeat(3, 1, 1, 1)

        high_pass_image = F.conv2d(style_image, high_pass_kernel, padding=1, groups=3)
        high_pass_image = style_image + high_pass_image
        high_pass_image = torch.clamp(high_pass_image, 0, 1)
        return high_pass_image
    
    def train_step(
        self,
        pred_rgbs,
        step_ratio=None,
        guidance_scale=100,
        as_latent=False,
        vers=None, hors=None, step=1,style_path=None,control_images=None,masks=None,input_nomal=None,depth_images=None,input_depth=None, canny_images=None,input_canny=None
    ):
        
           
     
        def _get_dir_ind(h):
            if abs(h) < 60: return 'front'
            elif abs(h) < 120: return 'side'
            else: return 'back'
        def gaussian_blur(image, blur_radius=2):
            blurred_image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            return blurred_image

      

            
        final_ims=[]
        i=1
        # print(style_path)
        # seed_everything(27)

        style_image = load_image(style_path).convert("RGB")#.resize((1024, 1024), Image.Resampling.LANCZOS)
        to_tensor = ToTensor()
        tensor_image = to_tensor(style_image).unsqueeze(0)
        enhanced_image = self.high_pass_filter2(tensor_image)
        to_pil = ToPILImage()
        style_image = to_pil(enhanced_image.squeeze(0))
        # style_image = gaussian_blur(style_image, blur_radius=2)
        
        
        ip_adapter_image=[style_image]
        ip_adapter_image_embeds = None
        global image_embeds
        if step <= 1:
            image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                self.device,
                1,
                True,
            )
        # print(image_embeds[0])
        transform1 = transforms.ToPILImage()
        transform2 = transforms.ToTensor()
 
        #control_images
        control_image = make_image_grid(control_images, rows=1)
        control_image.save("muiltview/control_images.png")
        # input_images = [input_nomal]*8
        input_image = make_image_grid([input_nomal]*8, rows=1)
        normal_muiltview_prompt, _ = self.ip_model.get_image_embeds(control_image)
        normal_input_prompt, _ = self.ip_model.get_image_embeds(input_image)

        #depth_images
        depth_image = make_image_grid(depth_images, rows=1)
        depth_image.save("muiltview/depth_images.png")
        # input_images = [depth_image[0]]*8
        input_image = make_image_grid([input_depth]*8, rows=1)
        # input_image.save("muiltview/depth_image.png")
        depth_muiltview_prompt, _ = self.ip_model.get_image_embeds(depth_image)
        depth_input_prompt, _ = self.ip_model.get_image_embeds(input_image)

        #canny_images
        canny_image = make_image_grid(canny_images, rows=1)
        canny_image.save("muiltview/canny_images.png")
        # input_images = [depth_image[0]]*8
        input_image = make_image_grid([input_canny]*6, rows=1)
        # input_image.save("muiltview/depth_image.png")
        canny_muiltview_prompt, _ = self.ip_model.get_image_embeds(canny_image)
        canny_input_prompt, _ = self.ip_model.get_image_embeds(input_image)


        #img_prompt_delta
      
        img_prompt_delta = (normal_input_prompt - normal_muiltview_prompt)*10
        img_prompt_delta = (depth_input_prompt - depth_muiltview_prompt)*1 + img_prompt_delta #
        img_prompt_delta = (canny_input_prompt - canny_muiltview_prompt)*1 + img_prompt_delta 


        # img_prompt_delta = None



  
        preds = []
        i = 1
        for pred_rgb in pred_rgbs:
            render_img = transform1(pred_rgb)
            print(i)
            # render_img.save("muiltview/"+ str(i) +"pred_rgbs.png")
            i = i+1

            preds.append(render_img)  #.save("initview/"+str(i)+".png")
        pred_rgb = make_image_grid(preds, rows=1)
        pred_rgb.save("muiltview/pred_rgbs.png")

 
        # for pred_rgb,h,control_image in zip(pred_rgbs,hors,control_images):
        #     # input_image_prompt = _get_dir_ind(h)
        #     input_image_prompt = ""
       

        #     img_prompt_embeddings = self.get_prompt(style_image, 1, prompt=input_image_prompt)
            
        #     control_image = transform2(control_image)


        #     final_im=self.edit(pred_rgb.to(self.dtype), input_image_prompt=input_image_prompt, edit_prompt=img_prompt_embeddings, num_steps=50, start_step=30, guidance_scale=7.5,image_embeds=image_embeds,control_image=control_image)
            
        #     final_im.save("muiltview/"+str(i)+".png")
        #     i = i+1
        #     # 执行转换
        #     final_im = transform2(final_im).unsqueeze(0)        
        #     final_ims.append(final_im)


        input_image_prompt = ""
       

        img_prompt_embeddings = self.get_prompt(style_image, 1, prompt=input_image_prompt,image_prompt_delta=img_prompt_delta)
        
        control_image = transform2(control_image)
        pred_rgb = transform2(pred_rgb).to(self.device).to(self.dtype)


        final_im=self.edit(pred_rgb, input_image_prompt=input_image_prompt, edit_prompt=img_prompt_embeddings, num_steps=20, start_step=3, guidance_scale=7.5,image_embeds=image_embeds,control_image=control_image)
    
        #50   30
        
        final_im.save("muiltview/8图.png")

        # i = i+1
        # # 执行转换
        final_im = transform2(final_im).to(self.device)  
        # final_ims.append(final_im)
        # final_ims = torch.cat(final_ims, dim=0).to(self.device)
        # loss = F.mse_loss(pred_rgbs, final_ims, reduction='sum')
        B,C,H,W=pred_rgbs.shape
        reshaped = final_im.view(C, H, B, W)

        # 第二步：调整维度顺序，将8移到最前面
        # 形状变为[8, 3, 512, 512]
        final_im = reshaped.permute(2, 0, 1, 3)
        transform1(final_im[0]).save("muiltview/test1.png")
        # print(final_im.shape)
        transform1(final_im[2]).save("muiltview/test3.png")
        transform1(final_im[4]).save("muiltview/test5.png")
        
        
        # loss = F.mse_loss(pred_rgbs[:8]*masks[:8], final_im[:8]*masks[:8], reduction='sum')
        
        l1_loss = F.l1_loss(pred_rgbs[:8]*masks[:8], final_im[:8]*masks[:8])
        # LPIPS 需要输入到 [-1, 1]
        pred_lpips = pred_rgbs[:8] * 2.0 - 1.0
        # print("11111111111111----------",pred_rgbs[:8])
        # print("pred_rgbs[:8] min =", pred_rgbs[:8].min().item())
        target_lpips = final_im[:8] * 2.0 - 1.0

        lpips_loss = self.lpips_fn(pred_lpips, target_lpips).mean()

        # total
        loss_recon = l1_loss + lpips_loss
        loss = 10000*loss_recon

        return loss, final_im[:8]

    def train_step2(
        self,
        pred_rgb,
        step_ratio=None,
        guidance_scale=100,
        as_latent=False,
        vers=None, hors=None,
        depth=None,depth_default=None
    ):
        
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        with torch.no_grad():
            if step_ratio is not None:
                # dreamtime-like
                # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
                t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
                # print(f"t: {t}")
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
                # print("****", t)
            else:
                # print("****") # 执行
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

            # predict the noise residual with unet, NO grad!
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            prompt = None

            if hors is None:
                embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])
            else:
                def _get_dir_ind(h):
                    if abs(h) < 60: return 'front'
                    elif abs(h) < 120: return 'side'
                    else: return 'back'
                prompt = _get_dir_ind(hors[0])
                # print(prompt)

                # embeddings = torch.cat([self.embeddings[_get_dir_ind(h)] for h in hors] + [self.embeddings['neg'].expand(batch_size, -1, -1)])
      


            # print(pred_rgb.shape,latent_model_input.shape)

            # low_threshold = 100
            # high_threshold = 200

            
            # pred_rgb = (
            #     (pred_rgb_512[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8).copy()
            # )

            # image = cv2.Canny(pred_rgb, low_threshold, high_threshold)
            # image = image[:, :, None]
            # image = np.concatenate([image, image, image], axis=2)




            # control = (
            #     torch.from_numpy(np.array(image)).float().to(self.device) / 255.0
            # ).permute(2, 0, 1).unsqueeze(0).repeat(2, 1, 1, 1)
            # canny_image = Image.fromarray(image)
            # print("control",control,t[0],embeddings)
            # down_block_res_samples, mid_block_res_sample = self.controlnet(
            # latent_model_input.to(self.dtype),
            # t[0],
            # encoder_hidden_states=embeddings.to(self.dtype),
            # controlnet_cond=control.to(self.dtype),
            # conditioning_scale=1.0,
            # return_dict=False,
            # )


            # ip_adapter_image_embeds=None
            style_image = load_image("data/hen_rgba.png").convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
            # print(prompt)
            img_prompt_embeddings = self.get_prompt(style_image, 1, prompt=prompt)
            # print(img_prompt_embeddings.shape) torch.Size([2, 81, 768])
            # print(embeddings.shape) torch.Size([2, 77, 768])
            # ip_adapter_image=[style_image]
            # image_embeds = self.prepare_ip_adapter_image_embeds(
            #     ip_adapter_image,
            #     ip_adapter_image_embeds,
            #     self.device,
            #     batch_size,
            #     True,
            # )

            # image_prompt_embeds, uncond_image_prompt_embeds = self.ip_model.get_image_embeds(style_image)
            # print(image_prompt_embeds.shape)
            # print(embeddings[0].shape,image_embeds[0].shape)
            
#             depth_embeds = self.prepare_ip_adapter_image_embeds(
#                 depth,
#                 ip_adapter_image_embeds,
#                 self.device,
#                 batch_size,
#                 True,
#             )
#             default_depth_embeds = self.prepare_ip_adapter_image_embeds(
#                 depth_default,
#                 ip_adapter_image_embeds,
#                 self.device,
#                 batch_size,
#                 True,
#             )
        
#             depth_embeds_delta = 0.8*(depth_embeds[0][1] - default_depth_embeds[0][1])
#             image_embeds[0][1] = image_embeds[0][1]+depth_embeds_delta
         
        #     added_cond_kwargs = (
        #     {"image_embeds": image_embeds}
        #     if ip_adapter_image is not None or ip_adapter_image_embeds is not None
        #     else None
        # )
            # print("3",image_embeds[0].shape,embeddings[0].shape)
            noise_pred,_ = self.unet(
                latent_model_input, 
                tt, 
                encoder_hidden_states=img_prompt_embeddings,
                # cross_attention_kwargs=None, #
                # down_block_additional_residuals=down_block_res_samples,
                # mid_block_additional_residual=mid_block_res_sample,
                # added_cond_kwargs=added_cond_kwargs,
            )
            noise_pred = noise_pred.sample
            # print(noise_pred.shape)torch.Size([2, 4, 64, 64])

            # perform guidance (high scale from paper!)
            noise_pred_uncond,noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)

            # seems important to avoid NaN...
            # grad = grad.clamp(-1, 1)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') #/ latents.shape[0]

        return loss
    

    @torch.no_grad()
    def refine(self, pred_rgb,
               guidance_scale=100, steps=50, strength=0.8,
        ):

        batch_size = pred_rgb.shape[0]
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False).to(self.dtype)
        latents = self.encode_imgs(pred_rgb_512).to(self.dtype)
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])
        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])
        style_image = load_image("data/hen_rgba.png").convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
        img_prompt_embeddings = self.get_prompt(style_image, 1, prompt=None)
        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
    
            latent_model_input = torch.cat([latents] * 2)

            noise_pred,_ = self.unet(
                latent_model_input, t, encoder_hidden_states=img_prompt_embeddings,
            )
            noise_pred=noise_pred.sample

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        return imgs

    @torch.no_grad()
    def produce_latents(
        self,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    1,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        batch_size = latents.shape[0]
        self.scheduler.set_timesteps(num_inference_steps)
        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=embeddings
            ).sample

            # perform guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        self.get_text_embeds(prompts, negative_prompts)
        
        # Text embeds -> img latents
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument(
        "--sd_version",
        type=str,
        default="2.1",
        choices=["1.5", "2.0", "2.1"],
        help="stable diffusion version",
    )
    parser.add_argument(
        "--hf_key",
        type=str,
        default=None,
        help="hugging face Stable diffusion model key",
    )
    parser.add_argument("--fp16", action="store_true", help="use float16 for training")
    parser.add_argument(
        "--vram_O", action="store_true", help="optimization for low VRAM usage"
    )
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device("cuda")

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
