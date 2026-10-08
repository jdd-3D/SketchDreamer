from diffusers import DDIMScheduler,UniPCMultistepScheduler
import torchvision.transforms.functional as TF

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL,UniPCMultistepScheduler
from diffusers.utils import load_image #, randn_tensor
import torch
import numpy as np
import cv2
from PIL import Image
import PIL.Image
from PIL import Image, ImageFilter
from torchvision import transforms
# from guidance.control_sd import refine_imgs
# #####from guidance.inpaint import refine_imgs
# from control_lora_v3.refine import refine_imgs
from MV.scripts.inference_i2mv_sdxl import refine_imgs


import sys
sys.path.append('./')

from zero123 import Zero123Pipeline
# from perceptual import PerceptualLoss


class Zero123(nn.Module):
    def __init__(self, device, fp16=True, t_range=[0.02, 0.98], model_key="ashawkey/zero123-xl-diffusers"):
        super().__init__()

        self.device = device
        self.fp16 = fp16
        self.dtype = torch.float16 if fp16 else torch.float32# 
        # self.perceptual_loss = PerceptualLoss().eval().to(self.device)

        assert self.fp16, 'Only zero123 fp16 is supported for now.'

        # model_key = "ashawkey/zero123-xl-diffusers"
        # model_key = './model_cache/stable_zero123_diffusers'

        self.pipe = Zero123Pipeline.from_pretrained(
            model_key,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)

        # stable-zero123 has a different camera embedding
        self.use_stable_zero123 = 'stable' in model_key

        self.pipe.image_encoder.eval()
        self.pipe.vae.eval()
        self.pipe.unet.eval()
        self.pipe.clip_camera_projection.eval()

        self.vae = self.pipe.vae
        self.unet = self.pipe.unet

        self.pipe.set_progress_bar_config(disable=True)

        # self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
   
    
        self.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config,
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing"
        )
    
        
        
        
        # self.scheduler =UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        # print("self.alpha",self.alphas.shape) #1000

        self.embeddings = None

    @torch.no_grad()
    def get_img_embeds(self, x):
        # x: image tensor in [0, 1] 
        x = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
        x_pil = [TF.to_pil_image(image) for image in x]
        x_clip = self.pipe.feature_extractor(images=x_pil, return_tensors="pt").pixel_values.to(device=self.device, dtype=self.dtype)
        # print(x_clip.shape) #([1, 3, 224, 224])
        c = self.pipe.image_encoder(x_clip).image_embeds
        v = self.encode_imgs(x.to(self.dtype)) / self.vae.config.scaling_factor
        self.embeddings = [c, v]
        # print(self.embeddings[0].shape) #([1, 768])
        # print(self.embeddings[1].shape) #([1, 4, 32, 32])
        
    
    def get_cam_embeddings(self, elevation, azimuth, radius, default_elevation=0):
        if self.use_stable_zero123:
            T = np.stack([np.deg2rad(elevation), np.sin(np.deg2rad(azimuth)), np.cos(np.deg2rad(azimuth)), np.deg2rad([90 + default_elevation] * len(elevation))], axis=-1)
        else:
            # original zero123 camera embedding
            T = np.stack([np.deg2rad(elevation), np.sin(np.deg2rad(azimuth)), np.cos(np.deg2rad(azimuth)), radius], axis=-1)
        T = torch.from_numpy(T).unsqueeze(1).to(dtype=self.dtype, device=self.device) # [8, 1, 4]
        return T

    @torch.no_grad()
    def refine(self, pred_rgb, elevation, azimuth, radius, 
               guidance_scale=20, steps=50, strength=0.8, default_elevation=0,
        ):

        batch_size = pred_rgb.shape[0]

        self.scheduler.set_timesteps(steps)

        if strength == 0:
            init_step = 0
            latents = torch.randn((1, 4, 32, 32), device=self.device, dtype=self.dtype)
        else:
            init_step = int(steps * strength)
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_256.to(self.dtype))
            latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

        T = self.get_cam_embeddings(elevation, azimuth, radius, default_elevation)
        cc_emb = torch.cat([self.embeddings[0].repeat(batch_size, 1, 1), T], dim=-1)
        cc_emb = self.pipe.clip_camera_projection(cc_emb)
        cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)

        vae_emb = self.embeddings[1].repeat(batch_size, 1, 1, 1)
        vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)
        

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
            
            x_in = torch.cat([latents] * 2)
            t_in = t.view(1).to(self.device)
            # print(x_in.shape, vae_emb.shape)
            # print(torch.cat([x_in, vae_emb], dim=1).shape)

            noise_pred,skip = self.unet(
                torch.cat([x_in, vae_emb], dim=1),
                t_in.to(self.unet.dtype),
                encoder_hidden_states=cc_emb,
            )#.sample
            # print(noise_pred.sample.shape)
            noise_pred = noise_pred.sample

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 256, 256]
       
        return imgs
    
    from PIL import Image
    import numpy as np





    def refine2(self, real_renders=None, guidance_scale=5, steps=1, strength=0.5,direction="",valid_mask2=0,edge_images=None):
        transform = transforms.ToPILImage()
        # pred_rgb = transform(refined_image).convert('RGB')
        # if steps==51:
        # pred_rgb.save("test0.png")

        refines = refine_imgs(multi_view_images=edge_images)

        # 转换为张量并添加批次维度
        # refine_img.save("test.png")
        # pred_rgb.save("test0.png")
        # transform(real_render).convert('RGB').save("test2.png")


        loss = 0
        for mask, refine,real_render in zip(valid_mask2, refines, real_renders):
            refine_img = transforms.ToTensor()(refine).to(self.device)
            # print(refine_img.shape,mask.squeeze(-1).unsqueeze(0).shape,real_render.shape,mask.squeeze(0).shape)
            transform(refine_img*mask.squeeze(0)).convert('RGB').save("11111.png")
            loss = loss+ F.mse_loss(refine_img*mask.squeeze(0), real_render*mask.squeeze(0))
        return loss, refine_img, refine_img

        # #转换分布
        # #IP-adapter 的分布
        # supervise_mean= torch.mean(refine_img, dim=[1,2]).unsqueeze(-1).unsqueeze(-1)
        # supervise_variance=torch.var(refine_img, dim=[1,2], unbiased=False).unsqueeze(-1).unsqueeze(-1).sqrt()

        # #渲染图的分布
        # mean_pred_rgb = torch.mean(render_image, dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
        # variance_pred_rgb = torch.var(render_image, dim=[1, 2], unbiased=False).unsqueeze(-1).unsqueeze(-1).sqrt()
        
        # #渲染图归一化
        # normalized_pred_rgb = (render_image - mean_pred_rgb) / variance_pred_rgb
        # #将渲染图的分布替换成IP-adapter的分布
        # pred_rgb_BCHW_512 = normalized_pred_rgb * supervise_variance + supervise_mean
        # pred_rgb_BCHW_512 = torch.clamp(pred_rgb_BCHW_512, 0, 1)
        # pred = transform(pred_rgb_BCHW_512).convert('RGB')
        
        # # 保存图像
        # pred.save("test2.png")
        


        # return F.mse_loss(refine_img, render_image), pred_rgb_BCHW_512[None].contiguous(), refine_img


        
        # # 加载 IP-Adapter
        # pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin") 
        # pipe.set_ip_adapter_scale(0.6)
        # lora_model="goofyai/3d_render_style_xl/3d_render_style_xl.safetensors"
        # model_, name_ = lora_model.rsplit("/", 1)
        # pipe.load_lora_weights(model_, weight_name=name_, lora_scale=1)

        # # 启用模型 CPU 卸载以节省内存
        # pipe.enable_model_cpu_offload()

        
        
    
    def train_step(self, pred_rgb, elevation, azimuth, radius, step_ratio=None, guidance_scale= 15, as_latent=False, default_elevation=0):
        # pred_rgb: tensor [1, 3, H, W] in [0, 1]

        batch_size = pred_rgb.shape[0]

        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode='bilinear', align_corners=False) * 2 - 1
        else:
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_256.to(self.dtype))

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            # print(self.num_train_timesteps) #1000
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)
        # generator = torch.Generator(device=self.device).manual_seed(1)

        with torch.no_grad():
            # shape = (1,) + latents.shape[1:]
            # print(shape)
            
            # noise = randn_tensor(shape, generator=generator, device=self.device)
            # # noise = noise.repeat(batch_size, 1, 1, 1)
            # print(noise.shape)
            noise = torch.randn_like(latents[0].unsqueeze(0))#.repeat(batch_size, 1, 1, 1)
            # noise = noise.repeat(batch_size, 1, 1, 1)
            # print(noise.shape)
            latents_noisy = self.scheduler.add_noise(latents, noise, t) #给渲染图加噪
            # print(t)

            x_in = torch.cat([latents_noisy] * 2)
            t_in = torch.cat([t] * 2)

            T = self.get_cam_embeddings(elevation, azimuth, radius, default_elevation)
            cc_emb = torch.cat([self.embeddings[0].repeat(batch_size, 1, 1), T], dim=-1)
            cc_emb = self.pipe.clip_camera_projection(cc_emb)
            cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)

            vae_emb = self.embeddings[1].repeat(batch_size, 1, 1, 1)
            vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)
            
            # noise_pred = self.unet(
            #     torch.cat([x_in, vae_emb], dim=1),
            #     t_in.to(self.unet.dtype),
            #     encoder_hidden_states=cc_emb,
            # ).sample

            skip = None
            noise_pred,skip = self.unet(
                torch.cat([x_in, vae_emb], dim=1), #渲染图 + 主图的VAE
                t_in.to(self.unet.dtype),
                encoder_hidden_states=cc_emb,
            )#.sample

            # noise_pred,skip = self.unet(
            #     torch.cat([x_in, vae_emb], dim=1), #渲染图 + 主图的VAE
            #     t_in.to(self.unet.dtype),
            #     encoder_hidden_states=cc_emb,
            #     skip = skip,
            # )




            
            noise_pred = noise_pred.sample
            
            
#             noise_pred = 0
#             sa = 10
#             for i in range(sa):
#                 noise_pred  = noise_pred + self.unet(
#                     torch.cat([x_in, vae_emb], dim=1),
#                     t_in.to(self.unet.dtype),
#                     encoder_hidden_states=cc_emb,
#                 )[0][0]
             
#         noise_pred = noise_pred/sa
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum')
     

        return loss
    

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs, mode=False):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        if mode:
            latents = posterior.mode()
        else:
            latents = posterior.sample() 
        latents = latents * self.vae.config.scaling_factor

        return latents
    
    
if __name__ == '__main__':
    import cv2
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt
    import kiui

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default="guidance/499.png")
    parser.add_argument('--elevation', type=float, default=0, help='delta elevation angle in [-90, 90]')
    parser.add_argument('--azimuth', type=float, default=0, help='delta azimuth angle in [-180, 180]')
    parser.add_argument('--radius', type=float, default=0, help='delta camera radius multiplier in [-0.5, 0.5]')
    parser.add_argument('--stable', action='store_true', default=True)

    opt = parser.parse_args()

    device = torch.device('cuda')

    print(f'[INFO] loading image from {opt.input} ...')
    image = kiui.read_image(opt.input, mode='tensor')
    image = image.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    image = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)

    print(f'[INFO] loading model ...')
    
    if opt.stable:
        zero123 = Zero123(device, model_key='/root/.cache/huggingface/hub/models--ashawkey--stable-zero123-diffusers/snapshots/a6c09344fbd45843d35f2095c203c4a8e55b4dbb')
    else:
        zero123 = Zero123(device, model_key='/root/.cache/huggingface/hub/models--ashawkey--zero123-xl-diffusers/snapshots/10e9e8e75adaa3bf77f5ac9c381419016fed68c7')

    print(f'[INFO] running model ...')
    zero123.get_img_embeds(image)
    

    azimuth = opt.azimuth
    # while True:
    outputs = zero123.refine(image, elevation=[opt.elevation], azimuth=[azimuth], radius=[opt.radius], strength=0)
    transform = transforms.ToPILImage()
    image1 = transform(outputs.squeeze(0))
    image1.save("guidance/output3.png")
        
        # plt.imshow(outputs.float().cpu().numpy().transpose(0, 2, 3, 1)[0])
        # plt.show()
    # azimuth = (azimuth + 10) % 360