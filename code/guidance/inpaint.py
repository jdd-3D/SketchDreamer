# diffusers                      0.21.4  0.30.0
from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline, AutoencoderKL,UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import numpy as np
import cv2
import PIL.Image
from PIL import Image, ImageFilter , ImageChops
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation
from guidance.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from guidance.style_encoder import ReferenceEncoder,ReferenceEncoder2
import torch.nn.functional as F
import math
from torchvision.transforms import ToTensor, ToPILImage
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler




# from controlnet_aux import HEDdetector, CannyDetector,PidiNetDetector

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def remove_bg(image, net, transform, device):
    image_size = image.size
    input_images = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = net(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image

def split_patches(pil_image, patch_size=64):
    
    # 检查图像模式，若为 RGBA 则转换为 RGB（避免透明通道影响）
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")  # 丢弃 alpha 通道，转换为 RGB
    
    # 2. 分割图片为 Patch
    transform = transforms.ToTensor()
    tensor_image = transform(pil_image)
    _, height, width = tensor_image.shape
    
    # 验证尺寸是否可整除
    assert height % patch_size == 0 and width % patch_size == 0, \
        f"图片尺寸 {height}x{width} 无法被 Patch 大小 {patch_size} 整除"
    
    # 分割逻辑（与 ViT 一致：通道优先，展开为 (H/patch, W/patch, C, patch, patch)）
    patches = tensor_image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()  # 调整维度为 (H/patch, W/patch, C, patch, patch)
    patches = patches.view(-1, 3, patch_size, patch_size)#[10:]  # 展平为 (总 Patch 数, C, H, W)
    return patches

def gaussian_kernel(kernel_size, sigma):
    kernel = torch.tensor([math.exp(-(x - kernel_size // 2) ** 2 / (2 * sigma ** 2))
                           for x in range(kernel_size)], dtype=torch.float32)
    kernel = kernel / kernel.sum()
    kernel_2d = torch.outer(kernel, kernel)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    kernel_2d = kernel_2d.repeat(3, 1, 1, 1)
    return kernel_2d


# 1. 高通滤波 (加强高频细节)
def high_pass_filter(style_image):
    high_pass_kernel = torch.tensor([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
    high_pass_image = F.conv2d(style_image, high_pass_kernel, padding=1, groups=3)
    high_pass_image = style_image + high_pass_image  # 叠加原图
    return torch.clamp(high_pass_image, 0, 1)

def high_pass_filter2(style_image):
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


# 2. 高斯模糊（去噪）
def gaussian_blur(style_image, kernel_size=3, sigma=1):
    gaussian_kernel_tensor = gaussian_kernel(kernel_size, sigma)
    blurred_image=F.conv2d(style_image, gaussian_kernel_tensor, padding=kernel_size//2, groups=3)
    diff = style_image - blurred_image
    enhanced_image = style_image + diff

    return torch.clamp(enhanced_image, 0, 1)


# 3. 锐化滤波（拉普拉斯算子）
def sharpen_image(style_image):
    laplacian_kernel = torch.tensor([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
    sharpened_image = F.conv2d(style_image, laplacian_kernel, padding=1, groups=3)
    sharpened_image = style_image + sharpened_image  # 叠加原图
    return torch.clamp(sharpened_image, 0, 1)


# 主流程：高通 → 高斯模糊 → 锐化
def enhanced_workflow(pil_image):
    # 转换为张量：[H, W, C] → [1, C, H, W]
    
    to_tensor = ToTensor()
    tensor = to_tensor(pil_image).unsqueeze(0)
    
    
    # 1. 高通滤波增强高频纹理
    high_pass_result = high_pass_filter(tensor)
    
    # 2. 高斯模糊去噪（可调整kernel_size和sigma）
    blurred_result = gaussian_blur(high_pass_result, kernel_size=3, sigma=1)  # 小sigma保留细节
    
    # 3. 锐化
    final_result = sharpen_image(blurred_result)
    
    # 转换回PIL图像
    to_pil = ToPILImage()
    return to_pil(final_result.squeeze(0))

# 全局模型缓存
models_cache = {}

def init_models(device="cuda" if torch.cuda.is_available() else "cpu"):
    """初始化并加载所有需要的模型，返回模型字典"""
    global models_cache
    
    # 如果已经初始化过，直接返回缓存
    if models_cache and all(key in models_cache for key in [
        "birefnet", "clip_image_encoder", "controlnet", "vae", 
        "unet_encoder", "pipe"
    ]):
        return models_cache
    
    # 模型路径配置
    model_paths = {
        "birefnet": "ZhengPeng7/BiRefNet",
        "controlnet": "/root/autodl-tmp/models--xinsir--controlnet-canny-sdxl-1.0/snapshots/1271357eda52d54b857c650cacb5b51144643ccb",
        "vae": "/root/autodl-tmp/models--madebyollin--sdxl-vae-fp16-fix/snapshots/207b116dae70ace3637169f1ddd2434b91b3a8cd",
        "unet": "/root/autodl-tmp/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b",
        "scheduler": "/root/autodl-tmp/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b",
        "ip_adapter": "/root/autodl-tmp/models--h94--IP-Adapter/snapshots/018e402774aeeddd60609b4ecdb7e298259dc729",
    }
    
    # 加载边缘检测模型
    birefnet = AutoModelForImageSegmentation.from_pretrained(
        model_paths["birefnet"], trust_remote_code=True
    ).to(device)
    
    # 加载 CLIP 图像编码器 (假设 ReferenceEncoder2 是自定义类)
    clip_image_encoder = ReferenceEncoder2(
        model_path='/root/autodl-tmp/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
    )
    
    # 加载 ControlNet
    controlnet = ControlNetModel.from_pretrained(
        model_paths["controlnet"], torch_dtype=torch.float16
    )
    
    # 加载 VAE
    vae = AutoencoderKL.from_pretrained(
        model_paths["vae"], torch_dtype=torch.float16
    )
    
    # 加载 UNet
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        model_paths["unet"], subfolder="unet", torch_dtype=torch.float16
    )
    
    # 加载调度器
    scheduler = DDIMScheduler.from_pretrained(
        model_paths["scheduler"], subfolder="scheduler"
    )
    
    # 加载扩散模型管线
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        model_paths["unet"],
        controlnet=controlnet,
        vae=vae,
        unet_encoder=unet_encoder,
        torch_dtype=torch.float16,
        add_watermarker=False,
        scheduler=scheduler,
    ).to(device)
    
    # 加载 IP-Adapter
    pipe.load_ip_adapter(
        model_paths["ip_adapter"], 
        subfolder="sdxl_models", 
        weight_name="ip-adapter-plus_sdxl_vit-h.bin", 
        image_encoder_folder="image_encoder2"
    )
    
    # 设置 IP-Adapter 缩放
    scale = {
        "down": {"block_2": [0.0, 0.0]},
        "up": {"block_0": [0.0, 1.0, 0.0]},
    }
    pipe.set_ip_adapter_scale(scale)
    pipe.set_progress_bar_config(disable=True)
    
    # 启用模型 CPU 卸载以节省内存
    # pipe.enable_model_cpu_offload()
    
    # 存储到缓存
    models_cache = {
        "birefnet": birefnet,
        "clip_image_encoder": clip_image_encoder,
        "controlnet": controlnet,
        "vae": vae,
        "unet_encoder": unet_encoder,
        "pipe": pipe,
        "device": device,
    }
    
    return models_cache

def refine_imgs(direction="",input=None, models=None,strength=0.5):
    """使用预加载的模型处理图像"""
    # 使用提供的模型或初始化新模型
    if models is None:
        models = init_models()
    
    # 定义提示和负提示 
 
    prompt = direction+"beautiful, cute, VAR2, colormap, muted realistic colors, detailed texture,realistic, natural, cute, artistic, detailed, realistic portrait, high quality, 4k resolution"
   
    negative_prompt = 'text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry, outline '
    
    device = models["device"]
    birefnet = models["birefnet"]
    clip_image_encoder = models["clip_image_encoder"]
    pipe = models["pipe"]
    
    # 加载边缘图
    content_image = input.copy()  # 测试图像
    
    # 定义图像转换
    transform_image = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # 背景处理和边缘检测
    mask_image = remove_bg(content_image, birefnet, transform_image, device).convert('L').convert('RGB')
    mask_image = ImageChops.invert(content_image)
    
    # 加载内容图和风格图
    image = input.copy()
    style_image = load_image("data/cup73_rgba.png")  # 需确保 load_image 函数存在or input=data/zelda_rgba.png


    
    # 风格图增强处理
    to_tensor = ToTensor()
    tensor_image = to_tensor(style_image).unsqueeze(0)
    
    enhanced_image = high_pass_filter2(tensor_image)
    to_pil = ToPILImage()
    style_image = to_pil(enhanced_image.squeeze(0))
    # style_patchs = split_patches(style_image)
    
    # 设置随机种子
    seed = 27
    seed_everything(seed)
    
    # 编码风格图像
    style_embeeding = clip_image_encoder(style_image)
    # style_patchs_embeeding = clip_image_encoder(style_patchs)
    # style_embeeding = torch.cat((style_embeeding, style_patchs_embeeding), dim=1)
    
    # 对内容图进行边缘检测
    content_image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    content_image = np.array(content_image)
    content_image = cv2.Canny(content_image, 50, 200)
    canny_map = Image.fromarray(cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB))
    # canny_map.save(f"edge_image59.png")
 
    # 生成带有风格的图像
    images = pipe(
        prompt,
        negative_prompt=negative_prompt,
        image=image,
        control_image=canny_map,
        mask_image=mask_image,
        num_inference_steps=30,
        guidance_scale=5,
        controlnet_conditioning_scale=0.8,
        ip_adapter_image=[style_image],
        style_image=style_image,
        style_embeeding=style_embeeding,
        strength= 0.6
        #strength
        # style_patchs=None,
    ).images
    
    return images[0]



# def refine_imgs(input=None):
#     # 定义提示和负提示 
#     prompt = "beautiful, cute, VAR2, colormap, muted realistic colors, detailed texture,realistic, natural, cute, artistic, detailed, realistic portrait, high quality, 4k resolution"#"3D Render Style, 3DRenderAF, smooth skin, masterpiece, best quality, high quality,8k resolution, best quality"# transparent None background  
#     negative_prompt = 'text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry, outline ' #
#     #"highly detailed, ultra realistic, 8k resolution, masterpiece, best quality,sharp focus, intricate details, ultra-detailed,denoising:0.4, strength:0.5-0.7,VAR2, colormap, muted realistic colors, detailed texture,realistic, natural, cute, artistic, detailed, realistic portrait"
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     # 加载边缘图
#     content_image = input.copy()  #test  /out59.png    rabbit.png
#     birefnet = AutoModelForImageSegmentation.from_pretrained(
#         "ZhengPeng7/BiRefNet", trust_remote_code=True
#     )
#     birefnet.to(device)
#     transform_image = transforms.Compose(
#         [
#             transforms.Resize((512, 512)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ]
#     )
#     mask_image = remove_bg(content_image, birefnet, transform_image, device).convert('L').convert('RGB') #转换为灰度图
#     mask_image = ImageChops.invert(content_image)
#     # mask_image.save(f"content_image.png")



#     # 加载内容图
#     image = input.copy()  # 渲染图 6002
#     # 加载风格图
#     style_image = load_image("data/or.png")  # 替换为你的风格图路径  0000.png   1.png
#     # 进行图像增强处理
#     to_tensor = ToTensor()
#     tensor_image = to_tensor(style_image).unsqueeze(0)
#     enhanced_image = high_pass_filter2(tensor_image)
#     to_pil = ToPILImage()
#     style_image = to_pil(enhanced_image.squeeze(0))
#     style_patchs = split_patches(style_image)
#     # print(style_patchs.shape) #torch.Size([6, 3, 64, 64])


#     seed  = 27 #-1
#     # 设置随机种子
#     seed_everything(seed) #1655793223
#     clip_image_encoder = ReferenceEncoder2(model_path='/root/autodl-tmp/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41')

#     style_embeeding = clip_image_encoder(style_image)
#     style_patchs_embeeding = clip_image_encoder(style_patchs)

#     style_embeeding = torch.cat((style_embeeding,style_patchs_embeeding),dim=1)

#     # print(style_patchs_embeeding.shape) #torch.Size([2, 1542, 2048])
#     # print(style_embeeding.shape) #torch.Size([2, 1799, 2048])


#     # 控制网络条件缩放
#     controlnet_conditioning_scale = 0.6
#     # 加载控制网络模型
#     # controlnet = ControlNetModel.from_pretrained(
#     #     "/root/autodl-tmp/models--diffusers--controlnet-canny-sdxl-1.0/snapshots/eb115a19a10d14909256db740ed109532ab1483c",
#     #     torch_dtype=torch.float16
#     # )


#     controlnet = ControlNetModel.from_pretrained(
#         "/root/autodl-tmp/models--xinsir--controlnet-canny-sdxl-1.0/snapshots/1271357eda52d54b857c650cacb5b51144643ccb",
#         torch_dtype=torch.float16
#     )

#     # controlnet = ControlNetModel.from_pretrained(
#     #     "xinsir/controlnet-scribble-sdxl-1.0",
#     #     torch_dtype=torch.float16
#     # )
#     # eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("/root/autodl-tmp/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b", subfolder="scheduler")
#     eulera_scheduler = DDIMScheduler.from_pretrained("/root/autodl-tmp/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b", subfolder="scheduler")
#     # 加载 VAE
#     vae = AutoencoderKL.from_pretrained("/root/autodl-tmp/models--madebyollin--sdxl-vae-fp16-fix/snapshots/207b116dae70ace3637169f1ddd2434b91b3a8cd", torch_dtype=torch.float16)
#     unet_encoder = UNet2DConditionModel_ref.from_pretrained(
#         "/root/autodl-tmp/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b",
#         subfolder="unet",
#         torch_dtype=torch.float16
#     )
#     # 加载扩散模型管线
#     pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
#         "/root/autodl-tmp/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b",
#         controlnet=controlnet,
#         vae=vae,
#         unet_encoder = unet_encoder,
#         torch_dtype=torch.float16,
#         add_watermarker=False,
#         scheduler=eulera_scheduler,
#     )

#     # 加载 IP-Adapter
#     pipe.load_ip_adapter("/root/.cache/huggingface/hub/models--h94--IP-Adapter/snapshots/018e402774aeeddd60609b4ecdb7e298259dc729", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.bin", image_encoder_folder="image_encoder2")#ip-adapter-plus_sdxl_vit-h.bin 
#     # pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin", image_encoder_folder="image_encoder")

#     scale = {
#         "down": {"block_2": [0.0, 0.0]},
#         "up": {"block_0": [0.0, 1.0, 0.0]},
#     }
#     pipe.set_ip_adapter_scale(scale)


#     # 启用模型 CPU 卸载以节省内存
#     pipe.enable_model_cpu_offload()




#     # 对内容图进行边缘检测
#     content_image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
#     content_image = np.array(content_image)
#     content_image = cv2.Canny(content_image, 50, 200) #100, 200
#     canny_map = Image.fromarray(cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB))
#     # canny_map.save(f"edge_image59.png")



#     # generator = torch.Generator(pipe.device).manual_seed(seed)
#     # 生成带有风格的图像
#     # print(image,canny_map,mask_image)
#     # image.save(f"edge_image61.png")
#     # mask_image.save(f"edge_image62.png")

#     images = pipe(
#         prompt,
#         negative_prompt=negative_prompt,
#         # generator=generator,
#         image=image,
#         control_image=canny_map,
#         mask_image = mask_image,
#         num_inference_steps = 10,
#         guidance_scale = 5,
#         controlnet_conditioning_scale=controlnet_conditioning_scale,
#         ip_adapter_image=[style_image],  # 添加风格图
#         style_image = style_image,
#         style_embeeding = style_embeeding ,
#         style_patchs =None #style_patchs
#     ).images
#     return images[0]



    # images[0] = remove_bg(images[0], birefnet, transform_image, device)
    # 保存生成的图像    
    # print(images.shape) #torch.Size([1, 3, 256, 256])

    # images[0].save(f"paint4.png") # 