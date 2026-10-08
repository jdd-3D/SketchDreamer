# diffusers                      0.21.4  0.30.0
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL,UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation
from guidance.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from guidance.style_encoder import ReferenceEncoder2
import torch.nn.functional as F
import math
from torchvision.transforms import ToTensor, ToPILImage
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
# from controlnet_aux import HEDdetector, CannyDetector,PidiNetDetector
# from controlnet_aux import MidasDetector, ZoeDetector


# processor_zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
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
    print(tensor_image.shape)
    
    # 验证尺寸是否可整除
    assert height % patch_size == 0 and width % patch_size == 0, \
        f"图片尺寸 {height}x{width} 无法被 Patch 大小 {patch_size} 整除"
    
    # 分割逻辑（与 ViT 一致：通道优先，展开为 (H/patch, W/patch, C, patch, patch)）
    patches = tensor_image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()  # 调整维度为 (H/patch, W/patch, C, patch, patch)
    # print(patches.view(-1, 3, patch_size, patch_size).shape) #torch.Size([16, 3, 64, 64])
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
def refine_imgs(input=None):
    # 定义提示和负提示 
    prompt = "VAR2, colormap, muted realistic colors, detailed texture,realistic, natural, cute, artistic, detailed, realistic portrait, high quality, 4k resolution"# 3D Render Style, 3DRenderAF, smooth skin, masterpiece, best quality, high quality transparent None background  
    negative_prompt = 'text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry,outline ' #

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载边缘图
    content_images = input#test  600_z.png
    birefnet = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
    birefnet.to(device)
    transform_image = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # content_image = remove_bg(content_image, birefnet, transform_image, device)
    # content_image.save(f"content_image.png")

    # 加载风格图

    style_image = load_image("data/or_image.png")

    # style_image = remove_bg(style_image, birefnet, transform_image, device)#.convert("RGB")
    # style_image.save("img/or_image.png")


    # 进行图像增强处理
    to_tensor = ToTensor()
    tensor_image = to_tensor(style_image).unsqueeze(0)
    enhanced_image = high_pass_filter2(tensor_image)

    to_pil = ToPILImage()

    style_image = to_pil(enhanced_image.squeeze(0))




    # 524
    style_patchs = split_patches(style_image)






    seed  = 27 #-1 27
    # 设置随机种子
    seed_everything(seed) #1655793223
    clip_image_encoder = ReferenceEncoder2(model_path='/root/autodl-tmp/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41')
    style_embeeding = clip_image_encoder(style_image)
    # style_patchs_embeeding = clip_image_encoder(style_patchs)

    # style_embeeding = torch.cat((style_embeeding,style_patchs_embeeding),dim=1)



    # 控制网络条件缩放
    controlnet_conditioning_scale = 0.65
    # 加载控制网络模型
    controlnet = ControlNetModel.from_pretrained(
        "xinsir/controlnet-scribble-sdxl-1.0",
        torch_dtype=torch.float16
    )
    # controlnet = ControlNetModel.from_pretrained(
    #     "xinsir/controlnet-depth-sdxl-1.0",
    #     torch_dtype=torch.float16
    # )

    # eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("/root/autodl-tmp/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b", subfolder="scheduler")
    # eulera_scheduler = DDIMScheduler.from_pretrained("/root/autodl-tmp/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b", subfolder="scheduler")
    # 加载 VAE
    vae = AutoencoderKL.from_pretrained("/root/autodl-tmp/models--madebyollin--sdxl-vae-fp16-fix/snapshots/207b116dae70ace3637169f1ddd2434b91b3a8cd", torch_dtype=torch.float16)


    # vae = AutoencoderKL.from_pretrained("nubby/blessed-sdxl-vae-fp16-fix", subfolder="vae", weight_name="sdxl_vae-fp16fix-blessed.safetensors", torch_dtype=torch.float16) #madebyollin/sdxl-vae-fp16-fix
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        "/root/autodl-tmp/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b",
        subfolder="unet",
        torch_dtype=torch.float16
    )
    # 加载扩散模型管线
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "/root/autodl-tmp/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b",
        controlnet=controlnet,
        vae=vae,
        unet_encoder = unet_encoder,
        torch_dtype=torch.float16,
        add_watermarker=False,
        # scheduler=eulera_scheduler,
    )

    # 加载 IP-Adapter     sdxl_models     ip-adapter_sdxl_vit-h.bin   ip-adapter-plus_sdxl_vit-h.bin
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin", image_encoder_folder="image_encoder") #ip-adapter_sdxl.bin
    # ,"ip-adapter-plus_sdxl_vit-h.bin","ip-adapter-plus_sdxl_vit-h.bin"



    scale1 = {
        "down": {"block_2": [0.0, 1]}, # layout
        "up": {"block_0": [0.0, 1, 0.0]}, # style
    }
    pipe.set_ip_adapter_scale([scale1])
    pipe.enable_model_cpu_offload()
    transform = transforms.ToPILImage()
    print("content_image",content_images.shape)
  
    controlnet_images = []
    content_imgs = []


    # 遍历批次中的每张图像
    for i in range(content_images.shape[0]):
        # 提取单张图像 [3, 512, 512]
        single_image = content_images[i]
        
        # 转换为PIL图像
        pil_image = transform(single_image)
        content_image = pil_image #
        content_imgs.append(content_image)
        
        # 应用边缘增强滤镜
        enhanced_img = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        # 加载内容图
        
        
        # 转换为NumPy数组用于OpenCV处理
        img_np = np.array(enhanced_img)
        
        # 确保图像是灰度图（Canny需要单通道输入）
        if len(img_np.shape) == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_np
        
        # 应用Canny边缘检测
        edges = cv2.Canny(img_gray, 50, 200)
        # 转换回RGB以便保存
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges_pil = Image.fromarray(edges_rgb)
        # 保存单张图像
        edges_pil.save(f"edge_image_{i}.png")
        # 可选：将处理后的图像添加到列表中
        controlnet_images.append(edges_pil)



    # pil_image = transform(content_image)
    
    # controlnet_img  = content_image.filter(ImageFilter.EDGE_ENHANCE_MORE)

    # controlnet_img  = np.array(controlnet_img )
    # controlnet_img  = cv2.Canny(controlnet_img , 50, 200) #100, 200
    # controlnet_img = Image.fromarray(cv2.cvtColor(controlnet_img , cv2.COLOR_BGR2RGB))
    # controlnet_img.save(f"edge_image.png")



    generator = torch.Generator(pipe.device).manual_seed(seed)

    # 生成带有风格的图像
    width, height = style_image.size
    style_image_ip = style_image

    print("style_image_ip",style_image_ip)

    prompt = [prompt] * len(controlnet_images)
    negative_prompt = [negative_prompt] * len(controlnet_images)
    print("****",controlnet_images[0],content_imgs[0])


    images = pipe(
        prompt = prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        image=controlnet_images,
        content_image = content_imgs,
        num_inference_steps = 30,
        guidance_scale = 5,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        ip_adapter_image=[style_image_ip],  # 添加风格图   ,style_edge_ip
        style_image = None,#[style_image] * len(controlnet_images),
        style_embeeding = None,#[style_embeeding] * len(controlnet_images),
        style_patchs = None#style_patchs
    ).images





    print(images)
    i = 1
    for img in images:
        img = remove_bg(img, birefnet, transform_image, device) 
        img.save(f"test/"+str(i)+".png") # 
        i +=1
    return images[0]