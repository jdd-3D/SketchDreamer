import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, CLIPImageProcessor
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()

class ReferenceEncoder(nn.Module):
    def __init__(self, model_path="openai/clip-vit-large-patch14"):
        super(ReferenceEncoder, self).__init__()
        self.model = CLIPVisionModel.from_pretrained(model_path,local_files_only=True)
        self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state
    
class ReferenceEncoder2(nn.Module):
    def __init__(self, model_path="openai/clip-vit-large-patch14"):
        super(ReferenceEncoder2, self).__init__()
        self.model = CLIPVisionModel.from_pretrained(model_path,local_files_only=True)
        self.processor = CLIPProcessor.from_pretrained(model_path,local_files_only=True)
        self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        
        # print(inputs['pixel_values'].size())
        
        outputs = self.model(**inputs, output_hidden_states=True)
        # print(outputs['last_hidden_state'].shape)
        # print(outputs['pooler_output'].shape)
        # print(outputs.hidden_states[-2].shape)
        # print(torch.cat((outputs['last_hidden_state'],outputs.hidden_states[-2]),dim=2).shape)
        # print(outputs.hidden_states[-2].shape)
        hidden_states=outputs.hidden_states[-2]
        b,h,w = hidden_states.shape
        if b>1:
            hidden_states=hidden_states.view(1, b * h, w)
        img_clip = torch.cat((hidden_states,hidden_states),dim=2)
        neg_clip = torch.zeros_like(img_clip)

        # print(torch.cat([neg_clip , img_clip]).shape)
        # print(outputs.keys())

        # pooled_output = outputs.pooler_output

        return torch.cat([neg_clip , img_clip])

# # example
# model = ReferenceEncoder2(model_path='/root/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41')
# image_path = "/root/sd/img/rabbit_chinese_rgba.png"
# image = Image.open(image_path).convert('RGB')
# image = [image]

# pooled_output = model(image)


# from PIL import Image
# import torch
# import torchvision.transforms as transforms
# import os


# def split_and_save_patches(image_path, patch_size, save_dir):
#     # 1. 读取图片并处理 RGBA 通道（若存在）
#     pil_image = Image.open(image_path)
    
#     # 检查图像模式，若为 RGBA 则转换为 RGB（避免透明通道影响）
#     if pil_image.mode == "RGBA":
#         pil_image = pil_image.convert("RGB")  # 丢弃 alpha 通道，转换为 RGB
    
#     # 2. 分割图片为 Patch
#     transform = transforms.ToTensor()
#     tensor_image = transform(pil_image)
#     _, height, width = tensor_image.shape
    
#     # 验证尺寸是否可整除
#     assert height % patch_size == 0 and width % patch_size == 0, \
#         f"图片尺寸 {height}x{width} 无法被 Patch 大小 {patch_size} 整除"
    
#     # 分割逻辑（与 ViT 一致：通道优先，展开为 (H/patch, W/patch, C, patch, patch)）
#     patches = tensor_image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
#     patches = patches.permute(1, 2, 0, 3, 4).contiguous()  # 调整维度为 (H/patch, W/patch, C, patch, patch)
#     patches = patches.view(-1, 3, patch_size, patch_size)[10:]  # 展平为 (总 Patch 数, C, H, W)
    

    
#     # # 3. 保存 Patch 到本地
#     # if not os.path.exists(save_dir):
#     #     os.makedirs(save_dir)
    
#     # to_pil = transforms.ToPILImage()
#     # for idx, patch in enumerate(patches):
#     #     patch_img = to_pil(patch)
#     #     save_path = os.path.join(save_dir, f"patch_{idx:03d}.png")
#     #     patch_img.save(save_path)
    
#     # print(f"成功保存 {patches.shape[0]} 个 Patch 到 {save_dir}")


# # 配置参数
# IMAGE_PATH = "img/rabbit_chinese_rgba.png"  # 你的图片路径
# PATCH_SIZE = 64  # 单个 Patch 尺寸（如 ViT-B/16 为 16x16）
# SAVE_DIR = "rabbit_patches"  # 保存目录

# # 执行分割和保存
# split_and_save_patches(IMAGE_PATH, PATCH_SIZE, SAVE_DIR)

