from __future__ import print_function

import argparse
import os
import time, platform

import cv2
import torch.optim as optim
from torch.utils.data import DataLoader

from DexiNed.datasets import DATASET_NAMES, BipedDataset, TestDataset, dataset_info
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from DexiNed.model import DexiNed
from DexiNed.utils import (image_normalization, save_image_batch_to_disk,
                   visualize_result,count_parameters)

IS_LINUX = True if platform.system()=="Linux" else False

def test(checkpoint_path, dataloader, model, device, output_dir, args):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    # print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    # Put model in evaluation mode
    model.eval()

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            if not args.test_data == "CLASSIC":
                labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            # print(f"input tensor shape: {images.shape}")
            # images = images[:, [2, 1, 0], :, :]

            end = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            preds = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            tmp_duration = time.perf_counter() - end
            total_duration.append(tmp_duration)

            save_image_batch_to_disk(preds,
                                     output_dir,
                                     file_names,
                                     image_shape,
                                     arg=args)
            torch.cuda.empty_cache()

    total_duration = np.sum(np.array(total_duration))
    # print("******** Testing finished in", args.test_data, "dataset. *****")
    # print("FPS: %f.4" % (len(dataloader)/total_duration))

def testPich(checkpoint_path, image, model, device, output_dir, args):
    # a test model plus the interganged channels
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    # print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    # Put model in evaluation mode
    model.eval()

    with torch.no_grad():
        device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')
  
        # for batch_id, sample_batched in enumerate(dataloader):
        
        images = image.to(device)
        # images = image.permute(2, 0, 1).unsqueeze(0) .to(device)
        # print("images",images.shape) #torch.Size([1, 3, 512, 512])
        if images.shape[0] != 1:
            images = images.permute(2, 0, 1).unsqueeze(0)
        

        # images2 = images[:, [1, 0, 2], :, :]  #GBR
        # print(images.shape)
        images2 = images[:, [2, 1, 0], :, :] # RGB torch.Size([1, 3, 512, 512])
        # print(f"input tensor shape: {images2.shape}") 
        preds = model(images)
        preds2 = model(images2)
        image_shape=[]
        for pred in preds:
            image_shape.append(pred.shape) 
        # print(preds2.shape)
        file_names = "edge_pic/"
        fuse,average = save_image_batch_to_disk([preds,preds2],
                                    output_dir,
                                    file_names,
                                    image_shape,
                                    arg=args, is_inchannel=True)
        del images,images2,model,image_shape            
        torch.cuda.empty_cache()
        return fuse,average


@torch.no_grad()
def main1(input_path):
    """Main function."""
    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')
    # Instantiate model and move it to the computing device
    model = DexiNed().to(device)
    # Testing

    img_width,img_height = 512,512
    
    image = cv2.imread((input_path), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (img_width,img_height))
    image = torch.from_numpy(image.copy()).float()
    print(image.shape)
    output_dir = "result/"
    checkpoint_path = "DexiNed/checkpoints/BIPED/10/10_model.pth"
    fuse,average = testPich(checkpoint_path, image, model, device, output_dir, args=None)
 
    return fuse,average

@torch.no_grad()
def main2(input):
    """Main function."""
    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')
    # Instantiate model and move it to the computing device
    model = DexiNed().to(device)
    # Testing
    image = input
    # print(input.shape)
    # print(image)
    transform = transforms.ToPILImage()
    image1 = transform(input.squeeze(0))
    image1.save("mask/111TETST.png")

    image = cv2.imread(("mask/111TETST.png"), cv2.IMREAD_COLOR)
   
    image = torch.from_numpy(image.copy()).float()




    # print(image.shape)

    output_dir = "result/"
    checkpoint_path = "DexiNed/checkpoints/BIPED/10/10_model.pth"
    fuse,average = testPich(checkpoint_path, image, model, device, output_dir, args=None)
    # print(fuse)
    # pil_image = Image.fromarray(fuse)
    # pil_image.save('mask/111.png')
    # cv2.imwrite('mask/111.png', fuse.astype(np.uint8))
    # del model

    # torch.cuda.empty_cache()
    return fuse,average

if __name__ == '__main__':
   
    fuse,average = main1("/root/dr/mask/111TETST.png")
    print(fuse)
    pil_image = Image.fromarray(fuse)

    pil_image.save('output_pillow.png')

    pil_image = Image.fromarray(average)

    pil_image.save('output_pillow1.png')
