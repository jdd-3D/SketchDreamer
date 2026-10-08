import os
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg
from torchvision import transforms
from diffusers.utils import export_to_gif
from cam_utils import orbit_camera, OrbitCamera
import math
from gs_renderer import Renderer, MiniCam
import argparse
from omegaconf import OmegaConf


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="path to the yaml config file")
args, extras = parser.parse_known_args()

# override default config from cli
opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

# Allow `python see.py --config ...` immediately after training: infer the
# checkpoint and output directory when they are not explicitly provided.
if opt.load is None:
    opt.load = os.path.join(opt.outdir, opt.save_path + '_model.ply')
if not opt.get('workspace'):
    opt.workspace = opt.outdir
if not os.path.isfile(opt.load):
    raise SystemExit(f'Model file does not exist: {opt.load}')

renderer = Renderer(sh_degree=opt.sh_degree)
renderer.initialize(opt.load)
nframes = 120
hor = 180
delta_hor = 360 / nframes
image_list = []

# # Create a directory to save images if it doesn't exist
# output_dir = os.path.join(opt.workspace, "images")
# os.makedirs(output_dir, exist_ok=True)

cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
for idx in range(nframes):
    pose = orbit_camera(0, hor-180, opt.radius)
    cur_cam = MiniCam(
                pose,
                opt.W,
                opt.H,
                cam.fovy,
                cam.fovx,
                cam.near,
                cam.far,
            )
    out = renderer.render(cur_cam)
    hor = (hor + delta_hor) % 360

    buffer_image = out["image"]  # [3, H, W]
    buffer_image = F.interpolate(
        buffer_image.unsqueeze(0),
        size=(opt.H, opt.W),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    image_list.append(transforms.ToPILImage()(buffer_image))
    
    # # Save each frame image
    # frame_image = transforms.ToPILImage()(buffer_image)
    # frame_image.save(os.path.join(output_dir, f"frame_{idx:04d}.png"))
    
name = os.path.splitext(os.path.basename(opt.load))[0]
if name.endswith('_model'):
    name = name[:-len('_model')]

if not os.path.exists(opt.workspace):
    os.makedirs(opt.workspace)

# Export to GIF
gif_path = os.path.join(opt.workspace, name + '.gif')
export_to_gif(image_list, gif_path, fps=30)
print(f'[INFO] save GIF to {gif_path}')

# Export to MP4
video_filename = name + '.mp4'
video_path = os.path.join(opt.workspace, video_filename)
with imageio.get_writer(video_path, fps=30) as writer:  # fps可以根据需要调整
    for image in image_list:
        writer.append_data(np.array(image))
print(f'[INFO] save MP4 to {video_path}')

