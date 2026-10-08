import os
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F
import random
import rembg

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize



from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
# from edge import main1 as get_edge1
from edge import main2 as get_edge
from controlnet_aux import MidasDetector, ZoeDetector, NormalBaeDetector
# import math
# from Restormer.demo1 import main as get_deblur
from PIL import Image, ImageFilter, ImageEnhance
# from controlnet_aux import AnylineDetector
from transformers import pipeline
from torchvision.transforms import ToTensor, ToPILImage
from diffusers.utils import load_image
import imageio
import lpips

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

def get_canny(image,processor=None,device=None):
    low_threshold = 100
    high_threshold = 200
    if isinstance(image, np.ndarray):
        pred_rgb = image
    else:
        pred_rgb = ((image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8).copy() )

    # pred_rgb = Image.fromarray(pred_rgb)
    # enhanced_img = pred_rgb.filter(ImageFilter.EDGE_ENHANCE_MORE)
    # pred_rgb = np.array(enhanced_img)

    image = cv2.Canny(pred_rgb, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny = Image.fromarray(image)
    
    return canny

def get_normal(image,processor=None,device=None):
    # image=high_pass_filter2(image.detach().cpu())#.to(device)
    # print(image.shape)
    pred_rgb = ((image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8).copy() )
    # pred_rgb = Image.fromarray(pred_rgb)
    # enhanced_img = pred_rgb.filter(ImageFilter.EDGE_ENHANCE_MORE)
    # pred_rgb = np.array(enhanced_img)
    normal = processor(pred_rgb, hand_and_face=False, output_type='cv2')
    normal = Image.fromarray(normal)
    return normal


def get_depth(image,processor=None,device=None):
##### 1
    image = processor(image)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)
    return control_image

##### 2

    # controlnet_img = processor(image, output_type='cv2')
    # controlnet_img = Image.fromarray(controlnet_img)
    # return controlnet_img



    
     

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        self.refine_img_0 = None
        self.refine_img_90 = None
        self.refine_img_180 = None
        self.refine_img_270 = None

        self.real_renders = None
        self.valid_mask2 = None
        self.input_nomal = None
        self.input_depth = None
        self.input_canny = None

        # self.anyline = AnylineDetector.from_pretrained(
        #     "TheMistoAI/MistoLine", filename="MTEED.pth", subfolder="Anyline"
        # ).to(self.device)

        self.processor_normal = NormalBaeDetector.from_pretrained("/root/.cache/huggingface/hub/models--lllyasviel--Annotators/snapshots/982e7edaec38759d914a963c48c4726685de7d96")

        self.depth_estimator = pipeline('depth-estimation')

        # self.processor_zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators",cache_dir="/root/autodl-tmp")
        # self.processor_midas = MidasDetector.from_pretrained("lllyasviel/Annotators",cache_dir="/root/autodl-tmp")
        
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt

        # override if provide a checkpoint
        if self.opt.load is not None:
            self.renderer.initialize(self.opt.load)            
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

        self.lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
        self.lpips_fn.eval()
    
        for p in self.lpips_fn.parameters():
            p.requires_grad = False

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # default camera
        if self.opt.mvdream or self.opt.imagedream:
            # the second view is the front view for mvdream/imagedream.
            pose = orbit_camera(self.opt.elevation, 90, self.opt.radius)
        else:
            pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            elif self.opt.imagedream:
                print(f"[INFO] loading ImageDream...")
                from guidance.imagedream_utils import ImageDream
                self.guidance_sd = ImageDream(self.device)
                print(f"[INFO] loaded ImageDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            if self.opt.stable_zero123: #
                #ashawkey/stable-zero123-diffusers
                self.guidance_zero123 = Zero123(self.device, model_key='/root/autodl-tmp/models--ashawkey--stable-zero123-diffusers/snapshots/a6c09344fbd45843d35f2095c203c4a8e55b4dbb')
                print(f"[INFO] loaded stable_zero123!")
            else:  
                self.guidance_zero123 = Zero123(self.device, model_key='/root/autodl-tmp/models--ashawkey--zero123-xl-diffusers/snapshots/10e9e8e75adaa3bf77f5ac9c381419016fed68c7')
                print(f"[INFO] loaded zero123!")

        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)
           
            # print(self.input_img_torch.shape)
            # target_size = (1024,1024)
            # input =F.interpolate(self.input_img_torch, size=target_size, mode='bilinear', align_corners=False)
            input = self.input_img_torch
            self.fuse,self.average = get_edge(input)
            self.average = torch.FloatTensor(self.average).to(self.device)

            
            # print(self.average.shape)
            # input = F.interpolate(self.input_img_torch.unsqueeze(0), size=(128,128), mode='bilinear', align_corners=False).squeeze(0)
            
            # print(input.shape)
            # self.best_edge = 10000

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)
            self.input_img_torch_channel_last = self.input_img_torch[0].contiguous()
            self.fourier_gt = torch.fft.fft2(self.input_img_torch, dim=(2, 3))
        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                if self.opt.imagedream:
                    self.guidance_sd.get_image_text_embeds(self.input_img_torch, [self.prompt], [self.negative_prompt])
                else:
                    self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        # print(self.input_img_torch.shape)
        

        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)
            # step_ratio = 2/(1 + np.exp(-5 * step_ratio))-1

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0
        
            ### known view
            if self.input_img_torch is not None and not self.opt.imagedream:
                transform = transforms.ToPILImage()
                
                cur_cam = self.fixed_cam
                out = self.renderer.render(cur_cam)

                # rgb loss
                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                loss = loss + 10000 * step_ratio *  F.mse_loss(image, self.input_img_torch)
                
                # mask loss  1000000
                mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
                loss = loss + 1000 *step_ratio *  F.mse_loss(mask, self.input_mask_torch)
           




            
            ### novel view (manual batch)
            # render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            render_resolution = 512 
            images = []
            masks = []
            poses = []
            vers, hors, radii,control_images,depth_images,canny_images = [], [], [], [], [], []
            # interval = 0
            if self.step == 1: # or self.step%200==0:

                
                if self.step< 200:
                    interval = 0
                else:
                    interval = random.randint(20, 30)
              
                horss = [0,40,80,120,160,200,240,280]
                horss = [0,40,90,120,180,200,240,280]
                for hor in [num + interval for num in horss]:
                    # render random view
                    ver = 0
                    radius = 0
                    vers.append(ver)
                    # print(hor)
                    hors.append(hor)
                    radii.append(radius)

                    pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                    poses.append(pose)

                    cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                    
                    bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                    # print(hor)
                    out = self.renderer.render(cur_cam, bg_color=bg_color)
                    image = out["image"]# [1, 3, H, W] in [0, 1]

                    canny_image = get_canny(image,processor=self.processor_normal,device=self.device)
                    
                    Normal_image = get_normal(image,processor=self.processor_normal,device=self.device)
                    
                    depth_image = get_depth(transform(image),processor=self.depth_estimator,device=self.device)
     
                    canny_images.append(canny_image)
                    control_images.append(Normal_image)
                    depth_images.append(depth_image)
                    
                    images.append(image.unsqueeze(0))
                    mask = out["alpha"].unsqueeze(0)
                    masks.append(mask)  
                        
                masks = torch.cat(masks, dim=0) 
                images = torch.cat(images, dim=0)
            else:
                min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
                max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)
                # print(self.opt.batch_size)
                for _ in range(self.opt.batch_size):
                    # render random view
                    ver = np.random.randint(min_ver, max_ver)
                    # ver = np.random.randint(-30, 30)
                    hor = np.random.randint(-180, 180)
                    radius = 0

                    vers.append(ver)
                    hors.append(hor)
                    radii.append(radius)

                    pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                    poses.append(pose)

                    cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                    
                    bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                    out = self.renderer.render(cur_cam, bg_color=bg_color)
                    image = out["image"].unsqueeze(0)# [1, 3, H, W] in [0, 1]
                    images.append(image)
                images = torch.cat(images, dim=0)
            
        

            
            if self.enable_sd and self.step >= 1:
                    global final_img
                    if self.step == 1: # or self.step % 200 == 0:
                        if self.step <= 200:
                            final_img = None
                            sd_loss, final_img = self.guidance_sd.train_step(images, vers=vers, hors=hors,step_ratio=step_ratio if self.opt.anneal_timestep else None, step = self.step,style_path=self.opt.style_path,control_images =control_images,masks=masks,input_nomal=self.input_nomal,depth_images=depth_images,canny_images=canny_images,input_depth=self.input_depth,input_canny=self.input_canny)
                            loss = loss + sd_loss 
    
                    else:
                        images = []
                        masks = []
                        poses = []
                        vers, hors, radii,control_images,depth_images, canny_images = [], [], [], [], [], []
                  
                        if self.step< 200:
                            interval = 0
                        else:
                            interval = 20
                            print("***********",interval) 
                        horss = [0,40,80,120,160,200,240,280]
                        horss = [0,40,90,120,180,200,240,280]
                
                        for hor in [num + interval for num in horss]:
                            edge_images = []
                            ver = 0
                            radius = 0
                            vers.append(ver)
                            hors.append(hor)
                            radii.append(radius)

                            pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                            poses.append(pose)

                            cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                            
                            bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                            # print(hor)
                            out = self.renderer.render(cur_cam, bg_color=bg_color)
                            image = out["image"]# [1, 3, H, W] in [0, 1]

                             
                            

                     
                            images.append(image.unsqueeze(0))
                            mask = out["alpha"].unsqueeze(0)
                            masks.append(mask)  
                                
                        masks = torch.cat(masks, dim=0)
                        images = torch.cat(images, dim=0)
                   
                        l1_loss = F.l1_loss(images[:8]*masks[:8], final_img[:8]*masks[:8])
                     
                        pred_lpips = images[:8] * 2.0 - 1.0
                        target_lpips = final_img[:8] * 2.0 - 1.0
                
                        lpips_loss = self.lpips_fn(pred_lpips, target_lpips).mean()
                
                        # total
                        loss_RGB = l1_loss + lpips_loss
                        # print("loss_recon",loss_recon)
                        loss = loss+10000*loss_RGB
                       
                     
            

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
    

            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
       
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
          
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
              
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if  self.step > 0 and self.step % self.opt.densification_interval == 0:
                
                    self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.05, extent=4, max_screen_size=1, step=self.step)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )

  

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            out = self.renderer.render(cur_cam, self.gaussain_scale_factor)

            buffer_image = out[self.mode]  # [3, H, W]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(3, 1, 1)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            # display input_image
            if self.overlay_input_img and self.input_img is not None:
                self.buffer_image = (
                    self.buffer_image * (1 - self.overlay_input_img_ratio)
                    + self.input_img * self.overlay_input_img_ratio
                )

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!

    
    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)
  
        if img.shape[0] != self.W: 
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        input_nomal = self.processor_normal(img, hand_and_face=False, output_type='cv2')
        self.input_nomal = Image.fromarray(input_nomal)
        self.input_nomal.save("muiltview/normal_image.png")
    
        from PIL import ImageOps
        self.input_canny = get_canny(img,processor=self.processor_normal,device=self.device)

        inverted_canny = ImageOps.invert(self.input_canny).filter(ImageFilter.EDGE_ENHANCE_MORE)

        inverted_canny.save("muiltview/edge_image.png")

        # self.input_canny.save("muiltview/edge_image.png")
        # image = load_image(file)
        self.input_depth = get_depth(file,processor=self.depth_estimator,device=self.device)
        self.input_depth.save("muiltview/depth_image.png")
        
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

        # load prompt
        file_prompt = file.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            print(f'[INFO] load prompt from {file_prompt}...')
            with open(file_prompt, "r") as f:
                self.prompt = f.read().strip()
                
    
    @torch.no_grad()
    def edge_loss(self, input_edge, image):
        H,W=input_edge.shape
        target_size = (H, W)
        image =F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)
        fuse,average_edge = get_edge(image)
        # print(average_edge.shape)
        transform = transforms.ToPILImage()
        # print(average_edge)
        image4 = transform(average_edge)
        image4.save("edge_pic/edge.png")
        average_edge= torch.FloatTensor(average_edge).to(self.device)
        loss = F.mse_loss(average_edge, input_edge)
        return loss
    

    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.ply')
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.' + self.opt.mesh_format)
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
                glctx = dr.RasterizeCudaContext()
            else:
                glctx = dr.RasterizeCudaContext()
                
            # if not self.opt.gui or os.name == 'nt':
            #     glctx = dr.RasterizeCudaContext()
            # else:
            #     glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                
                cur_out = self.renderer.render(cur_cam)

                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]

                # enhance texture quality with zero123 [not working well]
                # if self.opt.guidance_model == 'zero123':
                #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
                    # import kiui
                    # kiui.vis.plot_image(rgbs)
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")
                
                # overlay stuff
                with dpg.group(horizontal=True):

                    def callback_toggle_overlay_input_img(sender, app_data):
                        self.overlay_input_img = not self.overlay_input_img
                        self.need_update = True

                    dpg.add_checkbox(
                        label="overlay image",
                        default_value=self.overlay_input_img,
                        callback=callback_toggle_overlay_input_img,
                    )

                    def callback_set_overlay_input_img_ratio(sender, app_data):
                        self.overlay_input_img_ratio = app_data
                        self.need_update = True

                    dpg.add_slider_float(
                        label="ratio",
                        min_value=0,
                        max_value=1,
                        format="%.1f",
                        default_value=self.overlay_input_img_ratio,
                        callback=callback_set_overlay_input_img_ratio,
                    )

                # prompt stuff
            
                dpg.add_input_text(
                    label="prompt",
                    default_value=self.prompt,
                    callback=callback_setattr,
                    user_data="prompt",
                )

                dpg.add_input_text(
                    label="negative",
                    default_value=self.negative_prompt,
                    callback=callback_setattr,
                    user_data="negative_prompt",
                )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        self.save_model(mode=user_data)

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_button(
                        label="geo",
                        tag="_button_save_mesh",
                        callback=callback_save,
                        user_data='geo',
                    )
                    dpg.bind_item_theme("_button_save_mesh", theme_button)

                    dpg.add_button(
                        label="geo+tex",
                        tag="_button_save_mesh_with_tex",
                        callback=callback_save,
                        user_data='geo+tex',
                    )
                    dpg.bind_item_theme("_button_save_mesh_with_tex", theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    # dpg.add_button(
                    #     label="init", tag="_button_init", callback=self.prepare_train
                    # )
                    # dpg.bind_item_theme("_button_init", theme_button)

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

                def callback_set_gaussain_scale(sender, app_data):
                    self.gaussain_scale_factor = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label="gaussain scale",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                    default_value=self.gaussain_scale_factor,
                    callback=callback_set_gaussain_scale,
                )

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="Gaussian3D",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
            
            # do a last prune
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1) #0.01
      
        # Save   self.opt.save_path = "refine"
        self.save_model(mode='model')
        # self.save_model(mode='geo+tex')
      
        
        
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)

    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters)
