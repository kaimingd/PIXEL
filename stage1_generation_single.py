import stage1_generation_files_single.config as config
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os
import time
from pytorch_lightning import seed_everything
from stage1_generation_files_single.cldm.util import resize_image, HWC3
from stage1_generation_files_single.cldm.hed import HEDdetector
from stage1_generation_files_single.cldm.model import create_model, load_state_dict
from stage1_generation_files_single.cldm.ddim_hacked import DDIMSampler


apply_hed = HEDdetector()

model = create_model('./stage1_generation_files_single/mimic_models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('stage1_generation_files_single/mimic_models/epoch=3-step=40387.ckpt', location='cuda'),strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

label = ["nofinding", "atelectasis", "cardiomegaly","consolidation","edema","enlargedcardiomediastinum","fracture","lunglesion",
        "lungopacity", "pleuraleffusion","pleuralother", "pneumothorax","pneumonia","supportdevices"]

strength_list = [0.5,0.35,0.5,0.35,0.5,0.25,0.3,0.35,0.25,0.5,0.3,0.5,0.35,0.35]  

ref_path = "stage1_generation_files_single/reference_data"
out_dir = "stage1_generation_files_single/results/adapter_chestxray_paired_mimic1.8"
out_dir2 = "stage1_generation_files_single/results/adapter_chestxray_paired_mimic1.8_minus1"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)    
if not os.path.exists(out_dir2):
    os.makedirs(out_dir2)


a_prompt = ""
n_prompt = ""   
num_samples = 1
image_resolution = 512
detect_resolution = 512
ddim_steps = 150
guess_mode = False
scale = 9.0
eta = 0
        
with torch.no_grad():
    for j in range(300):
        for i,label_item in enumerate(label):
            seed = 300000 + j
            cond_name = ref_path + "/image_reference_"+str(seed)+"_ske.jpg"
            input_image = cv2.imread(cond_name)
            prompt = "a xray with {}".format(label_item)
            strength = strength_list[i]
            input_image = HWC3(input_image)
            detected_map = apply_hed(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=True)

            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            
            samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            x_samples = model.decode_first_stage(samples)


            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            img_dir = "image_"+label_item+"_"+str(seed)+".jpg"
            img_path = os.path.join(out_dir,img_dir)
            cv2.imwrite(img_path,x_samples[0])
            
            
            ref_img_path = os.path.join(out_dir,"image_"+"nofinding"+"_"+str(seed)+".jpg")
            ref_img = cv2.imread(ref_img_path).astype('float32') / 255
            ref_mask_path = os.path.join(ref_path, "image_"+"reference"+"_"+str(seed)+"_seg"+".jpg")
            ref_mask =cv2.imread(ref_mask_path).astype('float32') / 255
            exm_img = x_samples[0].astype('float32') / 255
            
            del_img = (abs(ref_img-exm_img))**1
            del_img = ((ref_mask*del_img)*255).astype('uint8')
            
            out_mask_dir = os.path.join(out_dir2,"image_"+label_item+"_"+str(seed)+".jpg")
            cv2.imwrite(out_mask_dir,del_img)
            print("image_"+label_item+"_"+str(seed)+".jpg")

