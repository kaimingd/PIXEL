import torch
import torchvision
import os
import glob
import time 
import pickle
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from PIL import ImageDraw
from stage2_segmentation_files_single.src.data2 import LungDataset, blend, Pad, Crop, Resize
from stage2_segmentation_files_single.src.models import UNet, PretrainedUNet
from stage2_segmentation_files_single.src.metrics import jaccard, dice
from skimage.metrics import peak_signal_noise_ratio
from stage2_segmentation_files_single.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from stage2_segmentation_files_single.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

Root_dir = "stage2_segmentation_files_single"
add = "minus1"
dir_root = "stage1_generation_files_single/results/adapter_chestxray_paired_mimic1.8"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_folder = Path("input", "dataset")
origins_folder = Path(dir_root)
masks_folder = Path(dir_root+"_"+add)
task = "images_transunet_minus1"
images_folder = Path(os.path.join(Root_dir,"results",task))
   
def shorten_filename(filename, limit=180):
    """返回合适长度文件名，中间用...显示"""
    if len(filename) <= limit:
        return filename
    else:
        return filename[:int(limit / 2) - 3] + '...' + filename[len(filename) - int(limit / 2):]
   
def val_real_padchest():
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(512 / 16), int(512 / 16))
    unet = ViT_seg(config_vit, img_size=512, num_classes=config_vit.n_classes).cuda()
    model_name = "unet-6v-latest.pt"
    unet.load_state_dict(torch.load(images_folder / model_name, map_location=torch.device("cpu")))
    unet.to(device)
    unet.eval();

    img_root = "source_data/PadChest_Data/PadChest"
    csv_root = "source_data/PadChest_Data/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
    
    dict = pd.read_csv(csv_root,low_memory=False)
    name_list = dict["ImageID"].tolist()
    label_list = dict["Labels"].tolist()

    dataset_id = "padchest_sampledata"
    images_folder_gt = Path(str(images_folder),dataset_id,"gt")
    images_folder_img = Path(str(images_folder),dataset_id,"img")
    images_folder_pred = Path(str(images_folder),dataset_id,"pred")
    images_folder_mask = Path(str(images_folder),dataset_id,"mask")
    images_folder_mix = Path(str(images_folder),dataset_id,"mix")
    if not os.path.exists(images_folder_gt):
        os.makedirs(images_folder_gt)
    if not os.path.exists(images_folder_img):
        os.makedirs(images_folder_img)
    if not os.path.exists(images_folder_pred):
        os.makedirs(images_folder_pred)
    if not os.path.exists(images_folder_mask):
        os.makedirs(images_folder_mask)
    if not os.path.exists(images_folder_mix):
        os.makedirs(images_folder_mix)
        
    img_list = os.listdir(img_root)
    for i,img_item in enumerate(img_list):
        id = name_list.index(img_item.replace('jpg','png'))
        labels = label_list[id]
        img_dir = os.path.join(img_root,img_item)
        img_item = img_item.replace(".jpg","_"+labels+".jpg")
        img_item = shorten_filename(img_item)
        gt = Image.open(img_dir).convert("RGB")
        origin = gt.convert("P")
        origin = torchvision.transforms.functional.to_tensor(origin) - 0.5
        with torch.no_grad():
            origin = torch.stack([origin])
            origin = origin.to(device)
            outs = unet(origin)
            out = outs.float()
            pil_origin = torchvision.transforms.functional.to_pil_image(origin[0] + 0.5).convert("RGB")
            pil_mask_pred = torchvision.transforms.functional.to_pil_image(out[0] + 0.5).convert("RGB")
            
            pil_mask_pred = pil_mask_pred.convert("L")
            
            image_array = np.array(pil_mask_pred)

            white_mask = np.where(image_array >= 240)
            image_array[white_mask] = 0
            pil_mask_pred = Image.fromarray(image_array)
            
            pil_mask = pil_mask_pred.point(lambda p: p > 30 and 255)
            pil_mask_pred.save(images_folder_pred / img_item)
            pil_origin.save(images_folder_img / img_item)
            pil_mask.save(images_folder_mask / img_item)
            gt.save(images_folder_gt / img_item)
            
            
            plt.subplot(1, 3, 1)
            plt.title("origin image")
            plt.imshow(np.array(gt))

            plt.subplot(1, 3, 2)
            plt.title("manually location_map_pred")
            plt.imshow(np.array(pil_mask_pred))
            
            plt.subplot(1, 3, 3)
            plt.title("manually mask_pred")
            plt.imshow(np.array(pil_mask))
            
            
            plt.savefig(images_folder_mix / img_item, bbox_inches='tight')

        print(i)
    
if __name__ == "__main__":
    val_real_padchest()
