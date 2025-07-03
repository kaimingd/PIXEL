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
import pydicom
from skimage import exposure
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from PIL import ImageDraw
from stage2_segmentation_files_single.src.data2 import LungDataset, blend, Pad, Crop, Resize
from stage2_segmentation_files_single.src.models import UNet, PretrainedUNet
from stage2_segmentation_files_single.src.metrics import jaccard, dice
from pycocotools.coco import COCO
from skimage.metrics import peak_signal_noise_ratio
from stage2_segmentation_files_single.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from stage2_segmentation_files_single.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from stage2_segmentation_files_single.src.utils import score_cal

Root_dir = "stage2_segmentation_files_single"
add = "minus1"
dir_root = "stage1_generation_files_single/results/adapter_chestxray_paired_mimic1.8"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_folder = Path("input", "dataset")
origins_folder = Path(dir_root)
masks_folder = Path(dir_root+"_"+add)
task = "images_transunet_minus1"
images_folder = Path(os.path.join(Root_dir,"results",task))

def detect_and_draw_edges(binary_image):
    binary_image = cv2.cvtColor(np.asarray(binary_image),cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    _,binary    = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
                             
def val_real_chexlocalize():
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(512 / 16), int(512 / 16))
    unet = ViT_seg(config_vit, img_size=512, num_classes=config_vit.n_classes).cuda()
    model_name = "unet-6v-latest.pt"
    unet.load_state_dict(torch.load(images_folder / model_name, map_location=torch.device("cpu")))
    unet.to(device)
    unet.eval();
    
    image_root = "/hhd/2/dkm/chexlocalize/CheXpert_Image_combineresized_test"
    mask_root = "/hhd/2/dkm/chexlocalize/CheXpert_Label_combineresized_test"

    img_path_list = os.listdir(image_root)
    bbox_list = os.listdir(mask_root)

    dataset_id = "chexlocalize"
    images_folder_gt = Path(str(images_folder),dataset_id,"gt")
    images_folder_img = Path(str(images_folder),dataset_id,"img")
    images_folder_pred = Path(str(images_folder),dataset_id,"pred")
    images_folder_mask = Path(str(images_folder),dataset_id,"mask")
    images_folder_gtmask = Path(str(images_folder),dataset_id,"gtmask")
    images_folder_mix = Path(str(images_folder),dataset_id,"mix")
    
    if not os.path.exists(images_folder_gt):
        os.makedirs(images_folder_gt)
    if not os.path.exists(images_folder_img):
        os.makedirs(images_folder_img)
    if not os.path.exists(images_folder_pred):
        os.makedirs(images_folder_pred)
    if not os.path.exists(images_folder_mask):
        os.makedirs(images_folder_mask)
    if not os.path.exists(images_folder_gtmask):
        os.makedirs(images_folder_gtmask)
    if not os.path.exists(images_folder_mix):
        os.makedirs(images_folder_mix)
        

    total_num_list = []
    point_num_list = []

    
    for i,info in enumerate(img_path_list):
        img_item = img_path_list[i]
        
        img_path = os.path.join(image_root,img_item)
        mask_path = os.path.join(mask_root,img_item)
        
        gt = Image.open(img_path).convert("RGB")
        seg_map = Image.open(mask_path).convert("RGB")

        contours = detect_and_draw_edges(seg_map)
        
        origin = gt.convert("P")
        seg_map = seg_map.convert("L")
        gt = cv2.drawContours(cv2.cvtColor(np.asarray(gt),cv2.COLOR_RGB2BGR),contours,-1,(255,0,0),2,lineType=cv2.LINE_AA)       
        gt = Image.fromarray(gt)
            
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
            
            pil_mask = pil_mask_pred.point(lambda p: p > 45 and 255)

            total_num,point_num = score_cal(seg_map,pil_mask)
            
            

            total_num_list.append(total_num)
            point_num_list.append(point_num)
  
            
            
            seg_map.save(images_folder_gtmask / img_item)
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
    point_score = np.mean(point_num_list)/np.mean(total_num_list)
    print('The average point_score is {point_score:.5f}'.format(point_score=point_score))    
          
if __name__ == "__main__":
    val_real_chexlocalize()

