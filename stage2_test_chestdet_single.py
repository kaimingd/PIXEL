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

def binary_image_union(images):
    """
    对一组二值图像进行取并集操作

    :param images: 一个包含多张二值图像的列表，每张图像是一个二维的NumPy数组
    :return: 并集后的二值图像
    """
    if not images:
        raise ValueError("The input image list is empty")

    # 确保所有图像的形状相同
    image_shape = images[0].shape
    # for img in images:
    #     if img.shape != image_shape:
    #         raise ValueError("All images must have the same shape")

    # 初始化结果图像为全零数组
    union_image = np.zeros(image_shape, dtype=np.uint8)

    # 对所有图像进行按位或操作
    for img in images:
        union_image = np.bitwise_or(union_image, img)

    return union_image

def box2mask(box, w, h):
    """
    Transfer the box to mask
    :param box: box in the original image (x, y, w, h)
    :param w: width of the original image
    :param h: height of the original image
    :return: mask in the original image
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[box[1]:box[3], box[0]:box[2]] = 255
    return mask
                      
def val_real_chestdet():
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(512 / 16), int(512 / 16))
    unet = ViT_seg(config_vit, img_size=512, num_classes=config_vit.n_classes).cuda()
    model_name = "unet-6v-latest.pt"
    unet.load_state_dict(torch.load(images_folder / model_name, map_location=torch.device("cpu")))
    unet.to(device)
    unet.eval();

    img_root = "source_data/Chest-Det_Data/test_data"
    json_root = "source_data/Chest-Det_Data/test.json"
    with open(json_root, 'r') as f:
        dict = json.load(f)
    dataset_id = "chestxdet_test"
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

    
    for i,info in enumerate(dict):  
        img_item = info['file_name'].replace('png','jpg')
        labels = "_".join(info['syms'])
        bbox = info['boxes']
        bbox_new = []
        mask_new = []
        for line in bbox:
            line_new = []
            mask = []
            for j,item in enumerate(line):
                line_new.append(round(item/2))
            mask_new.append(box2mask(line_new,512,512))
            bbox_new.append(line_new)  
        
        if len(bbox_new) == 0:
            mask_gt = []
        elif len(bbox_new) == 1:
            mask_gt = mask_new[0]
        else:
            mask_gt = binary_image_union(mask_new)

        
        img_dir = os.path.join(img_root,img_item)
        img_item = img_item.replace(".jpg","_"+labels+".jpg")
        gt = Image.open(img_dir).resize((512, 512), Image.LANCZOS).convert("RGB")
        origin = gt.convert("P")
        draw = ImageDraw.Draw(gt)
        if bbox != []:
            for bbox_item in bbox_new:
                draw.rectangle(bbox_item, outline=(255,0,0))
                
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
            if len(bbox_new) != 0:
                total_num,point_num = score_cal(pil_mask,Image.fromarray(mask_gt))
                
                total_num_list.append(total_num)
                point_num_list.append(point_num)
                
                
                Image.fromarray(mask_gt).save(images_folder_gtmask / img_item)

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
    val_real_chestdet()

