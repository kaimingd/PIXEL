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


def box_transfer(box, w, h, scale):
    """
    Transfer the box from the original image to the resized image
    :param box: box in the original image (x, y, w, h)
    :param w: width of the original image
    :param h: height of the original image
    :param scale: the scale of the resized image
    :return: box in the resized image
    """
    size = (h, w)
    max_dim = max(size)
    max_ind = size.index(max_dim)
    box = np.array(box)

    # Resizing
    if max_ind == 0:
        # image is heigher
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
        box = box * wpercent
    else:
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
        box = box * hpercent

    # Padding
    if max_ind == 0:
        # height fixed at scale, pad the width
        pad_size = scale - desireable_size[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top = int(0)
        bottom = int(0)
    else:
        # width fixed at scale, pad the height
        pad_size = scale - desireable_size[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left = int(0)
        right = int(0)

    box[0] = int(np.floor(box[0] + left))
    box[1] = int(np.floor(box[1] + top))
    box[2] = int(np.floor(box[2]))
    box[3] = int(np.floor(box[3]))

    return box.astype(np.int32)

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
         
def get_annotation(path_to_json, scale=512, use_cxr_text=True):
    coco = COCO(annotation_file=path_to_json)
    cats = coco.cats
    merged = {}
    merged["path"] = []
    merged["gtmasks"] = []
    merged["label_text"] = []
    merged["boxes"] = []
    merged["category"] = []

    for img_id, anns in coco.imgToAnns.items():
        img = coco.loadImgs(img_id)[0]
        path = img["path"]
        mask_dct = {}
        bbox_dct = {}
        cats_dct = {}
        for ann in anns:
            bbox = ann["bbox"]
            w = ann["width"]
            h = ann["height"]
            category_id = ann["category_id"]
            tbox = box_transfer(bbox, w, h, scale)
            mask = box2mask(tbox, scale, scale)
            if use_cxr_text:
                category = cats[category_id]["name"]      #一共8类
                label_text = ann["label_text"].lower()

                if label_text not in mask_dct:
                    mask_dct[label_text] = mask
                    bbox_dct[label_text] = [tbox]
                    cats_dct[label_text] = category
                else:
                    mask_dct[label_text] += mask
                    bbox_dct[label_text].append(tbox)
                    cats_dct[label_text] = category
            else:
                category = cats[category_id]["name"]
                label_text = f"Findings suggesting {category}."
                if label_text not in mask_dct:
                    mask_dct[label_text] = mask
                    bbox_dct[label_text] = [tbox]
                    cats_dct[label_text] = category
                else:
                    mask_dct[label_text] += mask
                    bbox_dct[label_text].append(tbox)
                    cats_dct[label_text] = category

        for k, v in mask_dct.items():
            merged["path"].append(path)
            merged["gtmasks"].append(v)
            merged["label_text"].append(k)
            merged["boxes"].append(bbox_dct[k])
            merged["category"].append(cats_dct[k])

    return merged


def val_real_mscxr():
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(512 / 16), int(512 / 16))
    unet = ViT_seg(config_vit, img_size=512, num_classes=config_vit.n_classes).cuda()
    model_name = "unet-6v-latest.pt"
    unet.load_state_dict(torch.load(images_folder / model_name, map_location=torch.device("cpu")))
    unet.to(device)
    unet.eval();


    MS_CXR_JSON = Path("/hhd/2/dkm/ms-cxr/files/ms-cxr/0.1/MS_CXR_Local_Alignment_v1.0.0.json")
    MIMIC_IMG_DIR = Path("/hhd/0/dkm/REFERS-master/data/MIMIC/data/files")
    
    data = get_annotation(MS_CXR_JSON, use_cxr_text=True)
    data["path"] = list(map(lambda x: MIMIC_IMG_DIR/x.replace("files/", ""), data["path"]))
  
    img_path_list = data["path"]
    class_list = data["category"]
    bbox_list = data["boxes"]

    dataset_id = "mscxr"
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
        img_path = img_path_list[i]
        img_item = img_path_list[i].name[:-4].split('/')[-1] + "_" + class_list[i] + ".jpg"
        
        gt = Image.open(img_path).resize((512, 512), Image.LANCZOS).convert("RGB")

        boxes = bbox_list[i] 
        bbox_new = []
        mask_new = []
    
        seg_map = np.zeros((512,512))

        for box in boxes:
            cc = box
            seg_map[int(float(cc[1])):(int(float(cc[1]))+int(float(cc[3]))),int(float(cc[0])):(int(float(cc[0]))+int(float(cc[2])))]=255
            bbox_new.append([int(float(cc[0])),int(float(cc[1])),int(float(cc[0]))+int(float(cc[2])),int(float(cc[1]))+int(float(cc[3]))])
        
        origin = gt.convert("P")
        draw = ImageDraw.Draw(gt) 
        if bbox_new != []:
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
                total_num,point_num = score_cal(pil_mask,Image.fromarray(seg_map).convert("L"))

                

                total_num_list.append(total_num)
                point_num_list.append(point_num)

                Image.fromarray(seg_map).convert("L").save(images_folder_gtmask / img_item)

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
    val_real_mscxr()