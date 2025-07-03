import albumentations
from albumentations import Compose, Resize 
from albumentations.pytorch import ToTensorV2

import torch.nn as nn
import torch
import os
import cv2
import numpy as np
import argparse
from PIL import Image

from yacs.config import CfgNode as CN       #cvcore/config/default.py
import segmentation_models_pytorch as smp   #cvcore/model/model_zoo.py


cfg = CN()                  #cvcore/config/default.py
cfg.MODEL = CN()
cfg.MODEL.NAME = "unet(b0)" #cvcore/config/multi_unet_b0_diceloss.yaml
cfg.MODEL.NUM_CLASSES = 20  #cvcore/config/multi_unet_b0_diceloss.yaml
cfg.DATA = CN()
cfg.DATA.INP_CHANNEL = 1    #cvcore/config/multi_unet_b0_diceloss.yaml

def build_model(cfg):
    if 'unet(b0)'== cfg.MODEL.NAME:
        model= smp.Unet('efficientnet-b0',classes=cfg.MODEL.NUM_CLASSES,encoder_weights='imagenet'\
        ,in_channels=cfg.DATA.INP_CHANNEL)
    return model


def get_args():
    parser = argparse.ArgumentParser(description="Test the valided model on a test image.")
    parser.add_argument("--weights", type=str, default="weights/multi_unet_b0_DiceLoss.yaml.pth",
                        help="Path to the valided weights file.")
    parser.add_argument("--image", type=str, 
                        default="/Users/ucy-compsci/U-Net lung segmentation/input/shcxr-lung/images/CHNCXR_0015_0.png", 
                        help="Path to the test image file.")
    return parser.parse_args()


def load_image(image_path, transform):
    # Load the image and convert it to grayscale
    img = Image.open(image_path)
    # img_ori = cv2.imread(image_path.replace('valid','valid_masks'))
    img_ori = cv2.imread(image_path.replace('_ref.jpg',"_seg.jpg"))
    img_ori = cv2.cvtColor(img_ori, code = cv2.COLOR_BGR2GRAY)
    img_size = img.size
    print(img_size)


    print("Image mode:", img.mode)
    img = img.convert('L')

    print("Image mode:", img.mode) #mode is L (Luminance) indicates a grayscale image single-channel black and white.

    img = np.asarray(img, dtype=np.float32)
    img = img / 255
    img = np.expand_dims(img, axis=2)

    print("Image shape:", img.shape)

    dic = transform(image=img)
    img_tensor = dic['image']


    return img_tensor, img_size, img_ori


def main(args):
    print("Build model...")
    model = build_model(cfg)

    if getattr(torch, 'has_mps', False):
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(device)
    model = model.to(device)
    model = nn.DataParallel(model)

    print("Load weights...")
    checkpoint = torch.load(args.weights, "cpu")   #dkm
    model.load_state_dict(checkpoint['state_dict'])
    # print(f"Loaded checkpoint '{args.weights}' (epoch {checkpoint['epoch']})")

    print("Build data loader...")
    #data_loader = make_multi_ribs_dataloader(cfg, mode='test')
    data_transform = Compose([
        Resize(512, 512),
        ToTensorV2()
    ])
    
    # root = '/hhd/1/dkm/KAD/CheXpert-v1.0-small/'
    # img_root = os.path.join(root,'valid')
    # out_root = os.path.join(root,'valid_sketchs2')
    # a = os.listdir(img_root)
    # a.sort()
    # for img_dir0 in a:
    #     img_dir1 = os.path.join(img_root,img_dir0)
    #     for img_dir2 in os.listdir(img_dir1):
    #         img_dir3 = os.path.join(img_dir1,img_dir2)
    #         for img_dir in os.listdir(img_dir3):
    #             if not os.path.exists(img_dir3.replace('valid','valid_sketchs2')):
    #                 os.makedirs(img_dir3.replace('valid','valid_sketchs2'))
    #             origin_filename = os.path.join(img_dir3,img_dir)
    #             final_out = os.path.join(img_dir3.replace('valid','valid_sketchs2'),img_dir)
                

    for j in range(300000,303000):
        img_tensor, img_size, img_mask = load_image("reference_data/image_reference_"+str(j)+"_ref.jpg",transform=data_transform)
        print("Inference...")
        model.eval()

        list_label= ['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10',\
                    'L1','L2','L3','L4','L5','L6','L7','L8','L9','L10']

        
        with torch.no_grad():
            img_tensor = img_tensor.to(device=device,dtype=torch.float)
            print("img_tensor shape:", img_tensor.shape)
            output = model(img_tensor.unsqueeze(0))
            print("output shape:", output.shape)
            output = torch.sigmoid(output) > 0.5
            output = output.cpu().numpy().astype(np.uint8)
            print(range(output.shape[1]))

            # Resize the output masks to the original image size and write to disk
            img_ori = (output[0][0] * 255).astype(np.uint8)
            img_ori  = cv2.resize(img_ori , (img_size[0], img_size[1]))

            for i in range(output.shape[1]):
                mask = (output[0][i]* 255).astype(np.uint8)
                mask = cv2.resize(mask, (img_size[0], img_size[1]))
                # cv2.imwrite("predictions/prediction_"+list_label[i]+".png", mask)
                # mask0 = np.stack([mask,mask,mask]).transpose(1,2,0)
                img_ori = cv2.add(mask,img_ori)
            
            
            img_ori = img_ori*(img_mask/255).astype(np.uint8)
            threshold, binary = cv2.threshold(img_ori, 180, 255, cv2.THRESH_OTSU) 
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
            mask = np.zeros_like(img_ori, dtype = np.uint8)
            mask = cv2.drawContours(mask, contours, -1,
                            (255,255,255),
                            thickness = 2)  
            
            
            threshold, binary = cv2.threshold(img_mask, 180, 255, cv2.THRESH_OTSU) 
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
            mask2 = np.zeros_like(img_mask, dtype = np.uint8)
            mask2 = cv2.drawContours(mask2, contours, -1,
                            (255,255,255),
                            thickness = 2) 
            
            sketck = cv2.add(mask,mask2)
            

            cv2.imwrite("reference_data/image_reference_"+str(j)+"_ske.jpg", sketck)


if __name__ == '__main__':
    args = get_args()
    main(args)
