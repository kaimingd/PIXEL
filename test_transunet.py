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

from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

from src.data import LungDataset, blend, Pad, Crop, Resize
from src.models import UNet, PretrainedUNet
from src.metrics import jaccard, dice


def test():
    unet = PretrainedUNet(
        in_channels=1,
        out_channels=2, 
        batch_norm=True, 
        upscale_mode="bilinear"
    )

    model_name = "unet-6v.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unet.load_state_dict(torch.load(Path(model_name), map_location=torch.device("cpu")))
    unet.to(device)
    unet.eval();
    
    root = '/hhd/1/dkm/KAD/CheXpert-v1.0-small/'
    img_root = os.path.join(root,'valid')
    out_root = os.path.join(root,'valid_masks')
    a = os.listdir(img_root)
    a.sort()
    for img_dir0 in a:
        img_dir1 = os.path.join(img_root,img_dir0)
        for img_dir2 in os.listdir(img_dir1):
            img_dir3 = os.path.join(img_dir1,img_dir2)
            for img_dir in os.listdir(img_dir3):
                if not os.path.exists(img_dir3.replace('valid','valid_masks')):
                    os.makedirs(img_dir3.replace('valid','valid_masks'))
                origin_filename = os.path.join(img_dir3,img_dir)
                final_out = os.path.join(img_dir3.replace('valid','valid_masks'),img_dir)
                
                origin = Image.open(origin_filename).convert("P")
                origin = torchvision.transforms.functional.resize(origin, (512, 512))
                origin = torchvision.transforms.functional.to_tensor(origin) - 0.5


                with torch.no_grad():
                    origin = torch.stack([origin])
                    origin = origin.to(device)
                    out = unet(origin)
                    softmax = torch.nn.functional.log_softmax(out, dim=1)
                    out = torch.argmax(softmax, dim=1)
                    
                    origin = origin[0].to("cpu")
                    out = out[0].to("cpu")

                img_out = (out.numpy()*255).astype(np.uint8)
                cv2.imwrite(final_out,img_out)
            
            
if __name__ == "__main__":
    test()
