import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
input_dir = "stage1_generation_files_single/results/adapter_chestxray_paired_mimic1.8"
output_dir = "stage1_generation_files_single/results//adapter_chestxray_paired_mimic1.8_gradcam"


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

name_list = os.listdir(input_dir)

for name in name_list:
    img_dir = os.path.join(input_dir,name)
    img_out_dir = os.path.join(output_dir,name)
    image = cv2.imread(img_dir,2)
    norm_img = np.zeros(image.shape)
    norm_img = cv2.normalize(image,None, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8) 
    heat_img=cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    ori_img = cv2.imread(img_dir.replace("_minus1",""))
    img_add = cv2.addWeighted(heat_img, 0.5, ori_img, 0.5, 0)
    cv2.imwrite(img_out_dir,img_add)