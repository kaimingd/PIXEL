import torch
import torchvision

import pandas as pd
import numpy as np
from scipy.ndimage.interpolation import zoom
from PIL import Image
import random
import re

class LungDataset(torch.utils.data.Dataset):
    def __init__(self, origin_list, origins_folder, origins_folder2, masks_folder, masks_folder2, transforms=None):
        self.origin_list = origin_list
        self.origins_folder = origins_folder
        self.origins_folder2 = origins_folder2
        self.masks_folder = masks_folder
        self.masks_folder2 = masks_folder2
        self.transforms = transforms
        self.pathology_list = ["atelectasis", "cardiomegaly","consolidation","edema","enlarged cardiomediastinum","fracture","lung lesion",
        "lung opacity", "pleural effusion","pleural other","pneumothorax","pneumonia","support devices"]
        # self.thres = [70,70,70,60,50,60,70,60,60,60,60,60,60,70]
        
    
    def __getitem__(self, idx):
        origin_name = self.origin_list[idx]
        origin = Image.open(self.origins_folder / (origin_name + ".jpg")).convert("P")
        mask = Image.open(self.masks_folder / (origin_name + ".jpg"))
        
        origin_array = np.array(origin).astype(np.float32)
        mask_array = np.array(mask).astype(np.float32)
        
        origin_array = (origin_array - np.min(origin_array)) / (np.max(origin_array) - np.min(origin_array))
        if mask_array.max() != 0:
            mask_array = (mask_array - np.min(mask_array)) / (np.max(mask_array) - np.min(mask_array))

        pathology_name_ls = self.extract_elements_from_string(origin_name, self.pathology_list)
        pathology_name = pathology_name_ls[0]
        pathology_id = self.pathology_list.index(pathology_name)
        
        mask_array[mask_array >= 0.9] = pathology_id + 1
        mask_array[mask_array < 0.9] = 0
        
        if self.transforms is not None:
            origin, mask = self.transforms((origin_array, mask_array))
           
        return origin, mask, origin_name
        
    
    def __len__(self):
        return len(self.origin_list)
    
    def extract_elements_from_string(self, input_string, element_list):
        # 使用正则表达式来匹配字符串中的元素
        found_elements = [element for element in element_list if re.search(r'\b' + re.escape(element) + r'\b', input_string)]
        return found_elements

    
class Pad():
    def __init__(self, max_padding):
        self.max_padding = max_padding
        
    def __call__(self, sample):
        origin, mask = sample
        padding = np.random.randint(0, self.max_padding)
#         origin = torchvision.transforms.functional.pad(origin, padding=padding, padding_mode="symmetric")
        origin = torchvision.transforms.functional.pad(origin, padding=padding, fill=0)
        mask = torchvision.transforms.functional.pad(mask, padding=padding, fill=0)
        return origin, mask

class Crop():
    def __init__(self, max_shift):
        self.max_shift = max_shift
        
    def __call__(self, sample):
        origin, mask = sample
        tl_shift = np.random.randint(0, self.max_shift)
        br_shift = np.random.randint(0, self.max_shift)
        origin_w, origin_h = origin.size
        crop_w = origin_w - tl_shift - br_shift
        crop_h = origin_h - tl_shift - br_shift
        
        origin = torchvision.transforms.functional.crop(origin, tl_shift, tl_shift,
                                                        crop_h, crop_w)
        mask = torchvision.transforms.functional.crop(mask, tl_shift, tl_shift,
                                                        crop_h, crop_w)
        return origin, mask


class Resize():
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, sample):
        # origin, mask = sample['image'], sample['label']
        # x, y = origin.shape
        # if x != self.output_size[0] or y != self.output_size[1]:
        #     origin = zoom(origin, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
        #     mask = zoom(mask, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # origin = torch.from_numpy(origin.astype(np.float32)).unsqueeze(0)
        # mask = torch.from_numpy(mask.astype(np.float32))
    
        origin, mask = sample
        origin = torchvision.transforms.functional.resize(origin, self.output_size)
        mask = torchvision.transforms.functional.resize(mask, self.output_size)
        
        return origin, mask


def blend(origin, mask1=None, mask2=None):
    img = torchvision.transforms.functional.to_pil_image(origin + 0.5).convert("RGB")
    if mask1 is not None:
        mask1 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.zeros_like(origin),
            torch.stack([mask1.float()]),
            torch.zeros_like(origin)
        ]))
        img = Image.blend(img, mask1, 0.2)
        
    if mask2 is not None:
        mask2 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.stack([mask2.float()]),
            torch.zeros_like(origin),
            torch.zeros_like(origin)
        ]))
        img = Image.blend(img, mask2, 0.2)
    
    return img
