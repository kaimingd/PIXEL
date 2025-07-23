import cv2
import sys
import os
import torch
import glob
import numpy as np
import random
from torch.utils.data import Dataset
import csv
import pandas as pd
    
# TODO: TRAIN / VALIDATION dataset for condition adaptor on COLLECTED DATASET with TEXT-TO-IMAGE diffusion models
class T2ICollectedDataset(Dataset):
    def __init__(self, cond_type, image_dir, cond_dir, text_dir, image_size, dataset_scale=None, kmeans_center=None):
        super().__init__()
        self.image_size = image_size
        self.cond_type = "edge"                                
        self.cond_dir = cond_dir
        self.text_dir = text_dir
        if image_dir == '/hhd/1/dkm/DiffuMask/DiffMask_mimic_lcdg/images':
            self.split = 'train'
        elif image_dir == '/hhd/1/dkm/DiffuMask/DiffMask_mimic_lcdg_val/images':
            self.split = 'valid'
        
        # self.split = "valid"
        data = pd.read_csv(self.split+'.csv', encoding='gbk')
        self.image_paths = ["/hhd/1/dkm/KAD/"+item for item in data['Path'].values.tolist()]
        labels = data[['Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
                      'Pleural Effusion','Pleural Other','No Finding','Pneumothorax','Pneumonia','Support Devices']]
        labels2 = ["atelectasis", "cardiomegaly","consolidation","edema","enlargedcardiomediastinum","fracture","lunglesion",
        "lungopacity", "pleuraleffusion","pleuralother","nofinding","pneumothorax","pneumonia","supportdevices"]
        arr = labels.values.astype(float)
        index = np.where(arr==1.0)
        self.labels = []
        for id in index[1]:
            self.labels.append(labels2[id])
            
            
        for i,item in enumerate(self.image_paths):
            num = 300000+i
            img = cv2.imread(item)
            img = cv2.resize(img,(image_size,image_size))
            out_dir = os.path.join("/hhd/1/dkm/lung-segmentation/real_images_train/","image_"+str(num)+"_"+self.labels[i]+".jpg")
            cv2.imwrite(out_dir,img)
            
            
        
        

        
        # self.image_paths0 = os.listdir(image_dir)
        # self.image_paths = [os.path.join(image_dir,dir_item) for dir_item in self.image_paths0]
        # self.image_paths = sorted(self.image_paths)
        # with open(os.path.join(self.text_dir,'captions.txt')) as f:
        #     self.text = f.readline()

        
        
        if dataset_scale is not None:
            self.image_paths = self.image_paths[:dataset_scale]

        # k-means for stroke generation
        if self.cond_type == 'stroke':  
            assert kmeans_center is not None
            self.kmeans_center = random.randint(kmeans_center // 2, kmeans_center)
            self.criteria = (cv2.TERM_CRITERIA_EPS + 
                             cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            self.flags = cv2.KMEANS_RANDOM_CENTERS

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
                
        # load in image
        image = cv2.imread(self.image_paths[index])
        label = self.labels[index]
        filename = os.path.basename(self.image_paths[index]).split('.')[0]
        
        if self.cond_type == 'stroke':
            filtered_image = cv2.medianBlur(image, ksize=23)

            # k-means
            compactness, label, center = cv2.kmeans(np.float32(filtered_image.reshape(-1, 3)),
                                                    self.kmeans_center,
                                                    None, self.criteria, 10, self.flags)

            center = np.uint8(center)
            cond = center[label.flatten()]
            cond = cond.reshape((image.shape))
        
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        
        
        
        image = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)      #-[0,255]-[-1,1]            # [H, W, C] -> [C, H, W]
        
        # load correpsonding condition
        if self.cond_type == 'image':
            cond = cv2.imread(self.image_paths[index])
        elif self.cond_type == 'edge' or self.cond_type == "mask" or self.cond_type == "depth":    # read from local path
            # cond = cv2.imread(os.path.join(self.cond_dir, filename + '.jpg'))
            cond = cv2.imread(self.image_paths[index].replace(self.split,self.split+'_sketchs'))                         
        # blur
            
        cond = cv2.resize(cond, (self.image_size, self.image_size))
        
        # TODO: only binarize condition of ``edge'' or ``mask''
        # TODO: switch BGR channels of color conditions into RGB channels
        if self.cond_type == "edge":
            threshold = random.uniform(120, 180)
            _, cond = cv2.threshold(cond, thresh=threshold, maxval=255.0, type=cv2.THRESH_BINARY)              # to binarize
            
            # random dilate and erode the edges
            dilate_iter = random.uniform(1, 4)
            erode_iter = random.uniform(1, 4)
            select_index = random.uniform(1, 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

            if select_index == 1:
                cond = cv2.dilate(cond, kernel, iterations=int(dilate_iter))
            else:
                cond = cv2.erode(cond, kernel, iterations=int(erode_iter))
        elif self.cond_type == "mask":
            _, cond = cv2.threshold(cond, thresh=127.5, maxval=255.0, type=cv2.THRESH_BINARY)              # to binarize
        elif self.cond_type == "stroke" or self.cond_type == 'palette':
            cond = cv2.cvtColor(cond, cv2.COLOR_BGR2RGB)    

        # cond = cv2.GaussianBlur(cond,(13,13),0)
        cond = torch.from_numpy(cond.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)                    # [H, W, C] -> [C, H, W]
        
        # load corresponding text description
        # with open(os.path.join(self.text_dir, filename + '.txt')) as f:
        #     for line in f.readlines():
        #         text = line
        # f.close()
        
        # if filename.split('_')[1] == 'pleuraleffusion':
        #     text = 'a xray with pleuraleffusion'
        # elif filename.split('_')[1] == 'nofinding':
        #     text = 'a xray with nofinding'
        
        # text = "a xray with {}".format(filename.split('_')[1])
        text = "a xray with {}".format(label)
        # warp in batch dict
        batch = {}
        batch['image'] = image
        batch['cond'] = cond
        batch['text'] = text
        
        return batch

    def get_files_from_txt(self, path):
        file_list = []
        f = open(path)
        for line in f.readlines():
            line = line.strip("\n")
            file_list.append(line)
            sys.stdout.flush()
        f.close()

        return file_list

    def get_files_from_path(self, path):
        ret = []
        for root, dirs, files in os.walk(path):
            for filespath in files:
                ret.append(os.path.join(root, filespath))
        return ret
    
    
# TODO: TRAIN / VALIDATION dataset for condition adaptor on COLLECTED DATASET with UNCONDITIONAL diffusion model
class U2ICollectedDataset(Dataset):
    def __init__(self, cond_type, image_dir, cond_dir, image_size, kmeans_center):
        super().__init__()
        self.image_size = image_size
        self.cond_type = cond_type                                   
        self.cond_dir = cond_dir
        self.image_paths = self.get_files_from_txt(image_dir)
        self.image_paths = sorted(self.image_paths)
        
        # k-means for stroke generation
        if self.cond_type == 'stroke':
            assert kmeans_center is not None
            self.kmeans_center = random.randint(kmeans_center // 2, kmeans_center)
            self.criteria = (cv2.TERM_CRITERIA_EPS + 
                             cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            self.flags = cv2.KMEANS_RANDOM_CENTERS

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        
        # load in image
        image = cv2.imread(self.image_paths[index])
        filename = os.path.basename(self.image_paths[index]).split('.')[0]
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)                  # [H, W, C] -> [C, H, W]

        # load correpsonding condition
        if self.cond_type == "palette":
            cond = cv2.imread(self.image_paths[index])
        else:
            cond = cv2.imread(os.path.join(self.cond_dir, filename + '.png'))
        cond = cv2.resize(cond, (self.image_size, self.image_size))
        
        # TODO: preprocess conditions - binarize or transfer to RGB channels
        if self.cond_type == "palette" or self.cond_type == "stroke":
            cond = cv2.cvtColor(cond, cv2.COLOR_BGR2RGB)
        elif self.cond_type == "edge":
            threshold = random.uniform(120, 180)
            _, cond = cv2.threshold(cond, thresh=threshold, maxval=255.0, type=cv2.THRESH_BINARY)              # to binarize
            
            # random dilate and erode the edges
            dilate_iter = random.uniform(1, 4)
            erode_iter = random.uniform(1, 4)
            select_index = random.uniform(1, 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

            if select_index == 1:
                cond = cv2.dilate(cond, kernel, iterations=dilate_iter)
            else:
                cond = cv2.erode(cond, kernel, iterations=erode_iter)
        elif self.cond_type == "mask":
            _, cond = cv2.threshold(cond, thresh=127.5, maxval=255.0, type=cv2.THRESH_BINARY)              # to binarize
        
        #如果cond为mask则直接处理，不为mask需要别的处理方法
        cond = torch.from_numpy(cond.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)                    # [H, W, C] -> [C, H, W]

        # warp in batch dict
        batch = {}
        batch['image'] = image
        batch['cond'] = cond
        
        return batch

    def get_files_from_txt(self, path):
        file_list = []
        f = open(path)
        for line in f.readlines():
            line = line.strip("\n")
            file_list.append(line)
            sys.stdout.flush()
        f.close()

        return file_list

    def get_files_from_path(self, path):
        ret = []
        for root, dirs, files in os.walk(path):
            for filespath in files:
                ret.append(os.path.join(root, filespath))
        return ret