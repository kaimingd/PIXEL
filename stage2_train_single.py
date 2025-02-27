#!/usr/bin/env python
# coding: utf-8
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
import shutil
from skimage.metrics import peak_signal_noise_ratio
from stage2_segmentation_files_single.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from stage2_segmentation_files_single.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from stage2_segmentation_files_single.src.utils import score_cal

add = "minus1"
dir_root = "stage1_generation_files_single/results/adapter_chestxray_paired_mimic1.8"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Root_dir = "stage2_segmentation_files_single"

origins_folder = Path(dir_root)
masks_folder = Path(dir_root+"_"+add)
task = "images_transunet_minus1"

images_folder = Path(os.path.join(Root_dir,"results",task))
if not os.path.exists(images_folder):
    os.makedirs(images_folder)
    
batch_size = 2

def load():
    origins_list = [f.stem for f in origins_folder.glob("*.jpg")]
    origin_mask_list = [f.stem for f in masks_folder.glob("*.jpg")]
    print(len(origins_list))
    print(len(origin_mask_list))

    split_file = os.path.join(Root_dir,"splits_transunet_simu_chexpert.pk")
    if os.path.isfile(split_file):
        with open(split_file, "rb") as f:
            splits = pickle.load(f)
    else:
        splits = {}
        splits["train"], splits["test"] = train_test_split(origin_mask_list, test_size=0.2, random_state=42) #0.2
        splits["train"], splits["val"] = train_test_split(splits["train"], test_size=0.1, random_state=42) #0.1

        with open(split_file, "wb") as f:
            pickle.dump(splits, f)

    val_test_transforms = torchvision.transforms.Compose([
        Resize((512, 512)),
    ])

    train_transforms = torchvision.transforms.Compose([
        Pad(200),
        # Crop(300),
        val_test_transforms,
    ])

    datasets = {x: LungDataset(
        splits[x], 
        origins_folder, 
        masks_folder, 
        train_transforms if x == "train" else val_test_transforms
    ) for x in ["train", "test", "val"]}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size) for x in ["train", "test", "val"]}

    idx = 0
    phase = "train"

    plt.figure(figsize=(20, 10))
    origin, mask, _ = datasets[phase][idx]
    pil_origin = torchvision.transforms.functional.to_pil_image(origin + 0.5).convert("RGB")
    # pil_mask = torchvision.transforms.functional.to_pil_image(mask.float())
    pil_mask = torchvision.transforms.functional.to_pil_image(mask + 0.5).convert("RGB")

    plt.subplot(1, 3, 1)
    plt.title("origin image")
    plt.imshow(np.array(pil_origin))

    plt.subplot(1, 3, 2)
    plt.title("manually labeled mask")
    plt.imshow(np.array(pil_mask))

    # plt.subplot(1, 3, 3)
    # plt.title("blended origin + mask")
    # plt.imshow(np.array(blend(origin, mask)));

    plt.savefig(images_folder / "data-example.png", bbox_inches='tight')
    return datasets,dataloaders


def train(datasets,dataloaders):
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(512 / 16), int(512 / 16))
    unet = ViT_seg(config_vit, img_size=512, num_classes=config_vit.n_classes).cuda()


    unet = unet.to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=0.0005)
    criterion = torch.nn.MSELoss()
    train_log_filename = "train-log.txt"
    epochs = 100
    best_val_loss = np.inf
    model_name = "unet-6v-try.pt"     #需要修改

    hist = []

    for e in range(epochs):
        start_t = time.time()
        
        print("train phase")
        unet.train()
        train_loss = 0.0
        for origins, masks, name in dataloaders["train"]:
            num = origins.size(0)
            
            origins = origins.to(device)#[4,1,512,512]
            masks = masks.to(device)#[4,512,512]
            
            optimizer.zero_grad()
            
            outs = unet(origins) #[1,1,512,512]
            loss = criterion(outs,masks)
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item() * num
            print('loss : %f' % (loss.item()))

            
        train_loss = train_loss / len(datasets['train'])
        
        print("validation phase")
        unet.eval()
        val_loss = 0.0
        val_jaccard = 0.0
        val_dice = 0.0

        for origins, masks,name in dataloaders["val"]:
            num = origins.size(0)

            origins = origins.to(device)
            masks = masks.to(device)

            with torch.no_grad():
                outs = unet(origins)
                softmax = torch.nn.functional.log_softmax(outs, dim=1)
                val_loss +=  criterion(outs,masks).item() * num
                outs = torch.argmax(softmax, dim=1)
                outs = outs.float()
                masks = masks.float()
                # val_jaccard += jaccard(masks, outs.float()).item() * num
                # val_dice += dice(masks, outs).item() * num

            # print(".", end="")
            # break
            
        val_loss = val_loss / len(datasets["val"])
        # val_jaccard = val_jaccard / len(datasets["val"])
        # val_dice = val_dice / len(datasets["val"])
        val_jaccard = 0
        val_dice = 0
        
        end_t = time.time()
        spended_t = end_t - start_t
        
        with open(images_folder / train_log_filename, "a") as train_log_file:
            report = f"epoch: {e+1}/{epochs}, time: {spended_t}, train loss: {train_loss}, \n"\
                + f"val loss: {val_loss}, val jaccard: {val_jaccard}, val dice: {val_dice}"

            hist.append({
                "time": spended_t,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_jaccard": val_jaccard,
                "val_dice": val_dice,
            })

            print(report)
            train_log_file.write(report + "\n")
            
            torch.save(unet.state_dict(), images_folder / model_name.replace('-6v','-6v-latest'))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(unet.state_dict(), images_folder / model_name.replace('-6v','-6v-best'))
                print("model saved")
                train_log_file.write("model saved\n")
            
            
    plt.figure(figsize=(15,7))
    train_loss_hist = [h["train_loss"] for h in hist]
    plt.plot(range(len(hist)), train_loss_hist, "b", label="train loss")

    val_loss_hist = [h["val_loss"] for h in hist]
    plt.plot(range(len(hist)), val_loss_hist, "r", label="val loss")

    val_dice_hist = [h["val_dice"] for h in hist]
    plt.plot(range(len(hist)), val_dice_hist, "g", label="val dice")

    val_jaccard_hist = [h["val_jaccard"] for h in hist]
    plt.plot(range(len(hist)), val_jaccard_hist, "y", label="val jaccard")

    plt.legend()
    plt.xlabel("epoch")
    plt.savefig(images_folder / model_name.replace(".pt", "-train-hist.png"))

    time_hist = [h["time"] for h in hist]
    overall_time = sum(time_hist) // 60
    mean_epoch_time = sum(time_hist) / len(hist)
    print(f"epochs: {len(hist)}, overall time: {overall_time}m, mean epoch time: {mean_epoch_time}s")


    torch.cuda.empty_cache()




def val(datasets,dataloaders):

    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    # if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(512 / 16), int(512 / 16))
    unet = ViT_seg(config_vit, img_size=512, num_classes=config_vit.n_classes).cuda()
    criterion = torch.nn.MSELoss()
    model_name = "unet-6v.pt".replace('-6v','-6v-latest')
    unet.load_state_dict(torch.load(images_folder / model_name, map_location=torch.device("cpu")))
    unet.to(device)
    unet.eval();
    
    
    dataset_id = "generated_test_from_chexpert"
    images_folder_gt = Path("images_transunet_"+add,dataset_id,"gt")
    images_folder_img = Path("images_transunet_"+add,dataset_id,"img")
    images_folder_pred = Path("images_transunet_"+add,dataset_id,"pred")
    images_folder_mask = Path("images_transunet_"+add,dataset_id,"mask")
    if not os.path.exists(images_folder_gt):
        os.makedirs(images_folder_gt)
    if not os.path.exists(images_folder_img):
        os.makedirs(images_folder_img)
    if not os.path.exists(images_folder_pred):
        os.makedirs(images_folder_pred)
    if not os.path.exists(images_folder_mask):
        os.makedirs(images_folder_mask)
        
    test_loss = 0.0
    test_jaccard = 0.0
    test_dice = 0.0
    test_psnr = 0.0

    for origins, masks, name in dataloaders["test"]:
        num = origins.size(0)

        origins = origins.to(device)
        masks = masks.to(device)

        with torch.no_grad():
            outs = unet(origins)
            softmax = torch.nn.functional.log_softmax(outs, dim=1)
            # test_loss +=  criterion((outs[:,0]+outs[:,1])/2,masks[:,0]).item() * num

            test_loss +=  criterion(outs,masks).item() * num
            # test_loss += torch.nn.functional.nll_loss(softmax, masks).item() * num

            # outs = torch.argmax(softmax, dim=1)
            out = outs.float()
            # out = (outs[:,0]+outs[:,1])/2
            masks = masks.float()
            # test_jaccard += jaccard(masks, outs).item() * num
            # test_dice += dice(masks, outs).item() * num

            pil_origin = torchvision.transforms.functional.to_pil_image(origins[0] + 0.5).convert("RGB")
            pil_mask_gt = torchvision.transforms.functional.to_pil_image(masks[0] + 0.5).convert("RGB")
            pil_mask_pred = torchvision.transforms.functional.to_pil_image(out[0] + 0.5).convert("RGB")
                    
            pil_mask_pred = pil_mask_pred.convert("L")
            pil_mask_gt = pil_mask_gt.convert("L")
            
            image_array = np.array(pil_mask_pred)

            white_mask = np.where(image_array >= 240)
            image_array[white_mask] = 0
            pil_mask_pred = Image.fromarray(image_array)            
            pil_mask = pil_mask_pred.point(lambda p: p > 35 and 255)     
                        
            pil_origin.save(images_folder_img / name[0])
            pil_mask_gt.save(images_folder_gt / name[0])
            pil_mask_pred.save(images_folder_pred / name[0])
            pil_mask.save(images_folder_mask / name[0])
            # plt.subplot(1, 3, 1)
            # plt.title("origin image")
            # plt.imshow(np.array(pil_origin))

            # plt.subplot(1, 3, 2)
            # plt.title("manually mask_pred")
            # plt.imshow(np.array(pil_mask_pred))
            
            # plt.subplot(1, 3, 3)
            # plt.title("manually mask_gt")
            # plt.imshow(np.array(pil_mask_gt))
            
            # plt.savefig(images_folder_mix / name[0], bbox_inches='tight')
            # test_psnr += peak_signal_noise_ratio(np.array(pil_mask_pred), np.array(pil_mask_gt)) * num



        print(".", end="")

    # test_loss = test_loss / len(datasets["test"])
    # # test_jaccard = test_jaccard / len(datasets["test"])
    # test_psnr = test_psnr / len(datasets["test"])
    
    # test_jaccard = 0
    # test_psnr = 0

    # print()
    # print(f"avg test loss: {test_loss}")
    # print(f"avg test jaccard: {test_jaccard}")
    # print(f"avg test dice: {test_psnr}")


    # num_samples = 9
    # phase = "test"

    # subset = torch.utils.data.Subset(
    #     datasets[phase], 
    #     np.random.randint(0, len(datasets[phase]), num_samples)
    # )
    # random_samples_loader = torch.utils.data.DataLoader(subset, batch_size=1)
    # plt.figure(figsize=(20, 25))

    # for idx, (origin, mask) in enumerate(random_samples_loader):
    #     plt.subplot((num_samples // 3) + 1, 3, idx+1)

    #     origin = origin.to(device)
    #     mask = mask.to(device)

    #     with torch.no_grad():
    #         out = unet(origin)
    #         softmax = torch.nn.functional.log_softmax(out, dim=1)
    #         out = torch.argmax(softmax, dim=1)

    #         jaccard_score = jaccard(mask.float(), out.float()).item()
    #         dice_score = dice(mask.float(), out.float()).item()

    #         origin = origin[0].to("cpu")
    #         out = out[0].to("cpu")
    #         mask = mask[0].to("cpu")

    #         plt.imshow(np.array(blend(origin, mask, out)))
    #         plt.title(f"jaccard: {jaccard_score:.4f}, dice: {dice_score:.4f}")
    #         print(".", end="")
                
    # plt.savefig(images_folder / "obtained-results.png", bbox_inches='tight')
    # print()         
    # print("red area - predict")
    # print("green area - ground truth")
    # print("yellow area - intersection")
        

if __name__ == "__main__":
    datasets,dataloaders = load()
    train(datasets,dataloaders)
    # val(datasets,dataloaders)

