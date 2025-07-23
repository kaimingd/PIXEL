# PIXEL
The official codes for **A generative model uses healthy and diseased image pairs for pixel-level chest X-ray pathology localization** published in Nature Biomedical Engineering.


## Dependencies

To clone all files:

```
git clone git@github.com:kaimingd/PIXEL.git
```

To install Python dependencies:


```
conda create -n pixel python=3.9.0
conda activate pixel
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```


#### **Training Dataset**   
**1. MIMIC-CXR Dataset**

Navigate to [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/) to download the training dataset. Note: in order to gain access to the data, you must be a credentialed user as defined on [PhysioNet](https://physionet.org/settings/credentialing/).

**2. CheXpert Dataset**
The CheXpert dataset consists of chest radiographic examinations from Stanford Hospital, performed between October 2002 and July 2017 in both inpatient and outpatient centers. Population-level characteristics are unavailable for the CheXpert test dataset, as they are used for official evaluation on the CheXpert leaderboard.
The main data (CheXpert data) supporting the results of this study are available at https://aimi.stanford.edu/chexpert-chest-x-rays.


#### **Evaluation Dataset**   

**1. RSNA Pneumonia dataset**

The RSNA Pneumonia dataset is a publicly available dataset created for a competition hosted by the Radiological Society of North America (RSNA) in collaboration with Kaggle. The primary goal of the challenge was to develop machine learning models capable of detecting pneumonia in chest X-rays. This dataset contains 26,684 training data and 3000 test data. For image downloading, please visit [link](https://www.rsna.org/rsnai/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018).

**2. ChestX-Det10 dataset**

ChestX-Det10 dataset is a subset with instance-level box annotations of [NIH ChestX-14](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community), which contains 3,543 images and labels corresponding 10 common categories of diseases or abnormalities annotated by three board-certified radiologists. For image downloading, please visit [link](https://github.com/Deepwise-AILab/ChestX-Det10-Dataset).

**3. MS-CXR dataset**

MS-CXR dataset is a large collection of chest X-ray images used for developing AI models in medical imaging. It includes thousands of X-rays labeled with various lung conditions, like pneumonia and lung opacities. The images come with detailed annotations, sometimes including bounding boxes to show specific areas of concern. For image downloading, please visit [link](https://physionet.org/content/ms-cxr/0.1/).

**4. CheXlocalize dataset**

CheXlocalize dataset is a specialized collection of chest X-ray images that includes detailed annotations to help locate specific abnormalities in the lungs. Chexlocalize contains 234 images and 643 expert segmentation. For image downloading, please visit [link](https://stanfordaimi.azurewebsites.net/datasets/abfb76e5-70d5-4315-badc-c94dd82e3d6d).

**5. PadChest Dataset**

The PadChest dataset contains chest X-rays that were interpreted by 18 radiologists at the Hospital Universitario de San Juan, Alicante, Spain, from January 2009 to December 2017. The dataset contains 109,931 image studies and 168,861 images. PadChest also contains 206,222 study reports. The [PadChest](https://arxiv.org/abs/1901.07441) is publicly available at https://bimcv.cipf.es/bimcv-projects/padchest. Those who would like to use PadChest for experimentation should request access to PadChest at the [link](https://bimcv.cipf.es/bimcv-projects/padchest).


## Data Preprocessing
**1. Disease-free data generation**

For generating disease-free data, We adopt [RoentGen](https://github.com/StanfordMIMI/RoentGen) with prompts "a photo of xray with no finding" and build the file list to collect preprocessed data
+ reference_data
  + image_reference_300000_ref.jpg
  + ......

Run the following command to perform disease-free data generation:

`python diffuser_mode_test.py`

**2. Rib extraction**

For training ControlNet, We adopt [TransUnet](https://github.com/Beckschen/TransUNet) to extract chest boundary constraint and [VinDr-RibCXR](https://github.com/vinbigdata-medical/MIDL2021-VinDr-RibCXR) to extract rib constraint from [CheXpert-small](https://www.kaggle.com/datasets/ashery/chexpert) dataset and add them to orign file list as 
+ CheXpert-v1.0-small
  + train
  + train_masks
  + train_sketchs
  + valid
  + valid_masks
  + valid_sketchs
  + train.csv
  + valid.csv
  
train_masks and train_sketchs have the same subdirectory structure as train.

For generating abnormality data, We adopt [TransUnet](https://github.com/Beckschen/TransUNet) to extract chest boundary constraint and [VinDr-RibCXR](https://github.com/vinbigdata-medical/MIDL2021-VinDr-RibCXR) to extract rib constraint from generated disease-free data and add boundary and ribs sketch to orign file list as
+ reference_data
  + image_reference_300000_ref.jpg
  + image_reference_300000_ske.jpg
  + image_reference_300000_seg.jpg
  + ......

Run the following command to perform rib chest boundary and constraint extraction:

`python test_transunet.py`

`python test_ribcxr.py`

## Generative Model Training

Run the following command to train ControlNet based on the weight of RoentGen model and parpared CheXpert-v1.0-small files

`python controlnet_train.py`

## Generation and Localization

**1. Stage1 Generation**


Run the following command to perform stage1 testing on ControlNet to generate paired normal/diseased images and labels files based on the reference_data files and trained ControlNet weight:

`python stage1_generation_single.py ` 

Run the following command to perform stage1 post-precessing paired normal/diseased data to get pathology location map

`python ./stage1_generation_files_single/gray2hotmap.py ` 


**2. Stage2 Localization**

Run the following command to perform stage2 training on Transunet(trained on train split of Generated paired data) to train a pathology localization model

`python stage2_train_single.py ` 

Run the following command to perform stage2 testing on Transunet(inference on test split of RSNA dataset) to test a pathology localization model

`python stage2_test_rsna_single.py ` 

Run the following command to perform stage2 testing on Transunet(inference on test split of MS-CXR dataset) to test a pathology localization model

`python stage2_test_mscxr_single.py ` 

Run the following command to perform stage2 testing on Transunet(inference on test split of ChestX-Det10 dataset) to test a pathology localization model

`python stage2_test_chestdet_single.py ` 

Run the following command to perform stage2 testing on Transunet(inference on test split of Chexlocalize dataset) to test a pathology localization model

`python stage2_test_chexlocalize_single.py ` 

Run the following command to perform stage2 testing on Transunet(inference on test split of PadChest dataset) to test a pathology localization model

`python stage2_test_padchest_single.py ` 



## Acknowledgement
We use code from [TransUnet](https://github.com/Beckschen/TransUNet), [VinDr-RibCXR](https://github.com/vinbigdata-medical/MIDL2021-VinDr-RibCXR), [RoentGen](https://github.com/StanfordMIMI/RoentGen), [ControlNet](https://github.com/lllyasviel/ControlNet) and [LCDG](https://github.com/AlonzoLeeeooo/LCDG?tab=readme-ov-file).
We thank the authors for releasing their code.


## Citation
If you use PIXEL in your research, please cite our paper:

'@article{dong2025generative,
  title={A generative model uses healthy and diseased image pairs for pixel-level chest X-ray pathology localization},
  author={Dong, Kaiming and Cheng, Yuxiao and He, Kunlun and Suo, Jinli},
  journal={Nature Biomedical Engineering},
  pages={1--13},
  year={2025},
  publisher={Nature Publishing Group UK London}
}'


