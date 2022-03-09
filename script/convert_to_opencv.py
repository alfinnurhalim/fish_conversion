import sys
import os
import json

import pandas as pd

from datetime import datetime
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath('fish_conversion'))
sys.path.append(BASE_DIR)

from lib.dataloader.Unity_dataloader import load_unity_dataset
from lib.folder_manager.OpenCV_folder_manager import OpenCV_folder_manager
from lib.folder_manager.COCO_folder_manager import COCO_folder_manager
from lib.dataset.OpenCV_Dataset import OpenCV_Dataset
from lib.dataset.COCO_Dataset import COCO_Dataset

#=================================================CONFIGURATIONS==============================================================
# root dir 
DATASET_DIR = '../synthetic_dataset/yellowtail_9_raw_small'

DATASET_NAME = os.path.basename(DATASET_DIR)

# output_path
OUT_DIR = os.path.join('data','converted',DATASET_NAME)

# number of data will be used
DATA_NUM = 100

# delete pervious folder
reset = True

# add color augmentation
aug = False

# choose cam id
cam_id = 'cam3'

# cam rel
cam_relative = True

#=============================================LOAD FROM UNITY=============================================================

unity_dataset = load_unity_dataset(dataset_dir = DATASET_DIR,data_num = DATA_NUM)

#=============================================CONVERT TO OPENCV=============================================================

opencv_dataset = OpenCV_Dataset()
opencv_dataset.load_from_unity(unity_dataset)

fm = OpenCV_folder_manager(data_dir=DATASET_DIR,name=DATASET_NAME)
fm.create_folder()

print('saving to OPENCV dataset ....')
opencv_dataset.save_to_json(os.path.join(fm.ann_dir,DATASET_NAME+'.json'))
opencv_dataset.save_to_pkl(os.path.join(fm.ann_dir,DATASET_NAME+'.pkl'))

#=============================================CONVERT TO COCO=============================================================
coco_dataset = COCO_Dataset()
coco_dataset.load_from_opencv(opencv_dataset)

fm = COCO_folder_manager(data_dir=DATASET_DIR,name=DATASET_NAME)
fm.create_folder()

print('saving to COCO dataset ....')
coco_dataset.save_to_json(os.path.join(fm.ann_dir,DATASET_NAME+'.json'))
coco_dataset.save_to_pkl(os.path.join(fm.ann_dir,DATASET_NAME+'.pkl'))
print('Dataset Converted Successfully')