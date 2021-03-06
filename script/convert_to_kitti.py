import sys
import os
import json

import pandas as pd

from datetime import datetime
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath('fish_conversion'))
sys.path.append(BASE_DIR)

from lib.dataloader.Unity_dataloader import load_unity_dataset
from lib.folder_manager.KITTI_folder_manager import KITTI_folder_manager
from lib.folder_manager.OpenCV_folder_manager import OpenCV_folder_manager
from lib.dataset.OpenCV_Dataset import OpenCV_Dataset
from lib.dataset.KITTI_Dataset import KITTI_Dataset
#=================================================CONFIGURATIONS==============================================================
# root dir 
DATASET_DIR = '../synthetic_dataset/yellowtail_multi_2/cam3'

DATASET_NAME = 'yellowtail_multi_2_cam3_test2'
# os.path.basename(DATASET_DIR)

# number of data will be used
DATA_NUM = 99

#=============================================LOAD FROM UNITY=============================================================

unity_dataset = load_unity_dataset(dataset_dir = DATASET_DIR,data_num = DATA_NUM)

#=============================================CONVERT TO OPENCV=============================================================
fm = OpenCV_folder_manager(data_dir=DATASET_DIR,name=DATASET_NAME)
fm.create_folder()

opencv_dataset = OpenCV_Dataset()
opencv_dataset.load_from_unity(unity_dataset,fm)
opencv_dataset.save_image()

#=============================================CONVERT TO COCO Tracking =============================================================
tag = 'detection'
fm = KITTI_folder_manager(data_dir=DATASET_DIR,name=DATASET_NAME,tag=tag,split='training')
fm.create_folder()

kitti_dataset = KITTI_Dataset()
kitti_dataset.load_from_opencv(opencv_dataset,fm)

kitti_dataset.save_image(tag=tag)
kitti_dataset.save_camera(tag=tag)
kitti_dataset.save_ann(tag=tag)
print('Dataset Converted Successfully')