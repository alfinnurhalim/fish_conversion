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
from lib.folder_manager.KITTI_folder_manager import KITTI_folder_manager
from lib.folder_manager.COCO_folder_manager import COCO_folder_manager

from lib.dataset.OpenCV_Dataset import OpenCV_Dataset
from lib.dataset.COCO_Dataset import COCO_Dataset
from lib.dataset.ModKITTI_Dataset import KITTI_Dataset

#=================================================CONFIGURATIONS==============================================================
# root dir 
DATASET_NAME = ['normal_exp','high_exp']
DATASET_DIR = '../synthetic_dataset/NEW_RESOLUTION'
# DATASET_DIR = '/media/alfin/Data_ext/dataset/REID_BIG_DATASET_2023/'
# os.path.basename(DATASET_DIR)

# number of data will be used
DATA_NUM = 300

FPS = 1
for name in DATASET_NAME:
	print(name)
	dataset_path = os.path.join(DATASET_DIR,name)
#=============================================LOAD FROM UNITY=============================================================

	unity_dataset = load_unity_dataset(dataset_dir = dataset_path,data_num = DATA_NUM, fps = FPS)

#=============================================CONVERT TO OPENCV=============================================================
	fm = OpenCV_folder_manager(data_dir=dataset_path,name=name)
	fm.create_folder()

	opencv_dataset = OpenCV_Dataset()
	opencv_dataset.load_from_unity(unity_dataset,fm)
	opencv_dataset.save_image()
	# opencv_dataset.save_to_json()
	# opencv_dataset.save_to_pkl()

#=============================================CONVERT TO KITTI detection =============================================================
	tag = 'detection'
	fm = KITTI_folder_manager(data_dir=dataset_path,name=name,tag=tag,split='training')
	fm.create_folder()

	kitti_dataset = KITTI_Dataset()
	kitti_dataset.load_from_opencv(opencv_dataset,fm)

	kitti_dataset.save_image(tag=tag)
	kitti_dataset.save_camera(tag=tag)
	kitti_dataset.save_ann(tag=tag)
	print('Dataset Converted Successfully')