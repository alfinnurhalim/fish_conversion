import sys
import os
import pandas as pd

from datetime import datetime
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath('fish_conversion'))
sys.path.append(BASE_DIR)

from lib.dataloader.Unity_dataloader import Unity_Dataloader
from lib.dataset.OpenCV_Dataset import Fish_data
from lib.dataset.opencv_utils import get_3d_corner,get_cam_info,convert_to_opencv_coord,convert_to_cam_coord

#=================================================CONFIGURATIONS==============================================================
# root dir 
DATASET_DIR = '../cam_3_75'

DATASET_NAME = os.path.basename(DATASET_DIR)

# output_path
OUT_DIR = os.path.join('data','converted',DATASET_NAME)

# number of data will be used
# DATA_NUM = 50
DATA_NUM = 10

# delete pervious folder
reset = True

# add color augmentation
aug = False

# choose cam id
cam_id = 'cam3'

# cam rel
cam_relative = True
#============================================END OF CONFIGURATION==============================================================
fish_dl = Unity_Dataloader(DATASET_DIR,DATA_NUM)
data = fish_dl.fish_files[0]

cam_info = get_cam_info(data.cam_transform)
corners = get_3d_corner(data.ann_3d)
corners = [convert_to_cam_coord(corner,cam_info) for corner in corners]
corners = [convert_to_opencv_coord(corner) for corner in corners]


print(corners)



# filenames = get_filenames(DATASET_DIR,cam_id,DATA_NUM)
# print(filenames[0],extract_file_info(filenames[0]))
# print(get_img_path(filenames[0]))
