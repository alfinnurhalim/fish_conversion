# -*- coding: utf-8 -*-
# @Author: Muhammad Alfin N
# @Date:   2022-03-08 10:06:10
# @Last Modified by:   Muhammad Alfin N


# @Last Modified time: 2022-03-08 17:23:00

import os
import re
import cv2
import pickle
import json
import numpy as np
import math 

from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import lib.dataset.opencv_utils as utils 

IMAGE_SIZE = 512
RESIZE_FACTOR = 1

# Occluded pixel thr
VISIBILITY_THR_LOWER = 0.0001 # 0.05% of the img_size
VISIBILITY_THR_UPPER = 0.3 # 30% of the img_size

# Fog visibility thr
FOG_VIS_THR = 0.001

# Depth thr
Z_THR_LOWER = 0.2 
Z_THR_UPPER = 4

# Box Area thr
BBOX_2D_AREA_THR_LOWER = IMAGE_SIZE*0.05
BBOX_2D_AREA_THR_UPPER = IMAGE_SIZE*0.8

class OpenCV_Dataset(object):
	def __init__(self,image_size=256,resize_factor=4):
		self.data_dir = None
		self.fm = None

		self.fish_files = []

	def load_from_unity(self,Dataset,folder_manager):
		print('converting to opencv format')
		self.fm = folder_manager
		self.data_dir = Dataset.data_dir

		for unity_file in tqdm(Dataset.unity_files):
			fish_file = Fish_File()
			fish_file.load_from_unity(unity_file,self.fm)
			fish_file.convert_to_opencv()
			self.fish_files.append(fish_file)

	def load_from_testing(self,Dataset,folder_manager):
		print('converting to opencv format')
		self.fm = folder_manager
		self.data_dir = Dataset.data_dir

		for testing_file in tqdm(Dataset.testing_files):
			fish_file = Fish_File()
			fish_file.load_from_unity(testing_file,self.fm) #testing has same format as unity
			fish_file.convert_to_opencv()
			self.fish_files.append(fish_file)

	def to_dict(self):
		dataset = {
			'dataset_path':self.data_dir,
			'fish_files' : []
		}

		for file in self.fish_files:
			dataset['fish_files'].append(file.to_dict())

		return dataset

	def save_to_pkl(self,filename=None):
		if filename == None:
			filename=self.fm.data_name+'.pkl'

		with open(os.path.join(self.fm.ann_dir,filename), 'wb') as f:
			pickle.dump(self, f)

	def save_to_json(self,filename=None):
		if filename == None:
			filename = self.fm.data_name+'.json'

		with open(os.path.join(self.fm.ann_dir,filename), 'w') as f:
			json.dump(self.to_dict(), f,ensure_ascii=False, indent=4)

	def save_image(self):
		for file in self.fish_files:
			file.save_image()


class Fish_File(object):
	def __init__(self):
		# data --> unity_data
		self.data = None
		self.fm = None

		self.filename = None
		self.cycle = None
		self.frame = None
		self.camera = OpenCV_Camera()

		self.img_path = None
		self.fish = []

	def load_from_unity(self,data,fm):
		self.data = data
		self.fm = fm

		self.filename = data.name 
		self.cycle = int(data.cycle)
		self.frame = int(data.frame)

		# self._zero_index_error_fix()

		cam_info = utils.get_cam_info(self.data.cam_transform)
		self.camera.load_from_unity(data,cam_info)

		img_filename = str(self.cycle).zfill(4)+'_'+str(self.frame).zfill(4)+'.jpg'
		self.img_path = os.path.join(self.fm.img_dir,img_filename)

	def _zero_index_error_fix(self):
		# somehow cycle 0 frame 0 is always a blank black image, we removed it and make frame 1 index = 0
		if self.cycle == 0 :
			self.frame = self.frame - 1

	def convert_to_opencv(self):
		
		cam_info = utils.get_cam_info(self.data.cam_transform)

		corners = utils.get_3d_corner(self.data.ann_3d)
		corners = [utils.convert_to_cam_coord(corner,cam_info) for corner in corners] 
		corners = [utils.convert_to_opencv_coord(corner) for corner in corners]

		for i in range(len(corners)):
			obj_id = self.data.ann_2d['id'].iloc[i]
			
			ann_2d = self.data.ann_2d
			ann_3d = self.data.ann_3d
			visibility = self.data.visibility
			cam_dist_rot = self.data.cam_dist_rot

			ann_2d = ann_2d.loc[ann_2d['id'] == obj_id].iloc[0]
			ann_3d = ann_3d.loc[ann_3d['id'] == obj_id].iloc[0]
			cam_dist_rot = cam_dist_rot.loc[cam_dist_rot['id'] == obj_id].iloc[0]
			visibility = visibility.loc[visibility['id'] == obj_id].iloc[0]

			corner = corners[i]
			fish = OpenCV_Object()
			fish.id = int(re.findall('\d+$',ann_2d['id'])[0])

			h,w,_ = cv2.imread(self.data.img_path).shape
			bbox = utils.get_2d_box(ann_2d,h,w)

			# remove fish
			if bbox == None or self.frame == 0:
				continue

			xmin,ymin,xmax,ymax = bbox
			bbox_area = np.sqrt(abs(xmax-xmin)*abs(ymax-ymin))
			# print(bbox_area)

			# remove fish
			if bbox_area < BBOX_2D_AREA_THR_LOWER:
				continue
			if bbox_area > BBOX_2D_AREA_THR_UPPER:
				continue

			fish.xmin,fish.ymin,fish.xmax,fish.ymax = bbox
			fish.x,fish.y,fish.z = utils.get_xyz(corner)
			fish.h = ann_3d['Height']
			fish.w = ann_3d['Width']
			fish.l = ann_3d['Length']

			fish.rx = cam_dist_rot['rel_theta_x']
			fish.ry = cam_dist_rot['rel_theta_y'] 
			fish.rz = cam_dist_rot['rel_theta_z'] 

			fish.alphax = cam_dist_rot['l_theta_x']
			fish.alphay = cam_dist_rot['l_theta_y']

			fish.alpha = utils.get_alpha(fish.x,fish.z,0,0)

			# remove fish
			if fish.z < Z_THR_LOWER:
				continue
			if fish.z > Z_THR_UPPER	:
				continue
			if visibility['pct_screen_covered'] < VISIBILITY_THR_LOWER:
				continue
			if visibility['pct_screen_covered'] > VISIBILITY_THR_UPPER:
				continue
			if visibility['visibility_estimate'] < FOG_VIS_THR:
				continue
				
			fish.obj_transform()
			self.fish.append(fish)

		# sort the fish based on id
		self.fish.sort(key=lambda x: x.id, reverse=False)

	def to_dict(self):
		file_dict = {'filename' : self.filename,
				'cycle':self.cycle,
				'frame':self.frame,
				'img_path':self.img_path,
				'camera':self.camera.to_dict(),
				'fish':[]}

		for fish in self.fish:
			file_dict['fish'].append(fish.to_dict())

		return file_dict

	def _image_transform(self,img):
		img = cv2.resize(img,(int(IMAGE_SIZE*RESIZE_FACTOR),int(IMAGE_SIZE*RESIZE_FACTOR)))
		return img

	def save_image(self):
		img = cv2.imread(self.data.img_path)
		img = self._image_transform(img)
		cv2.imwrite(self.img_path,img)

class OpenCV_Camera(object):
	def __init__(self):
		self.cam_id = None

		self.x = None
		self.y = None
		self.z = None

		self.rx = None
		self.ry = None
		self.rz = None

		self.focal_length = None

		self.extrinsic = None
		self.intrinsic = None

	def load_from_unity(self,data,cam_info):
		self.cam_id = data.cam_transform.iloc[0]['name']

		self.x = cam_info['x']
		self.y = cam_info['y']
		self.z = cam_info['z']

		self.rx = cam_info['rx']
		self.ry = cam_info['ry']
		self.rz = cam_info['rz']

		self.focal_length = data.cam_info['pixelLength']*RESIZE_FACTOR #image resized to 1024

		# h,w,_ = cv2.imread(data.img_path).shape
		self.set_intrinsic((IMAGE_SIZE*RESIZE_FACTOR),(IMAGE_SIZE*RESIZE_FACTOR))
		self.set_extrinsic_to_identity()
		self.set_to_origin()

	def set_to_origin(self):
		self.x = 0
		self.y = 0
		self.z = 0

		self.rx = 0
		self.ry = 0 
		self.rz = 0	

	def set_intrinsic(self,img_w,img_h):
		intrinsic = np.eye(3)

		intrinsic[0,0] = self.focal_length
		intrinsic[1,1] = self.focal_length

		intrinsic[0,2] = int(img_w/2)
		intrinsic[1,2] = int(img_h/2)

		self.intrinsic = intrinsic

	def set_extrinsic_from_euler(self):
		extrinsic = np.eye(4)
		extrinsic = extrinsic[:,:-1]

		extrinsic[:3,:3] = R.from_euler('xyz', [self.rx, self,ry, self.rz], degrees=True).as_matrix()
		extrinsic[0,-1] = self.x
		extrinsic[1,-1] = self.y
		extrinsic[2,-1] = self.z

		self.extrinsic = extrinsic

	def set_extrinsic_to_identity(self):
		extrinsic = np.eye(4)
		self.extrinsic = extrinsic[:,:-1]

	def to_dict(self):
		cam = {
			'cam_id' : self.cam_id,
			'x'	: self.x,
			'y' : self.y,
			'z'	: self.z,
			'rx': self.rx,
			'ry': self.ry,
			'rz': self.rz,
			'focal_length' : self.focal_length,
			'intrinsic' : self.intrinsic.tolist(),
			'extrinsic' : self.extrinsic.tolist()
		}

		return cam

class OpenCV_Object(object):
	def __init__(self):
		# id
		self.id = None

		# 2d bbox
		self.xmin = None
		self.ymin = None
		self.xmax = None
		self.ymax = None

		# 3d bbox
		self.x = None
		self.y = None
		self.z = None

		# 3d dimension
		self.h = None
		self.w = None
		self.l = None

		# 3d rotation
		self.rx = None
		self.ry = None
		self.rz = None

		# Local Orient
		self.alphax = None
		self.alphay = None

	def obj_transform(self):
		self.xmin = self.xmin*RESIZE_FACTOR
		self.ymin = self.ymin*RESIZE_FACTOR
		self.xmax = self.xmax*RESIZE_FACTOR
		self.ymax = self.ymax*RESIZE_FACTOR

	def to_dict(self):
		fish = {
			'id' : self.id,

			'xmin' : self.xmin,
			'ymin' : self.ymin,
			'xmax' : self.xmax,
			'ymax' : self.ymax,

			'x' : self.x,
			'y' : self.y,
			'z' : self.z,

			'h' : self.h,
			'w' : self.w,
			'l' : self.l,

			'rx' : self.rx,
			'ry' : self.ry,
			'rz' : self.rz,
			
			'alpha' : self.alphay
		}
		return fish





	



