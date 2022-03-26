# -*- coding: utf-8 -*-
# @Author: Muhammad Alfin N
# @Date:   2022-03-07 20:21:56
# @Last Modified by:   Invalid macro definition.




# @Last Modified time: 2022-03-08 21:09:03


import cv2
import numpy as np
import os
import re
import json
import pickle

import pandas as pd 
import lib.dataloader.utils as utils

from tqdm import tqdm

class KITTI_Dataset(object):
	def __init__(self):
		self.data_dir = None
		self.fm = None
		self.kitti_files = []
		self.cycle_list = None

	def load_from_opencv(self,Dataset,folder_manager):
		print('converting to KITTI format')
		self.fm = folder_manager
		self.data_dir = Dataset.data_dir

		for fish_file in tqdm(Dataset.fish_files):
			kitti_file = KITTI_File()
			kitti_file.load_from_fish_file(fish_file,self.fm)
			self.kitti_files.append(kitti_file)

		self.cycle_list = set([x.cycle for x in self.kitti_files])

	def save_to_pkl(self,filename=None):
		if filename == None:
			filename=self.fm.data_name+'.pkl'

		with open(os.path.join(self.fm.ann_dir,filename), 'wb') as f:
			pickle.dump(self, f)

	def save_to_json(self,filename=None,tag='tracking',output=False):
		if filename == None:
			filename = self.fm.data_name+'.json'

		with open(os.path.join(self.fm.ann_dir,filename), 'w') as f:
			json.dump(self.to_dict(tag,output=output), f,ensure_ascii=False, indent=2)

	def save_image(self,tag = 'detection'):
		if tag == 'tracking':
			for cycle in self.cycle_list:
				self.fm.make_dir(os.path.join(self.fm.img_dir,str(cycle).zfill(4)))

		for i,file in enumerate(self.kitti_files):
			if tag	== 'tracking':
				file.img_path = os.path.join(self.fm.img_dir,str(file.cycle).zfill(4),file.filename+'.jpg')
			else:
				file.img_path = os.path.join(self.fm.img_dir,str(i).zfill(6)+'.jpg')
			file.save_image()

	def save_camera(self,tag='detection'):
		if tag == 'tracking':
			file = self.kitti_files[0]
			for cycle in self.cycle_list:
				file.camera.save_camera(file.fm,str(cycle).zfill(4))
		else:
			for i,file in enumerate(self.kitti_files):
				file.camera.save_camera(file.fm,str(i).zfill(6))

	def save_ann(self,tag='detection'):
		if tag == 'tracking':
			for cycle in self.cycle_list:
				ann = []
				for file in self.kitti_files:
					if file.cycle == cycle:
						ann = ann + file.to_list(tag='tracking')
				pd.DataFrame(ann).to_csv(os.path.join(self.fm.ann_dir,str(cycle).zfill(4)+'.txt'),sep=' ',header=False,index=False)
		else:
			for i,file in enumerate(self.kitti_files):
				file.filename = str(i).zfill(6)
				file.save_ann()


class KITTI_File(object):
	def __init__(self):
		self.data = None
		self.fm = None

		self.filename = None
		self.cycle = None
		self.frame = None
		self.camera = KITTI_Camera()

		self.img_path = None
		self.KITTI_Objects = []

	def load_from_fish_file(self,data,fm):
		self.fm = fm 
		self.data = data

		self.cycle = int(data.cycle)
		self.frame = int(data.frame)
		self.filename = str(self.cycle).zfill(3)+str(self.frame).zfill(3)

		self.camera.load_from_opencv_camera(data.camera)

		img_filename = self.filename+'.jpg'
		self.img_path = os.path.join(self.fm.img_dir,img_filename)

		self.convert_to_KITTI()
		# self.KITTI_Objects = self.KITTI_Objects[5:10]

	def convert_to_KITTI(self):
		for i,fish in enumerate(self.data.fish):
			kitti = KITTI_Object()
			kitti.load_from_fish_object(data=fish)
			self.KITTI_Objects.append(kitti)

	def _image_transform(self,img):
		img = cv2.resize(img,(1024,1024))
		return img

	def save_image(self):
		img = cv2.imread(self.data.img_path)
		img = self._image_transform(img)

		cv2.imwrite(self.img_path,img)

	def to_list(self,tag='detection'):
		label = [x.to_list(tag,frame=self.frame) for x in self.KITTI_Objects]
		return label

	def save_ann(self):
		out_path = os.path.join(self.fm.ann_dir,self.filename+'.txt')

		label = [x.to_list() for x in self.KITTI_Objects]
		pd.DataFrame(label).to_csv(out_path,sep=' ',header=False,index=False)


class KITTI_Camera(object):
	def __init__(self):
		self.intrinsic = None
		self.extrinsic = None

	def load_from_opencv_camera(self,camera):
		self.intrinsic = camera.intrinsic.tolist()

		# convert to 4x3 mattrix
		self.intrinsic[0].append(0)
		self.intrinsic[1].append(0)
		self.intrinsic[2].append(0)

		self.extrinsic = np.eye(3)

	def save_camera(self,fm,filename):
		out_path = os.path.join(fm.calib_dir,filename+'.txt')

		intrinsic = ' '.join([str(int(x)) for x in list(np.array(self.intrinsic).flatten())])
		P0 = 'P0: ' + intrinsic
		P1 = 'P1: ' + intrinsic
		P2 = 'P2: ' + intrinsic
		P3 = 'P3: ' + intrinsic

		R0_rect = 'R0_rect: ' + ' '.join([str(int(x)) for x in list(self.extrinsic.flatten())])
		Tr_velo_to_cam = 'Tr_velo_to_cam: ' + ' '.join(['0' for i in range(12)])
		Tr_imu_to_velo = R0_rect

		cam = [P0,P1,P2,P3,R0_rect,Tr_velo_to_cam,Tr_imu_to_velo]

		with open(out_path,'w+') as f:
			for line in cam:
				f.write(line + '\n')

class KITTI_Object(object):
	def __init__(self):
		# id
		self.id = None

		# class
		self.type = None
		self.truncated = 0
		self.occluded = 0

		self.alpha = None

		# 2d bbox
		self.xmin = None
		self.ymin = None
		self.xmax = None
		self.ymax = None

		# 3d dimension
		self.h = None
		self.w = None
		self.l = None

		# 3d bbox
		self.x = None
		self.y = None
		self.z = None

		# 3d rotation
		self.ry = None

	def load_from_fish_object(self,data):
		self.id = data.id

		self.type = 'Car'
		self.alpha = data.alpha

		self.xmin = data.xmin
		self.ymin = data.ymin
		self.xmax = data.xmax
		self.ymax = data.ymax

		self.h = data.h 
		self.w = data.w 
		self.l = data.l 

		self.x = data.x
		self.y = data.y + self.h/2
		self.z = data.z 

		self.ry = data.ry

	def to_list(self,tag='detection',frame=0):
		if tag == 'tracking':
			return [frame,self.id,self.type,self.truncated,self.occluded,self.alpha,self.xmin,self.ymin,self.xmax,self.ymax,self.h,self.w,self.l,self.x,self.y,self.z,self.ry]
		else:
			return [self.type,self.truncated,self.occluded,self.alpha,self.xmin,self.ymin,self.xmax,self.ymax,self.h,self.w,self.l,self.x,self.y,self.z,self.ry]