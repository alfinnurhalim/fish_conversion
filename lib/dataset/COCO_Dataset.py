# -*- coding: utf-8 -*-
# @Author: Muhammad Alfin N
# @Date:   2022-03-07 20:21:56
# @Last Modified by:   Invalid macro definition.




# @Last Modified time: 2022-03-08 21:09:03


import cv2
import os
import re
import json
import pickle

import pandas as pd 
import lib.dataloader.utils as utils

from tqdm import tqdm

class COCO_Dataset(object):
	def __init__(self):
		self.data_dir = None
		self.coco_files = []
		self.categories = [{'id':1,'name':'Fish'}]

	def load_from_opencv(self,Dataset,folder_manager):
		print('converting to COCO format')
		self.fm = folder_manager
		self.data_dir = Dataset.data_dir

		for fish_file in tqdm(Dataset.fish_files):
			coco_file = COCO_File()
			coco_file.load_from_fish_file(fish_file,self.fm)
			self.coco_files.append(coco_file)

	def to_dict(self,tag='tracking'):
		coco = {
			'categories':self.categories,
			'images' : [],
			'videos' : [],
			'annotations':[]
		}

		for file in self.coco_files:
			coco['images'].append(file.to_dict(tag='image'))
			for coco_obj in file.COCO_objects:
				coco['annotations'].append(coco_obj.to_dict())

		if tag=='detection':
			del coco['annotations']
		return coco

	def save_to_pkl(self,filename=None):
		if filename == None:
			filename=self.fm.data_name+'.pkl'

		with open(os.path.join(self.fm.ann_dir,filename), 'wb') as f:
			pickle.dump(self, f)

	def save_to_json(self,filename=None,tag='tracking'):
		if filename == None:
			filename = self.fm.data_name+'.json'

		with open(os.path.join(self.fm.ann_dir,filename), 'w') as f:
			json.dump(self.to_dict(tag), f,ensure_ascii=False, indent=2)

class COCO_File(object):
	def __init__(self):
		self.data = None
		self.fm = None

		self.video_id = None
		self.index = None
		self.camera = COCO_Camera()

		self.img_path = None
		self.height = None
		self.width = None

		self.first_frame = False

		self.COCO_objects = []

	def load_from_fish_file(self,data,fm):
		self.fm = fm 

		self.data = data
		self.video_id = int(data.cycle)
		self.index = int(data.frame)

		self.camera.load_from_opencv_camera(data.camera)

		img_filename = str(self.video_id).zfill(4)+'_'+str(self.index).zfill(4)+'.jpg'
		self.img_path = os.path.join(self.fm.img_dir,img_filename)
		
		h,w,_ = cv2.imread(data.img_path).shape
		self.height = h
		self.width = w
		
		self.first_frame = True if self.index == 0 else False

		self.copy_image()
		self.convert_to_COCO()

	def convert_to_COCO(self):
		for i,fish in enumerate(self.data.fish):
			coco = COCO_Object()
			coco.load_from_fish_object(index=1,data=fish,coco_file=self)

			self.COCO_objects.append(coco)


	def to_dict(self,tag='image'):
		image = {
			'file_name':self.img_path,
			'cali': self.camera.cali,
			'pose': {
				'rotation':self.camera.rotation,
				'position':self.camera.position
				},
			'height':self.height,
			'width' :self.width,
			'fov'	:self.camera.fov,
			'near_clip':self.camera.near_clip,
			'id'	:self.index,
			'video_id':self.video_id,
			'index'	:self.index,
			'first_frame':self.first_frame
		}

		videos = {
			'id':self.video_id,
			'name':str(self.video_id).zfill(4),
			'n_frames':0
		}
		annotations = []
		for coco in self.COCO_objects:
			annotations.append(coco.to_dict())

		if tag=='image':
			return image
		elif tag == 'annotations':
			return annotations
		elif tag == 'videos':
			return videos
		else:
			return image

	def _image_transform(self,img):
		img = cv2.resize(img,(1024,1024))
		return img

	def copy_image(self):
		img = cv2.imread(self.data.img_path)
		img = self._image_transform(img)

		cv2.imwrite(self.img_path,img)

class COCO_Camera(object):
	def __init__(self):
		self.cali = None

		self.position = []
		self.rotation = []

		self.fov = 60
		self.near_clip = 0.15

	def load_from_opencv_camera(self,camera):
		self.cali = camera.intrinsic.tolist()

		self.position = [camera.x,camera.y,camera.z]
		self.rotation = [camera.rx,camera.ry,camera.rz]

class COCO_Object(object):
	def __init__(self):
		# id
		self.id = None
		self.image_id = None

		self.category_id = None
		self.instance_id = None

		# 3d rotation
		self.ry = None
		self.alpha = None

		# 3d bbox
		self.dimension = []
		self.translation = []

		self.is_occluded = 0
		self.is_truncated = 0

		# 2d bbox
		self.center_2d = []
		self.delta_2d = []
		self.bbox = []
		self.area = None

		self.iscrowd = False
		self.ignore = False

		self.segmentation = []

	def load_from_fish_object(self,index,data,coco_file):
		self.id = index
		self.image_id = coco_file.index

		self.category_id = 1
		self.instance_id = data.id

		self.ry = data.ry
		self.alpha = data.alpha

		self.dimension = [data.w,data.h,data.l]
		self.translation = [data.x,data.y,data.z]

		h_2d = abs(int(data.ymax - data.ymin))
		w_2d = abs(int(data.xmax - data.xmin))
		self.center_2d = [data.xmin + w_2d,data.ymin+h_2d]
		self.delta_2d = [
						self.center_2d[0] - (data.xmin + data.xmax) / 2.0,
						self.center_2d[1] - (data.ymin + data.ymax) / 2.0
						]

		self.bbox = [data.xmin,data.ymin,w_2d,h_2d]
		self.area = w_2d*h_2d

		self.segmentation = [data.xmin,data.ymin,
							data.xmin,data.ymax,
							data.xmax,data.ymax,
							data.xmax,data.ymin]

	def to_dict(self):
		fish = {
			'id' : self.id,
			'image_id': self.image_id,
			'category_id':self.category_id,
			'instance_id':self.instance_id,

			'alpha'	: self.alpha,
			'roty': self.ry,

			'dimension'	: self.dimension,
			'translation': self.translation,

			'is_occluded':self.is_occluded,
			'is_truncated':self.is_truncated,

			'center_2d'	: self.center_2d,
			'delta_2d'	: self.delta_2d,
			'bbox'		: self.bbox,
			'area'		: self.area,
			'iscrowd'	: self.iscrowd,
			'ignore'	: self.ignore,
			'segmentation': [self.segmentation]
		}
		return fish
		
