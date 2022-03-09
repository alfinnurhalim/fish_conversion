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

	def load_from_opencv(self,Dataset):
		print('converting to COCO format')
		self.data_dir = Dataset.data_dir

		for fish_file in tqdm(Dataset.fish_files):
			coco_file = COCO_File()
			coco_file.load_from_fish_file(fish_file)
			self.coco_files.append(coco_file)

	def to_dict(self):
		coco = {
			'categories':self.categories,
			'images' : []
		}

		for file in self.coco_files:
			coco['images'].append(file.to_dict())

		return coco

	def save_to_pkl(self,path):
		with open(path, 'wb') as f:
			pickle.dump(self, f)

	def save_to_json(self,path):
		with open(path, 'w') as f:
			json.dump(self.to_dict(), f,ensure_ascii=False, indent=2)

class COCO_File(object):
	def __init__(self):
		self.video_id = None
		self.index = None
		self.camera = COCO_Camera()

		self.img_path = None
		self.height = None
		self.width = None

		self.first_frame = False

	def load_from_fish_file(self,data):
		self.data = data
		self.video_id = int(data.cycle)
		self.index = int(data.frame)

		self.camera.load_from_opencv_camera(data.camera)

		self.img_path = data.img_path
		
		h,w,_ = cv2.imread(data.img_path).shape
		self.height = h
		self.width = w
		
		self.first_frame = True if self.index == 0 else False

	def to_dict(self):
		file = {
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

		return file

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
		
