# -*- coding: utf-8 -*-
# @Author: Muhammad Alfin N
# @Date:   2022-03-07 20:21:56
# @Last Modified by:   Invalid macro definition.




# @Last Modified time: 2022-03-08 21:09:03



import os
import re
import json
import logging

import pandas as pd 
import lib.dataloader.utils as utils

from tqdm import tqdm


class Unity_Dataset(object):
	def __init__(self,data_dir,data_num=None):
		self.data_dir = data_dir

		filenames = utils.get_filenames(self.data_dir)
		if data_num != None:
			filenames = filenames[:data_num]

		self.unity_files = self.load_unity_file(filenames)
		self.cam_info = json.load(open(os.path.join(data_dir,'cycle_0_env_params.json')))

	def load_unity_file(self,filenames):
		print('Loading Dataset')

		files = []
		for filename in tqdm(filenames):
			files.append(Unity_File(self.data_dir,filename))

		return files

class Unity_File(object):
	def __init__(self,data_dir,filename):
		self.data_dir = data_dir
		self.name = filename
		self.cycle, self.frame, self.cam_id = utils.extract_file_info(filename)

		self.img_path = self.get_img_path(filename)
		self.ann_2d = self.get_ann_2d(filename)
		self.ann_3d = self.get_ann_3d(filename)

		self.cam_transform = self.get_cam_transform(filename)
		self.visibility = self.get_visibility(filename)
		self.cam_info = self.get_cam_info(filename)

	def get_img_path(self,filename):
		path = os.path.join(self.data_dir,filename+'_all.jpeg')
		return path if os.path.exists(path) else None

	def get_ann_2d(self,filename):
		path = os.path.join(self.data_dir,filename+'_Box2D.csv')
		data = pd.read_csv(path) if os.path.exists(path) else None
		return data

	def get_ann_3d(self,filename):
		path = os.path.join(self.data_dir,filename+'_Box3D.csv')
		data = pd.read_csv(path) if os.path.exists(path) else None

		# UPDATE change the order
		data = data[['id','camera rel origin_x','camera rel origin_y','camera rel origin_z','pitch','yaw',
					'roll','p0_world_x','p0_world_y','p0_world_z','p1_world_x','p1_world_y','p1_world_z',
					'p2_world_x','p2_world_y','p2_world_z','p3_world_x','p3_world_y','p3_world_z','p4_world_x',
					'p4_world_y','p4_world_z','p5_world_x','p5_world_y','p5_world_z','p6_world_x','p6_world_y',
					'p6_world_z','p7_world_x','p7_world_y','p7_world_z','p0_screen_x','p0_screen_y','p1_screen_x',
					'p1_screen_y','p2_screen_x','p2_screen_y','p3_screen_x','p3_screen_y','p4_screen_x','p4_screen_y',
					'p5_screen_x','p5_screen_y','p6_screen_x','p6_screen_y','p7_screen_x','p7_screen_y','Head_world_x',
					'Head_world_y','Head_world_z']]
		return data

	def get_cam_transform(self,filename):
		path =  os.path.join(self.data_dir,filename[:-5]+'_camera_transform.csv')
		data = pd.read_csv(path) if os.path.exists(path) else None
		return data

	def get_visibility(self,filename):
		path = os.path.join(self.data_dir,filename+'_visibility.csv')
		data = pd.read_csv(path) if os.path.exists(path) else None
		return data

	def get_cam_info(self,filename):
		path = os.path.join(self.data_dir,'cycle_0_env_params.json')
		data = json.load(open(path))
		data = next(x for x in data['cameraParameters'] if x["name"] == 'cam'+self.cam_id)
		return data

	def print_info(self):
		print('data dir : ',self.data_dir)
		print('name : ',self.name)
		print('cycle : ',self.cycle)
		print('frame : ',self.frame)
		print('cam_id : ',self.cam_id)
		print('img : ',self.img_path)
		print('ann_2d : ',type(self.ann_2d))
		print('ann_3d : ',type(self.ann_3d))
		print('cam_transform : ',type(self.cam_transform))
		print('visibility : ',type(self.visibility))