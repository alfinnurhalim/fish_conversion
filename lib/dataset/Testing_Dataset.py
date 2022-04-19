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


class Testing_Dataset(object):
	def __init__(self,data_dir,focal_length,data_num=None):
		self.data_dir = data_dir

		self.focal_length = focal_length
		self.testing_files = []
		video_dir = [os.path.join(data_dir,x) for x in os.listdir(data_dir)]

		for video_index,video in enumerate(video_dir):
			filenames = sorted(os.listdir(video))
			self.testing_files = self.testing_files + self.load_testing_file(video,filenames,video_index)

	def load_testing_file(self,video_dir,filenames,video_index):
		print('Loading Dataset')

		files = []
		for frame_id,filename in enumerate(tqdm(filenames)):
			files.append(Testing_File(video_dir,filename,video_index,frame_id,self.focal_length))

		return files

class Testing_File(object):
	def __init__(self,data_dir,filename,video_index,frame_id,focal_length):
		self.data_dir = data_dir

		self.name = filename
		self.cycle = video_index
		self.frame = frame_id
		self.cam_id = 0

		self.img_path = self.get_img_path(filename)
		self.ann_2d = self.get_ann_2d()
		self.ann_3d = self.get_ann_3d()

		self.cam_transform = self.get_cam_transform()
		self.visibility = self.get_visibility()
		self.cam_info = self.get_cam_info(focal_length)

	def get_img_path(self,filename):
		path = os.path.join(self.data_dir,filename)
		return path if os.path.exists(path) else None

	def get_ann_2d(self):
		column_names = ['id','p0_x','p0_y','p1_x','p1_y','p2_x','p2_y','p3_x','p3_y']
		data = pd.DataFrame(columns = column_names)
		return data

	def get_ann_3d(self):
		# UPDATE change the order
		column_names = ['id','camera rel origin_x','camera rel origin_y','camera rel origin_z','pitch','yaw',
					'roll','p0_world_x','p0_world_y','p0_world_z','p1_world_x','p1_world_y','p1_world_z',
					'p2_world_x','p2_world_y','p2_world_z','p3_world_x','p3_world_y','p3_world_z','p4_world_x',
					'p4_world_y','p4_world_z','p5_world_x','p5_world_y','p5_world_z','p6_world_x','p6_world_y',
					'p6_world_z','p7_world_x','p7_world_y','p7_world_z','p0_screen_x','p0_screen_y','p1_screen_x',
					'p1_screen_y','p2_screen_x','p2_screen_y','p3_screen_x','p3_screen_y','p4_screen_x','p4_screen_y',
					'p5_screen_x','p5_screen_y','p6_screen_x','p6_screen_y','p7_screen_x','p7_screen_y','Head_world_x',
					'Head_world_y','Head_world_z']
		data = pd.DataFrame(columns = column_names)
		return data

	def get_cam_transform(self):
		column_names = ['name','global_pos_x','global_pos_y','global_pos_z','global_rot_x','global_rot_y',
						'global_rot_z','global_rot_w','EulerAngle_x','EulerAngle_y','EulerAngle_z']

		data = [['cam1',0,0,0,0,0,0,0,0,0,0]]
		data = pd.DataFrame(data,columns = column_names)
		return data

	def get_visibility(self):
		column_names = ['id','pct_screen_covered','non_occluded_pixels','visibility_estimate']
		data = pd.DataFrame(columns = column_names)
		return data

	def get_cam_info(self,focal_length):
		data = {'pixelLength': focal_length}
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