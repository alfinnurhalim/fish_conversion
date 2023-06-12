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

IMG_SIZE = 512

class Unity_Dataset(object):
	def __init__(self,data_dir,data_num=None,fps=1):
		self.data_dir = data_dir
		self.fps = fps
		filenames = utils.get_filenames(self.data_dir)
		if data_num != None:
			filenames = filenames[:data_num]

		self.unity_files = self.load_unity_file(filenames)
		self.cam_info = json.load(open(os.path.join(data_dir,'cycle_0_env_params.json')))

	def load_unity_file(self,filenames):
		print('Loading Dataset')

		files = []
		for i,filename in enumerate(tqdm(filenames)):
			if i%self.fps != 0:
				continue
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
		self.ann_center = self.get_ann_center(filename)

		self.cam_transform = self.get_cam_transform(filename)
		self.visibility = self.get_visibility(filename)
		self.cam_info = self.get_cam_info(filename)
		self.cam_dist_rot = self.get_cam_dist_rot(filename)

	def get_img_path(self,filename):
		path = os.path.join(self.data_dir,filename+'_all.jpeg')
		return path if os.path.exists(path) else None

	def get_ann_2d(self,filename):
		path = os.path.join(self.data_dir,filename+'_Box2DOriented.csv')
		try:
			data = pd.read_csv(path) if os.path.exists(path) else None
		except:
			data = pd.DataFrame(columns=['id','p0_x','p0_y','p1_x','p1_y','p2_x','p2_y','p3_x','p3_y',
										'Width','Height','Angle','2D_Cx','2D_Cy'])
		return data

	def get_ann_center(self,filename):
		path = os.path.join(self.data_dir,filename+'_screen_pos.csv')
		try:
			data = pd.read_csv(path) if os.path.exists(path) else None
		except:
			data = pd.DataFrame(columns=['id','screen_pos_x','screen_pos_y'])
		return data

	def get_ann_3d(self,filename):
		path = os.path.join(self.data_dir,filename+'_Box3DOriented.csv')
		column_names = ['id','camera rel origin_x','camera rel origin_y','camera rel origin_z','pitch','yaw',
		'roll','Head_world_x','Head_world_y','Head_world_z','Top_world_x','Top_world_y','Top_world_z','Top_pixel_x',
		'Top_pxiel_y','Bot_world_x','Bot_world_y','Bot_world_z','Bot_pixel_x','Bot_pxiel_y','p0_world_x','p0_world_y',
		'p0_world_z','p1_world_x','p1_world_y','p1_world_z','p2_world_x','p2_world_y','p2_world_z','p3_world_x',
		'p3_world_y','p3_world_z','p4_world_x','p4_world_y','p4_world_z','p5_world_x','p5_world_y','p5_world_z',
		'p6_world_x','p6_world_y','p6_world_z','p7_world_x','p7_world_y','p7_world_z','p0_screen_x','p0_screen_y',
		'p1_screen_x','p1_screen_y','p2_screen_x','p2_screen_y','p3_screen_x','p3_screen_y','p4_screen_x','p4_screen_y',
		'p5_screen_x','p5_screen_y','p6_screen_x','p6_screen_y','p7_screen_x','p7_screen_y','Length','Height','Width',
		'center_X','center_Y','center_Z','CamRel_Length','CamRel_Height','CamRel_Width','CamRel_center_x','CamRel_center_y','CamRel_center_z']

		try:
			data = pd.read_csv(path) 

			# UPDATE change the order
			data = data[column_names]
		except Exception as e:
			print(e)
			data = pd.DataFrame(columns=column_names)

		# print(data.iloc[0]['Length']*data.iloc[0]['Width']*data.iloc[0]['Height']*1000)
		return data

	def get_cam_transform(self,filename):
		path =  os.path.join(self.data_dir,filename[:-6]+'_camera_transform.csv') #10 for camCenter, 5 rest. 16 fo camExtra2
		try:
			data = pd.read_csv(path) if os.path.exists(path) else None
		except:
			data = pd.DataFrame(columns=['name','global_pos_x','global_pos_y','global_pos_z','global_rot_x','global_rot_y',
										'global_rot_z','global_rot_w','EulerAngle_x','EulerAngle_y','EulerAngle_z'])
		return data

	def get_visibility(self,filename):
		path = os.path.join(self.data_dir,filename+'_visibility.csv')
		try:
			data = pd.read_csv(path)
		except:
			data = pd.DataFrame(columns=['id','pct_screen_covered','non_occluded_pixels','visibility_estimate'])
		return data

	def get_cam_info(self,filename):
		path = os.path.join(self.data_dir,'cycle_0_env_params.json')
		data = json.load(open(path))
		data = next(x for x in data['cameraParameters'] if (x["name"] == 'cam'+self.cam_id) and x['useThisCamera']==True)
		
		return data

	def get_cam_dist_rot(self,filename):
		path = os.path.join(self.data_dir,filename+'_cam_dist_rot.csv')
		data = pd.read_csv(path)
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