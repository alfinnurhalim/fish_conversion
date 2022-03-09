# -*- coding: utf-8 -*-
# @Author: Muhammad Alfin N
# @Date:   2022-03-08 10:06:10
# @Last Modified by:   Muhammad Alfin N

# @Last Modified time: 2022-03-08 17:23:00

from lib.dataloader.Unity_dataloader import Unity_Dataloader

class OpenCV_Dataset(object):
	def __init__(self):
		pass

	def init_from_Unity(self,data,cam_info=None):
		cam_info = get_cam_info(data.cam_transform)
		
		corners = get_3d_corner(data.ann_3d)
		corners = [convert_to_cam_coord(corner,cam_info) for corner in corners] 
		corners = [convert_to_opencv_coord(corner) for corner in corners]

class Fish_File(object):
	def __init__(self):
		pass

class OpenCV_Camera(object):
	def __init__(self,cam_id,cam_info):
		self.cam_id = None
		self.extrinsic = None
		self.intrinsic = None


class Fish_data(object):
	def __init__(self):
		self.id = None

		self.xmin = None
		self.ymin = None
		self.xmax = None
		self.ymax = None

		self.x = None
		self.y = None
		self.z = None

		self.h = None
		self.w = None
		self.l = None

		self.ry = None

	



