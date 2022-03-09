# -*- coding: utf-8 -*-
# @Author: Invalid macro definition.
# @Date:   2022-03-08 17:42:15
# @Last Modified by:   Invalid macro definition.
# @Last Modified time: 2022-03-08 17:42:27

import numpy as np
from scipy.spatial.transform import Rotation as R

def get_3d_corner(df):
	df = df[df.columns[7:31]]

	res = []
	for idx in range(len(df)):
		cx = []
		cy = []
		cz = []

		for i in range(len(df.iloc[idx].values)):
			val = df.iloc[idx].values[i]
			if i%3 == 0:
				cx.append(val)
			elif i%3 == 1:
				cy.append(val)
			elif i%3 == 2:
				cz.append(val)
		        
		data = np.array([cx,cy,cz])

		res.append(data)

	return res

def get_cam_info(df):
	df = df.iloc[0]

	res = {
	    'x':df['global_pos_x'],
	    'y':df['global_pos_y'],
	    'z':df['global_pos_z'],
	    'rx':0,
	    'ry':45,
	    'rz':0
	    # 'rx':df['EulerAngle_x'],
	    # 'ry':df['EulerAngle_y'],
	    # 'rz':df['EulerAngle_z'],
	}

	return res

def convert_to_cam_coord(data,cam_info):
	# translate with cam position
	data[0] = data[0] - cam_info['x']
	data[1] = data[1] - cam_info['y']
	data[2] = data[2] - cam_info['z']

	# rotate CCW
	rx = -cam_info['rx']
	ry = -cam_info['ry']
	rz = -cam_info['rz']

	r = R.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()

	data = np.dot(r,data)

	return data

def convert_to_opencv_coord(data):
	# Unity To OpenCV Coord
	# UNITY -----> OPENCV
	# x = x
	# y = -y
	# z = z

	data[0] = data[0]
	data[1] = data[1]*-1
	data[2] = data[2]

	return data