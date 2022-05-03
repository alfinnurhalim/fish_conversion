# -*- coding: utf-8 -*-
# @Author: Invalid macro definition.
# @Date:   2022-03-08 17:42:15
# @Last Modified by:   Invalid macro definition.

# @Last Modified time: 2022-03-08 17:42:27


import numpy as np
import math
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
	    'rx':df['EulerAngle_x'],
	    'ry':df['EulerAngle_y'],
	    'rz':df['EulerAngle_z'],
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

def get_2d_box(data,h,w):

	data=data.values[1:]

	x = []
	y = []

	for i in range(len(data)):
		if i%2==0:
			x.append(data[i])
		else:
			y.append(h - data[i])

	xmin = int(min(x))
	xmax = int(max(x))

	ymin = int(min(y))
	ymax = int(max(y))

	if (xmin <0 and xmax >w) and (ymin < 0 and ymax > h):
		return None

	if xmin < 0:
		xmin = 0
	if ymin < 0:
		ymin = 0
	if xmax > w:
		xmax = w
	if ymax > h:
		ymax = h
	
	return xmin,ymin,xmax,ymax

def get_middle(pts):
	res = min(pts) + (max(pts)-min(pts))/2
	return res

def get_xyz(corner):
	x = round(get_middle(corner[0]),2)
	z = round(get_middle(corner[2]),2)
	y = round(get_middle(corner[1]),2)

	return (x,y,z)

def get_whl(corner):
	# w = max(corner[0]) - min(corner[0])
	# h = max(corner[1]) - min(corner[1])
	# l = max(corner[2]) - min(corner[2])
	
	pts = [x for x in zip(corner[0],corner[1],corner[2])]
	w = round(calculate_distance(pts[0],pts[1]),2)
	h = round(calculate_distance(pts[0],pts[4]),2)
	l = round(calculate_distance(pts[0],pts[2]),2)
	return (w,h,l)

def calculate_distance(pts_0,pts_1):
	x1,y1,z1 = pts_0
	x2,y2,z2 = pts_1
	dis = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

	return dis

def get_alpha(x,z,cx,cz):
	dx = (x-cx)
	dz = (z-cz)
	
	if dx!=0:
		yaw = -math.atan(dz/dx)
	else:
		if dz<0:
			yaw = -np.pi/2
		else:
			yaw = np.pi/2
	yaw = yaw + np.pi/2
	return round(yaw,2)

def get_yaw(ry,cam_ry):
	yaw = ry 
	yaw = math.radians(yaw)

	return yaw