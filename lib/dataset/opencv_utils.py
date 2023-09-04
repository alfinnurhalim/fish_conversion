# -*- coding: utf-8 -*-
# @Author: Invalid macro definition.
# @Date:   2022-03-08 17:42:15
# @Last Modified by:   Invalid macro definition.

# @Last Modified time: 2022-03-08 17:42:27


import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def calc_theta_ray(width, center, proj_matrix,is_y=False):
    fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
    

    center = width - center if is_y else center
    
    dx = center - (width / 2)

    mult = 1
    if dx < 0:
        mult = -1
    dx = abs(dx)
    angle = np.arctan( (2*dx*np.tan(fovx/2)) / width )
    angle = angle * mult

    angle = fovx/2 - angle
    return angle

def iou(bboxA, bboxB):
	"""
	Similar to the above function, this one computes the intersection over union
	between two 2-d boxes. The difference with this function is that it accepts
	bounding boxes in the form [xmin, ymin, XMAX, YMAX].
	Attributes:
	    bboxA (list): defined by 4 values: [xmin, ymin, XMAX, YMAX].
	    bboxB (list): defined by 4 values: [xmin, ymin, XMAX, YMAX].
	Returns:
	    IOU (float): a value between 0-1 representing how much these boxes overlap.
	"""
	xminA, yminA, xmaxA, ymaxA = bboxA
	widthA = xmaxA - xminA
	heightA = ymaxA - yminA
	areaA = widthA * heightA

	xminB, yminB, xmaxB, ymaxB = bboxB
	widthB = xmaxB - xminB
	heightB = ymaxB - yminB
	areaB = widthB * heightB

	xA = max(xminA, xminB)
	yA = max(yminA, yminB)
	xB = min(xmaxA, xmaxB)
	yB = min(ymaxA, ymaxB)

	W = xB - xA
	H = yB - yA 

	if min(W, H) < 0:
		return 0

	intersect = W * H
	union = areaA + areaB - intersect

	if union == 0:
		return 1

	iou = areaB / union

	return iou

def get_3d_corner(df):
	# print(df.columns)
	columns = ['p0_world_x','p0_world_y','p0_world_z','p1_world_x','p1_world_y','p1_world_z',
			'p2_world_x','p2_world_y','p2_world_z','p3_world_x','p3_world_y','p3_world_z',
			'p4_world_x','p4_world_y','p4_world_z','p5_world_x','p5_world_y','p5_world_z',
			'p6_world_x','p6_world_y','p6_world_z','p7_world_x','p7_world_y','p7_world_z']

	df = df[columns]

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

	# print(cam_info['x'])
	# rotate CCW
	# print(cam_info['rx'])
	rx = -cam_info['rx']
	ry = -cam_info['ry']
	rz = -cam_info['rz']

	r = R.from_euler('zxy', [rz, rx, ry], degrees=True).as_matrix()
	# r = R.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
	
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

def get_2d_box(data,img_h,img_w,ann_center,ratio=0.8):
	raw_data = data.copy()
	columns = ['p0_x','p0_y','p1_x','p1_y','p2_x','p2_y','p3_x','p3_y']
	
	data = data[columns]
	corner_thr = 0.8

	data=data.values

	x = []
	y = []

	for i in range(len(data)):
		if i%2==0:
			x.append(data[i])
		else:
			y.append(img_h - data[i])

	xmin = int(min(x))
	xmax = int(max(x))

	ymin = int(min(y))
	ymax = int(max(y))

	h = abs(ymax - ymin) * ratio
	l = abs(xmax - xmin) * ratio

	cx = int(ann_center['screen_center_x'])
	cy = int(img_h - ann_center['screen_center_y'])

	# h = raw_data['Height']
	# l = raw_data['Width']

	# cx = raw_data['2D_Cx']
	# cy = img_h - raw_data['2D_Cy']

	# xmin = int(cx - l/2)
	# xmax = int(cx + l/2)
	# ymin = int(cy - h/2)
	# ymax = int(cy + l/2)
	# if xmin <0 :
	# 	if abs(xmin) > l*corner_thr:
	# 		return None

	# if xmax > img_w :
	# 	if abs(img_w-xmax) > l*corner_thr:
	# 		return None

	# if ymin <0 :
	# 	if abs(ymin) > h*corner_thr:
	# 		return None

	# if ymax > img_h :
	# 	if abs(img_h-xmax) > h*corner_thr:
	# 		return None

	if (xmin <0 and xmax >img_w) and (ymin < 0 and ymax > img_h):
		return None

	# # to remove all fish that's in the corner
	# if (xmin <0 or xmax >img_w) or (ymin < 0 or ymax > img_h):
	# 	return None

	if xmin < 0:
		xmin = 0
	if ymin < 0:
		ymin = 0
	if xmax > img_w:
		xmax = img_w
	if ymax > img_h:
		ymax = img_h
	
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

def project_3d(proj,points):
	if len(points.shape)==1:
		points = np.expand_dims(points, axis=1)
	P = np.zeros((3,4))
	P[:,:-1] = proj
	conn = np.concatenate((points.T, np.ones((points.shape[1], 1))), axis=1)
	corners_img_before = np.matmul(conn, P.T)
	corners_img = corners_img_before[:, :2] / corners_img_before[:, 2][:, None]

	return corners_img