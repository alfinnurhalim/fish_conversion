# -*- coding: utf-8 -*-
# @Author: Muhammad Alfin N
# @Date:   2022-03-07 20:53:05
# @Last Modified by:   Muhammad Alfin N


# @Last Modified time: 2022-03-08 10:47:11

import os
import re

def get_filenames(dataset_dir,data_num):
	r = re.compile('.*all.*')

	filenames = os.listdir(dataset_dir)
	filenames = list(filter(r.match,filenames))

	dataset = [[re.findall('.*(?=_all)',filename)[0],add_zero_padding(re.findall('.*(?=_all)',filename)[0])] for filename in filenames]
	dataset = sorted(dataset,key = lambda x: x[1])[1:data_num]
	dataset = [x[0] for x in dataset]

	return dataset

def extract_file_info(filename):
	cycle_id = re.findall('(?<=cycle_).*(?=_frame)',filename)[0].zfill(4)
	frame_id = re.findall('(?<=frame_).*(?=_cam)',filename)[0].zfill(4)
	cam_id = re.findall('(?<=_cam).*',filename)[0].zfill(4)

	return cycle_id,frame_id,cam_id

def add_zero_padding(filename):
	
	cycle_id,frame_id,cam_id = extract_file_info(filename)
	new_filename = "cycle_{}_frame_{}_cam_{}".format(cycle_id,frame_id,cam_id)

	return new_filename

# def get_2d_ann_path(filename):
