# -*- coding: utf-8 -*-
# @Author: Muhammad Alfin N
# @Date:   2022-03-07 20:21:56
# @Last Modified by:   Invalid macro definition.




# @Last Modified time: 2022-03-08 21:09:03

from lib.dataset.Testing_Dataset import Testing_Dataset

def load_testing_dataset(dataset_dir,focal_length,data_num=None):
	Dataset = Testing_Dataset(dataset_dir,focal_length,data_num)

	return Dataset