# -*- coding: utf-8 -*-
# @Author: Muhammad Alfin N
# @Date:   2022-03-07 20:21:56
# @Last Modified by:   Invalid macro definition.




# @Last Modified time: 2022-03-08 21:09:03

from lib.dataset.Unity_Dataset import Unity_Dataset

def load_unity_dataset(dataset_dir,data_num=None):
	Dataset = Unity_Dataset(dataset_dir,data_num)

	return Dataset