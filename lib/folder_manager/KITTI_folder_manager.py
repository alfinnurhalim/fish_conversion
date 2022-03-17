import os
import shutil

class KITTI_folder_manager(object):
	def __init__(self,data_dir,name,tag='',split='training'):
		self.data_dir = data_dir
		self.data_name = name
		self.out_dir = os.path.join('data',self.data_name,'KITTI',tag,split)

		self.ann_dir = os.path.join(self.out_dir,'label_2')
		self.img_dir = os.path.join(self.out_dir,'image_2')
		self.calib_dir = os.path.join(self.out_dir,'calib')
	
	def create_folder(self):
		if os.path.isdir(self.out_dir):
			shutil.rmtree(self.out_dir)

		os.makedirs(self.out_dir)
		os.makedirs(self.img_dir)
		os.makedirs(self.ann_dir)
		os.makedirs(self.calib_dir)

	def make_dir(self,path):
		if not os.path.isdir(path):
			os.makedirs(path)
			
