import os

class COCO_folder_manager(object):
	def __init__(self,data_dir,name):
		self.data_dir = data_dir
		self.data_name = name
		self.out_dir = os.path.join('data/converted',self.data_name,'coco')

		self.ann_dir = self.out_dir
		self.img_dir = self.out_dir
		self.calib_dir = self.out_dir
		
	def create_folder(self):
		if not os.path.isdir(self.out_dir):
			os.makedirs(self.out_dir)

