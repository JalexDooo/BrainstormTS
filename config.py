import torch as t
import warnings

class DefaultConfig(object):
	env = 'default'
	vis_port = 8097
	model = 'UNet3D'

	train_root_path = '/home/sunjindong/dataset/MICCAI_BraTS17_Data_Training_IPP/MICCAI_BraTS17_Data_Training'
	val_root_path = '/home/sunjindong/dataset/MICCAI_BraTS17_Data_Validation_IPP/Brats17ValidationData'

	test_img_path = ''
	test_images = './test_images/'

	
	load_model_path = None
	batch_size = 2
	use_gpu = True
	num_workers = 0
	print_freq = 20

	max_epoch = 2
	lr = 0.001
	lr_decay = 0.95
	weight_decay = 1e-4

	def _parse(self, kwargs):
		"""
		update config
		"""
		for k, v in kwargs.items():
			if not hasattr(self, k):
				warnings.warn("Warning: opt has not attribute %s" %k)
			setattr(self, k, v)
		
		# print('user config:')
		# for k, v in self.__class__.__dict__.items():
		# 	if not k.startswith('_'):
		# 		print(k, getattr(self, k))

opt = DefaultConfig()
