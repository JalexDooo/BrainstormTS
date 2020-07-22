import torch as t
import warnings

class DefaultConfig(object):
	env = 'default'
	vis_port = 8097
	model = 'UNet3D'

	train_root_path = '/home/sunjindong/dataset/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training'
	val_root_path = '/home/sunjindong/dataset/MICCAI_BraTS_2019_Data_Validation/MICCAI_BraTS_2019_Data_Validation'
	tttt = '/home/test/dateset/MICCAI_BraTS_2019_Data_Validation/MICCAI_BraTS_2019_Data_Validation'
	ttttt = '/home/test/dataset/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training'
	# val_root_path = '/home/sunjindong/dataset/MICCAI_BraTS17_Data_Validation_IPP/Brats17ValidationData'

	# aneu_path = '/Users/juntysun/Downloads/数据集/动脉瘤数据'
	aneu_path = '/home/aneu/dataset/dongmai'
	aneu_val_path = '/home/aneu/dataset/final_val_dataset'
	aneu_output_path = '/home/aneu/Brainstorm/final_val'

	local_root_path = '/Users/juntysun/Downloads/数据集/MICCAI_BraTS_2019_Data_Training'

	test_img_path = ''
	test_images = './test_images/'

	task = 'WT'

	predict_nibable_path = './predict_nibable'

	load_model_path = None
	batch_size = 4
	use_gpu = True
	num_workers = 0
	print_freq = 20

	max_epoch = 2
	random_epoch = 4
	lr = 0.001
	lr_decay = 0.99
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
