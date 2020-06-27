import os
import sys
import glob
import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.image as Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import scipy.misc

from .datautils import *

module = ['flair', 't1', 't1ce', 't2']


class BraTS2017(Dataset):
	"""
		先做tumor core任务试试，打通数据与模型。
		tumor core：label中只有像素值4代表肿瘤核。
		参数说明：
			train_root_path: 训练根目录
			val_root_path: 验证根目录
			is_train: 是否为训练模式
			step: 第几次模型使用的数据集
	"""
	def __init__(self, train_root_path, val_root_path, is_train=True, step=1):
		
		self.train_root_path = train_root_path
		self.val_root_path = val_root_path
		self.is_train = is_train
		self.data_box = [144, 192, 192]
		self.detection_box = [128, 160, 128]
		self.data_size = 16 # 16
		self.step = step

		# hgg_path_list = glob.glob(self.train_root_path+'/HGG/*')
		# lgg_path_list = glob.glob(self.train_root_path+'/LGG/*')
		self.path_list = load_hgg_lgg_files(self.train_root_path)
		# if self.is_train:
		# 	self.path_list = self.path_list[:int(0.9*len(self.path_list))] # 256个
		# else:
		# 	self.path_list = self.path_list[int(0.9*len(self.path_list)):]
		
		# print('len(self.path): ', len(self.path_list))
		# if not self.is_train:
		# 	self.path_list = load_val_file(self.val_root_path)
		
		# print(self.path_list)

	
	def __len__(self):
		return len(self.path_list)
	
	def __getitem__(self, index):
		path = self.path_list[index]
		# print(path)

		if self.step == 2:
			image, label = self.pretreat(path)
			image, label = self.train_pretreat(image, label)
			image = torch.from_numpy(image).float()
			label = torch.from_numpy(label).float()
			# print(label.shape)
			return image, label
		elif self.step == 1:
			image, label, index_min, index_max = self.module_detection(path, index)
			image, label = self.module_train(image, label)
			image = torch.from_numpy(image).float()
			label = torch.from_numpy(label).float()
			return image, label, index_min, index_max
	
	def module_detection(self, path, index):
		"""
			数据定位的数据处理。
			210+75=285
			原图：[155, 240, 240]
			BOX_MAX: 103, 155, 114
			---->  self.detection_box = [110, 160, 160]
		"""
		image = []
		label = []
		image_t, label_t = make_image_label(path)
		# print('File name is :', path)
		
		flair, t1, t1ce, t2 = image_t
		seg = label_t

		# 定位裁剪区域.
		box_min, box_max = get_box(seg, 0)
		index_min, index_max = make_box(flair, box_min, box_max, self.detection_box)
		# print(index_min, index_max)

		# 裁剪图像
		flair = crop_with_box(flair, index_min, index_max)
		t1 = crop_with_box(t1, index_min, index_max)
		t1ce = crop_with_box(t1ce, index_min, index_max)
		t2 = crop_with_box(t2, index_min, index_max)
		seg = crop_with_box(seg, index_min, index_max)

		# 标准化
		flair = normalization(flair)
		t1 = normalization(t1)
		t1ce = normalization(t1ce)
		t2 = normalization(t2)

		all_tumor_label = seg

		image.append(flair)
		image.append(t1)
		image.append(t1ce)
		image.append(t2)

		label.append(all_tumor_label)

		image = np.asarray(image)
		label = np.asarray(label)

		# print(self.xx, ' ', self.yy, ' ', self.zz)

		return image, label, index_min, index_max

	def module_train(self, image, label):
		times = int(image.shape[1] / self.data_size)

		image_volumn = []
		label_volumn = []

		for i in range(times):
			if self.is_train:
				st = np.random.randint(0, image.shape[1] - self.data_size+1)
			else:
				st = i * self.data_size
			
			image_volumn.append(image[:, st:st+self.data_size, :, :])
			label_volumn.append(label[:, st:st+self.data_size, :, :])
		
		image_volumn = np.asarray(image_volumn)
		label_volumn = np.asarray(label_volumn)

		return image_volumn, label_volumn

	def pretreat(self, path):
		"""
			处理图像与标签。
		"""
		image = []
		label = []

		image_t, label_t = make_image_label(path)
		# print(image_t[0].shape)
		flair, t1, t1ce, t2 = image_t
		seg = label_t

		# 按照flair确定裁剪区域
		box_min, box_max = get_box(flair, 0)
		index_min, index_max = make_box(flair, box_min, box_max, self.data_box)

		# 裁剪
		flair = crop_with_box(flair, index_min, index_max)
		t1 = crop_with_box(t1, index_min, index_max)
		t1ce = crop_with_box(t1ce, index_min, index_max)
		t2 = crop_with_box(t2, index_min, index_max)
		seg = crop_with_box(seg, index_min, index_max)

		# 标准化
		flair = normalization(flair)
		t1 = normalization(t1)
		t1ce = normalization(t1ce)
		t2 = normalization(t2)

		tumor_core_label = get_tumor_core_labels(seg)

		# 想法：阅读别人的程序，发现也可以多方向扫描MRI。
		# ...

		image.append(flair)
		image.append(t1)
		image.append(t1ce)
		image.append(t2)

		label.append(tumor_core_label)

		image = np.asarray(image)
		label = np.asarray(label)

		return image, label

	def train_pretreat(self, image, label):
		"""
			随机切片。
		"""
		times = int(image.shape[1] / self.data_size)

		image_volumn = []
		label_volumn = []

		for i in range(times):
			if self.is_train:
				st = np.random.randint(0, image.shape[1] - self.data_size+1)
			else:
				st = i * self.data_size
			
			image_volumn.append(image[:, st:st+self.data_size, :, :])
			label_volumn.append(label[:, st:st+self.data_size, :, :])
		
		image_volumn = np.asarray(image_volumn)
		label_volumn = np.asarray(label_volumn)
		
		return image_volumn, label_volumn


class BraTS2019(Dataset):
	"""
	Brats2019数据集。
	"""
	def __init__(self, train_root_path, val_root_path, is_train=True, task='WT', predict=False):
		self.train_root_path = train_root_path
		self.val_root_path = val_root_path
		self.is_train = is_train
		self.task = task
		self.predict = predict
		self.data_box = [144, 192, 192] # 240, 240, 155
		self.data_dim = 16

		self.path_list = load_hgg_lgg_files(self.train_root_path)
		# print(self.path_list)
		if not self.is_train:
			self.path_list = load_val_file(self.val_root_path)

	def __len__(self):
		return len(self.path_list)

	def __getitem__(self, item):

		path = self.path_list[item]
		if self.predict:
			print(path)
		if self.is_train:
			image, label, box_min, box_max = self.first_pre(path)
			# image, label = self.second_pre(image, label) # 切片
			image = torch.from_numpy(image).float()
			label = torch.from_numpy(label).float()

			return image, label
		else:
			image, label, box_min, box_max = self.first_pre(path)
			image = torch.from_numpy(image).float()
			name = path.split('/')[-1]

			return image, name, box_min, box_max

	def first_pre(self, path):
		"""
		从路径加载，第一步处理。
		:param path: 单个大脑路径。
		:return: 图像，标签。
		"""
		image = []
		label = []
		image_t, label_t = make_image_label(path)
		# print(image_t[0].shape)
		flair, t1, t1ce, t2 = image_t
		seg = label_t

		# 按照flair确定裁剪区域
		box_min, box_max = get_box(flair, 0)
		index_min, index_max = make_box(flair, box_min, box_max, self.data_box)

		# 裁剪
		flair = crop_with_box(flair, index_min, index_max)
		t1 = crop_with_box(t1, index_min, index_max)
		t1ce = crop_with_box(t1ce, index_min, index_max)
		t2 = crop_with_box(t2, index_min, index_max)
		if self.is_train:
			seg = crop_with_box(seg, index_min, index_max)

		# 标准化
		flair = normalization(flair)
		t1 = normalization(t1)
		t1ce = normalization(t1ce)
		t2 = normalization(t2)

		# label1 = get_ncr_labels(seg)
		# label2 = get_ed_labels(seg)
		# label3 = get_ot_labels(seg)
		# label4 = get_tumor_core_labels(seg)
		if self.task == 'WT' and seg:
			label = get_WT_labels(seg)
		elif self.task == 'TC' and seg:
			label = get_TC_labels(seg)
		elif self.task == 'ET' and seg:
			label = get_ET_labels(seg)
		elif self.task == 'NCR' and seg:
			label = get_NCR_NET_label(seg)
		elif self.is_train:
			label = seg * 1.0

		# 想法：阅读别人的程序，发现也可以多方向扫描MRI。
		# ...

		image.append(flair)
		image.append(t1)
		image.append(t1ce)
		image.append(t2)

		# label.append(label1)
		# label.append(label2)
		# label.append(label3)
		# label.append(label4)

		image = np.asarray(image)
		label = np.asarray(label)

		return image, label, index_min, index_max

	def second_pre(self, image, label):
		"""
			随机切片。
			output:[9, 4, 16, 192, 192]
		"""
		times = int(image.shape[1] / self.data_dim)

		image_volumn = []
		label_volumn = []

		for i in range(times):
			if self.is_train:
				st = np.random.randint(0, image.shape[1] - self.data_dim + 1)
			else:
				st = i * self.data_dim

			image_volumn.append(image[:, st:st + self.data_dim, :, :])
			label_volumn.append(label[:, st:st + self.data_dim, :, :])

		image_volumn = np.asarray(image_volumn)
		label_volumn = np.asarray(label_volumn)

		return image_volumn, label_volumn


class BraTS2019_Multi(Dataset):
	"""
	Brats2019数据集。
	"""

	def __init__(self, train_root_path, val_root_path, is_train=True, task='WT', predict=False):
		self.train_root_path = train_root_path
		self.val_root_path = val_root_path
		self.is_train = is_train
		self.task = task
		self.predict = predict
		self.data_box = [144, 192, 192]
		self.data_dim = 16

		self.path_list = load_hgg_lgg_files(self.train_root_path)

	def __len__(self):
		return len(self.path_list)

	def __getitem__(self, item):

		path = self.path_list[item]
		if self.predict:
			print(path)
		image, label = self.first_pre(path)
		# image, label = self.second_pre(image, label) # 切片
		image = torch.from_numpy(image).float()
		label = torch.from_numpy(label).float()

		return image, label

	def first_pre(self, path):
		"""
        从路径加载，第一步处理。
        :param path: 单个大脑路径。
        :return: 图像，标签。
        """
		image = []
		label = []
		image_t, label_t = make_image_label(path)
		# print(image_t[0].shape)
		flair, t1, t1ce, t2 = image_t
		seg = label_t

		# 按照flair确定裁剪区域
		box_min, box_max = get_box(flair, 0)
		index_min, index_max = make_box(flair, box_min, box_max, self.data_box)

		# 裁剪
		flair = crop_with_box(flair, index_min, index_max)
		t1 = crop_with_box(t1, index_min, index_max)
		t1ce = crop_with_box(t1ce, index_min, index_max)
		t2 = crop_with_box(t2, index_min, index_max)
		seg = crop_with_box(seg, index_min, index_max)

		# 标准化
		flair = normalization(flair)
		t1 = normalization(t1)
		t1ce = normalization(t1ce)
		t2 = normalization(t2)

		label = get_precise_labels(seg)

		# 想法：阅读别人的程序，发现也可以多方向扫描MRI。
		# ...

		image.append(flair)
		image.append(t1)
		image.append(t1ce)
		image.append(t2)

		# label.append(label1)
		# label.append(label2)
		# label.append(label3)
		# label.append(label4)

		image = np.asarray(image)
		label = np.asarray(label)

		return image, label

	def second_pre(self, image, label):
		"""
            随机切片。
            output:[9, 4, 16, 192, 192]
        """
		times = int(image.shape[1] / self.data_dim)

		image_volumn = []
		label_volumn = []

		for i in range(times):
			if self.is_train:
				st = np.random.randint(0, image.shape[1] - self.data_dim + 1)
			else:
				st = i * self.data_dim

			image_volumn.append(image[:, st:st + self.data_dim, :, :])
			label_volumn.append(label[:, st:st + self.data_dim, :, :])

		image_volumn = np.asarray(image_volumn)
		label_volumn = np.asarray(label_volumn)

		return image_volumn, label_volumn


class AneuMulti(Dataset):
	def __init__(self, train_path, is_train=True, val_path=''):
		self.train_path = train_path
		self.is_train = is_train
		self.val_path = val_path
		if self.is_train:
			self.path_list = load_aneu_image_path(self.train_path)
		else:
			self.path_list = load_aneu_image_path(self.val_path)
		self.index_box = [128, 448, 448]
		self.data_dim = 64

	def __len__(self):
		return len(self.path_list)

	def second_pre(self, image, label):
		"""
			随机切片。
			output:[9, 4, 16, 192, 192]
		"""
		times = int(image.shape[1] / self.data_dim)

		image_volumn = []
		label_volumn = []

		for i in range(times):
			if self.is_train:
				st = np.random.randint(0, image.shape[1] - self.data_dim + 1)
			else:
				st = i * self.data_dim

			image_volumn.append(image[:, st:st + self.data_dim, :, :])
			if self.is_train:
				label_volumn.append(label[st:st + self.data_dim, :, :])

		image_volumn = np.asarray(image_volumn)
		label_volumn = np.asarray(label_volumn)

		return image_volumn, label_volumn

	def __getitem__(self, item):
		# print('len: ', len(self.path_list))
		# print('item: ', item)
		# print(self.path_list[item])

		path = glob.glob(self.path_list[item]+'/*')
		image = None
		label = None
		# print(path)
		if 'Untitled.nii.gz' in path[0]:
			image_path = path[1]
			label_path = path[0]
		else:
			image_path = path[0]
			label_path = path[1]

		# print('path: ', image_path)
		image = load_nii_to_array(image_path)
		if self.is_train:
			label = load_nii_to_array(label_path)
		
		head_image = nib.load(image_path)
		affine = head_image.affine

		img_shape = image.shape

		index_min, index_max = get_box(image, margin=0)

		index_min, index_max = make_box(image, index_min, index_max, self.index_box)

		# print('image.shape: ', image.shape)
		# print('index: ', index_min, index_max)

		image = crop_with_box(image, index_min, index_max)
		if self.is_train:
			label = crop_with_box(label, index_min, index_max)

		image = normalization(image)

		img = []
		img.append(image)
		image = np.asarray(img)
		if self.is_train:
			label = get_NCR_NET_label(label)
			label = np.asarray(label)
		# print('image.shape: ', image.shape)

		image, label = self.second_pre(image, label)
		# print('imagere.shape: ', image.shape)
		name_list = image_path.split('/')
		name = name_list[-3] + '_' + name_list[-2]
		if self.is_train:
			return torch.from_numpy(image).float(), torch.from_numpy(label).float()
		else:
			return torch.from_numpy(image).float(), torch.from_numpy(label).float(), img_shape, index_min, index_max, affine, name


class BraTS2019_Random(Dataset):
	"""
	Brats2019数据集。
	"""
	def __init__(self, train_root_path, val_root_path, is_train=True, task='WT', predict=False):
		self.train_root_path = train_root_path
		self.val_root_path = val_root_path
		self.is_train = is_train
		self.task = task
		self.predict = predict
		self.data_box = [144, 192, 192] # 240, 240, 155
		self.data_dim = 16

		self.path_list = load_hgg_lgg_files(self.train_root_path)
		# print(self.path_list)
		if not self.is_train:
			self.path_list = load_val_file(self.val_root_path)

	def __len__(self):
		return len(self.path_list)

	def __getitem__(self, item):

		path = self.path_list[item]
		if self.predict:
			print(path)
		if self.is_train:
			image, label, box_min, box_max = self.first_pre(path)
			# image, label = self.second_pre(image, label) # 切片
			image = torch.from_numpy(image).float()
			label = torch.from_numpy(label).float()

			return image, label
		else:
			image, label, box_min, box_max = self.first_pre(path)
			image = torch.from_numpy(image).float()
			name = path.split('/')[-1]

			return image, name, box_min, box_max

	def first_pre(self, path):
		"""
		从路径加载，第一步处理。
		:param path: 单个大脑路径。
		:return: 图像，标签。
		"""
		image = []
		label = []
		image_t, label_t = make_image_label(path)
		# print(image_t[0].shape)
		flair, t1, t1ce, t2 = image_t
		seg = label_t

		# 按照flair确定裁剪区域
		box_min, box_max = get_box(flair, 0)
		index_min, index_max = make_box(flair, box_min, box_max, self.data_box)

		# 裁剪
		flair = crop_with_box(flair, index_min, index_max)
		t1 = crop_with_box(t1, index_min, index_max)
		t1ce = crop_with_box(t1ce, index_min, index_max)
		t2 = crop_with_box(t2, index_min, index_max)
		if self.is_train:
			seg = crop_with_box(seg, index_min, index_max)

		# 标准化
		flair = normalization(flair)
		t1 = normalization(t1)
		t1ce = normalization(t1ce)
		t2 = normalization(t2)

		# label1 = get_ncr_labels(seg)
		# label2 = get_ed_labels(seg)
		# label3 = get_ot_labels(seg)
		# label4 = get_tumor_core_labels(seg)
		if self.task == 'WT' and seg:
			label = get_WT_labels(seg)
		elif self.task == 'TC' and seg:
			label = get_TC_labels(seg)
		elif self.task == 'ET' and seg:
			label = get_ET_labels(seg)
		elif self.task == 'NCR' and seg:
			label = get_NCR_NET_label(seg)
		elif self.is_train:
			label = seg * 1.0

		# 想法：阅读别人的程序，发现也可以多方向扫描MRI。
		# ...

		image.append(flair)
		image.append(t1)
		image.append(t1ce)
		image.append(t2)

		# label.append(label1)
		# label.append(label2)
		# label.append(label3)
		# label.append(label4)
		image = np.asarray(image)
		label = np.asarray(label)
		image, label = self.second_pre(image, label)



		return image, label, index_min, index_max

	def second_pre(self, image, label):
		"""
			随机切片。
			output:[9, 4, 16, 192, 192]
		"""
		times = int(image.shape[1] / self.data_dim) # 12 * 2

		image_volumn = []
		label_volumn = []

		for i in range(times):
			if self.is_train:
				st = np.random.randint(0, image.shape[1] - self.data_dim + 1)
			else:
				st = i * self.data_dim

			image_volumn.append(image[:, st:st + self.data_dim, :, :])
			if self.is_train:
				label_volumn.append(label[st:st + self.data_dim, :, :])

		image_volumn = np.asarray(image_volumn)
		label_volumn = np.asarray(label_volumn)

		return image_volumn, label_volumn


class BraTS2019_Random_DataArg(Dataset):
	"""
	Brats2019数据集。
	"""
	def __init__(self, train_root_path, val_root_path, is_train=True, task='WT', predict=False):
		self.train_root_path = train_root_path
		self.val_root_path = val_root_path
		self.is_train = is_train
		self.task = task
		self.predict = predict
		self.data_box = [144, 192, 192] # 240, 240, 155
		self.data_dim = 16

		self.path_list = load_hgg_lgg_files(self.train_root_path)
		# print(self.path_list)
		if not self.is_train:
			self.path_list = load_val_file(self.val_root_path)

	def __len__(self):
		return len(self.path_list)

	def __getitem__(self, item):

		path = self.path_list[item]
		if self.predict:
			print(path)
		if self.is_train:
			image, label, box_min, box_max = self.first_pre(path)
			# image, label = self.second_pre(image, label) # 切片
			image = torch.from_numpy(image).float()
			label = torch.from_numpy(label).float()
			return image, label
		else:
			image, label, box_min, box_max = self.first_pre(path)
			image = torch.from_numpy(image).float()
			name = path.split('/')[-1]
			return image, name, box_min, box_max

	def first_pre(self, path):
		"""
		从路径加载，第一步处理。
		:param path: 单个大脑路径。
		:return: 图像，标签。
		"""
		image = []
		label = []
		image_t, label_t = make_image_label(path)
		# print(image_t[0].shape)
		flair, t1, t1ce, t2 = image_t
		seg = label_t

		# 按照flair确定裁剪区域
		box_min, box_max = get_box(flair, 0)
		index_min, index_max = make_box(flair, box_min, box_max, self.data_box)

		# 裁剪
		flair = crop_with_box(flair, index_min, index_max)
		t1 = crop_with_box(t1, index_min, index_max)
		t1ce = crop_with_box(t1ce, index_min, index_max)
		t2 = crop_with_box(t2, index_min, index_max)
		if self.is_train:
			seg = crop_with_box(seg, index_min, index_max)

		# 标准化
		flair = normalization(flair)
		t1 = normalization(t1)
		t1ce = normalization(t1ce)
		t2 = normalization(t2)

		# label1 = get_ncr_labels(seg)
		# label2 = get_ed_labels(seg)
		# label3 = get_ot_labels(seg)
		# label4 = get_tumor_core_labels(seg)
		if self.task == 'WT' and seg:
			label = get_WT_labels(seg)
		elif self.task == 'TC' and seg:
			label = get_TC_labels(seg)
		elif self.task == 'ET' and seg:
			label = get_ET_labels(seg)
		elif self.task == 'NCR' and seg:
			label = get_NCR_NET_label(seg)
		elif self.is_train:
			label = seg * 1.0

		# 想法：阅读别人的程序，发现也可以多方向扫描MRI。
		# ...

		image.append(flair)
		image.append(t1)
		image.append(t1ce)
		image.append(t2)

		# label.append(label1)
		# label.append(label2)
		# label.append(label3)
		# label.append(label4)
		image = np.asarray(image)
		label = np.asarray(label)
		image, label = self.second_pre(image, label)

		return image, label, index_min, index_max

	def second_pre(self, image, label):
		"""
			随机切片。
			output:[9, 4, 16, 192, 192]
		"""
		times = int(image.shape[-1] / self.data_dim) # 12 * 2
		lbl = []

		img = np.transpose(image, [0, 1, 3, 2])
		if self.is_train:
			lbl = np.transpose(label, [0, 2, 1])

		image_volumn = []
		label_volumn = []

		for i in range(times):
			if self.is_train:
				st = np.random.randint(0, image.shape[-1] - self.data_dim + 1)

				image_volumn.append(image[:, :, :, st:st + self.data_dim])
				label_volumn.append(label[:, :, st:st + self.data_dim])
				image_volumn.append(img[:, :, :, st:st + self.data_dim])
				label_volumn.append(lbl[:, :, st:st + self.data_dim])

			else:
				st = i * self.data_dim

				image_volumn.append(image[:, :, :, st:st + self.data_dim])


		image_volumn = np.asarray(image_volumn)
		label_volumn = np.asarray(label_volumn)

		return image_volumn, label_volumn


