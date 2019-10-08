# coding=utf-8
import os
import gzip
import glob
import numpy as np
import pandas as pd
import nibabel as nib

import cv2
import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.image as Image

from data.dataset import BraTS2017, BraTS2019

import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

import scipy.misc

import warnings
warnings.filterwarnings('ignore')

import models
from utils.utils import *

from config import opt

"""
nibable:
	nib.save(mat, 'path.nii.gz')
"""

def unzip():
	"""
	数据集解压：.nii.gz -> .nii
	"""
	hgg_gz_file = glob.glob(opt.train_root_path+'/HGG/*/*.gz')
	lgg_gz_file = glob.glob(opt.train_root_path+'/LGG/*/*.gz')
	val_file = glob.glob(opt.val_root_path+'/*/*.gz')


	for path in hgg_gz_file:
		print(path)
		g_file = gzip.GzipFile(path)
		name = path.replace(".gz", "")
		open(name, 'wb').write(g_file.read())
		g_file.close()
	print('HGG文件解压完毕。')

	for path in lgg_gz_file:
		print(path)
		g_file = gzip.GzipFile(path)
		name = path.replace(".gz", "")
		open(name, 'wb').write(g_file.read())
		g_file.close()
	print('LGG文件解压完毕。')

	# for path in val_file:
	# 	print(path)
	# 	g_file = gzip.GzipFile(path)
	# 	name = path.replace(".gz", "")
	# 	open(name, 'wb').write(g_file.read())
	# 	g_file.close()
	# print('VAL文件解压完毕。')

def train(**kwargs):
	device_ids = [0, 1, 2]
	opt._parse(kwargs)
	
	# 配置模型
	model = getattr(models, opt.model)()
	save_dir = 'ckpt_'+opt.model+'/'

	device = t.device('cuda') if opt.use_gpu else t.device('cpu')

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	
	print('Start load DataParallels')
	model = nn.DataParallel(model)
	model = model.cuda(device=device_ids[0])
	print('Loading DataParallel finished.')
	if opt.load_model_path:
		model.load_state_dict(t.load('./'+save_dir+opt.load_model_path))
		print('load model')


	# pytorch数据处理
	train_data = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=True, step=2)
	# test_data = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=False)
	print('train_data and test_data load finished.')

	train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
	# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
	print('train and test dataloader load finished.')

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
	best_dice = -1
	best_epoch = -1
	print('criterion and optimizer is finished.')


	for epoch in range(opt.max_epoch):
		print('epoch------------------------------' + str(epoch))
		train_loss = []
		test_loss = []
		train_dice = []
		test_dice = []

		print('training---------------------------')
		model.train()
		for ii, (image, label) in enumerate(train_dataloader):
			for i in range(len(image)):
				# print('image.size(): ', image.size())
				# print('label.size(): ', label.size())
				# image.size():  torch.Size([2, 14, 4, 10, 192, 192])
				# label.size():  torch.Size([2, 14, 1, 10, 192, 192])
				# return
				img = Variable(image[i].cuda())
				lbl = Variable(label[i].cuda())

				for k in range(2):
					img_ = img[:, :, :, :, 96*k:96*k+96]
					lbl_ = lbl[:, :, :, :, 96*k:96*k+96]
					optimizer.zero_grad()
					predicts = model(img_)
					# print('Output size: ', predicts.size())
					# print('lbl.size(): ', lbl_[:, 0, :, :, :].size())
					# Output size:                 torch.Size([4, 2, 32, 192, 96])
					# lbl_[:, 0, :, :, :].size():  torch.Size([4, 32, 192, 96])
					loss = criterion(predicts, lbl_[:, 0, :, :, :].long())
					train_loss.append(float(loss))
					loss.backward()
					optimizer.step()

					predicts = F.softmax(predicts, dim=1)
					predicts = (predicts[:, 1, :, :, :] > 0.5).long()
					d = dice(predicts, lbl_[:, 0, :, :, :].long())
					train_dice.append(d)
					
			
		# print('testing-----------------------------')
		# model.eval()
		# for ii, (image, label) in enumerate(test_dataloader):
			
		# 	for i in range(len(image)):
		# 		img = Variable(image[i].cuda())
		# 		lbl = Variable(label[i].cuda())

		# 		for k in range(2):
		# 			img_ = img[:, :, :, :, 96*k:96*k+96]
		# 			lbl_ = lbl[:, :, :, :, 96*k:96*k+96]
		# 			# optimizer.zero_grad()
		# 			predicts = model(img_)

		# 			# for kk in range():
		# 			# cv2.imwrite(opt.test_images+'img_%s_%s_%s'%(ii, i, k), img_)
		# 			# cv2.imwrite(opt.test_images+'lbl_%s_%s_%s'%(ii, i, k), lbl_)

		# 			loss = criterion(predicts, lbl_[:, 0, :, :, :].long())
		# 			test_loss.append(float(loss))
		# 			# loss.backward()
		# 			# optimizer.step()

		# 			predicts = F.softmax(predicts, dim=1)
		# 			predicts = (predicts[:, 1, :, :, :] > 0.5).long()
		# 			d = dice(predicts, lbl_[:, 0, :, :, :].long())
		# 			test_dice.append(d)
		
		# **************** save loss for one batch ****************
		print('train_loss ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
		# print('test_loss ' + str(sum(test_loss) / (len(test_loss) * 1.0)))
		print('train_dice ' + str(sum(train_dice) / (len(train_dice) * 1.0)))
		# print('test_dice ' + str(sum(test_dice) / (len(test_dice) * 1.0)))

		# if sum(test_dice) / (len(test_dice) * 1.0) > best_dice:
		# 	best_dice = sum(test_dice) / (len(test_dice) * 1.0)
		# 	best_epoch = epoch

		# **************** save model ****************
		# if epoch % 10 == 0:
		torch.save(model.state_dict(), os.path.join(save_dir, 'epoch_0.pth'))

def test(**kwargs):
	device_ids = [0, 1, 2]
	opt._parse(kwargs)
	
	# 配置模型
	model = getattr(models, opt.model)()
	save_dir = 'ckpt_'+opt.model+'/'

	device = t.device('cuda') if opt.use_gpu else t.device('cpu')

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	
	print('Start load DataParallels')
	model = nn.DataParallel(model)
	model = model.cuda(device=device_ids[0])
	print('Loading DataParallel finished.')
	if opt.load_model_path:
		model.load_state_dict(t.load('./'+save_dir+opt.load_model_path))
		print('load model')


	# pytorch数据处理
	test_data = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=False)
	print('train_data and test_data load finished.')

	test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
	print('train and test dataloader load finished.')

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
	best_dice = -1
	best_epoch = -1
	print('criterion and optimizer is finished.')


	for epoch in range(opt.max_epoch):
		print('epoch------------------------------' + str(epoch))

		test_loss = []
		test_dice = []
			
		print('testing-----------------------------')
		model.eval()
		for ii, (image, label) in enumerate(test_dataloader):
			print('image.shape:', image.shape)
			for i in range(len(image)):
				img = Variable(image[i].cuda())
				lbl = Variable(label[i].cuda())

				for k in range(2):
					img_ = img[:, :, :, :, 96*k:96*k+96]
					lbl_ = lbl[:, :, :, :, 96*k:96*k+96]
					# optimizer.zero_grad()
					
					predicts = model(img_)



					
					loss = criterion(predicts, lbl_[:, 0, :, :, :].long())
					test_loss.append(float(loss))
					# loss.backward()
					# optimizer.step()

					predicts = F.softmax(predicts, dim=1)
					predicts = (predicts[:, 1, :, :, :] > 0.5).long()
					print('pre___.shape:   ', predicts.shape)


					for k1 in range(9):
						for k2 in range(16):
							imggg = predicts[k1, k2, :, :].data.cpu().numpy()
							lblll = lbl_[k1, 0, k2, :, :].data.cpu().numpy()
							print('imgggg.shape: ', imggg.shape)
							print(lblll.shape)
							print(imggg)

							cv2.imwrite(opt.test_images+'img_%s_%s_%s_%s.png'%(i, k, k1, k2), imggg)
							cv2.imwrite(opt.test_images+'lbl_%s_%s_%s_%s.png'%(i, k, k1, k2), lblll)


					d = dice(predicts, lbl_[:, 0, :, :, :].long())
					test_dice.append(d)
			break
		
		# **************** save loss for one batch ****************
		print('test_loss ' + str(sum(test_loss) / (len(test_loss) * 1.0)))
		print('test_dice ' + str(sum(test_dice) / (len(test_dice) * 1.0)))

		if sum(test_dice) / (len(test_dice) * 1.0) > best_dice:
			best_dice = sum(test_dice) / (len(test_dice) * 1.0)
			best_epoch = epoch

def detection_and_train(**kwargs):
	device_ids = [0, 1]
	opt._parse(kwargs)
	
	# 配置模型
	model = getattr(models, opt.model)(output_data=5)
	save_dir = 'ckpt_'+opt.model+'/'

	device = t.device('cuda') if opt.use_gpu else t.device('cpu')

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	
	print('Start load DataParallels')
	model = nn.DataParallel(model)
	model = model.cuda(device=device_ids[0])
	print('Loading DataParallel finished.')
	if opt.load_model_path:
		model.load_state_dict(t.load('./'+save_dir+opt.load_model_path))
		print('load model')


	# pytorch数据处理
	train_data = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=True)
	# test_data = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=False)
	print('train_data and test_data load finished.')

	train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
	# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
	print('train and test dataloader load finished.')

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
	best_dice = -1
	best_epoch = -1
	print('criterion and optimizer is finished.')


	for epoch in range(opt.max_epoch):
		print('epoch------------------------------' + str(epoch))
		train_loss = []
		test_loss = []
		train_dice = []
		test_dice = []

		print('training---------------------------')
		model.train()
		for ii, (image, label, index_min, index_max) in enumerate(train_dataloader):
			for i in range(len(image)): # 每个batch_size
				# print('image.size(): ', image.size())
				# print('label.size(): ', label.size())
				# image.size():  torch.Size([2, 11, 4, 10, 160, 120])
				# label.size():  torch.Size([2, 11, 1, 10, 160, 120])

				img = Variable(image[i].cuda())
				lbl = Variable(label[i].cuda())

				optimizer.zero_grad()
				predicts = model(img)
				# print('Output size: ', predicts.size())
				# print('lbl.size(): ', lbl[:, 0, :, :, :].size())
				# Output size:  torch.Size([3, 2, 32, 160, 128])
				# lbl.size():  torch.Size([3, 32, 160, 128])
				loss = criterion(predicts, lbl[:, 0, :, :, :].long())
				train_loss.append(float(loss))
				loss.backward()
				optimizer.step()

				predicts = F.softmax(predicts, dim=1)
				predicts = (predicts[:, 1, :, :, :] > 0.5).long()
				d = dice(predicts, lbl[:, 0, :, :, :].long())
				train_dice.append(d)
			
			
				
		"""	
		print('testing-----------------------------')
		model.eval()
		for ii, (image, label) in enumerate(test_dataloader):
			
			for i in range(len(image)):
				img = Variable(image[i].cuda())
				lbl = Variable(label[i].cuda())

				for k in range(2):
					img_ = img[:, :, :, :, 96*k:96*k+96]
					lbl_ = lbl[:, :, :, :, 96*k:96*k+96]
					# optimizer.zero_grad()
					predicts = model(img_)

					# for kk in range():
					# cv2.imwrite(opt.test_images+'img_%s_%s_%s'%(ii, i, k), img_)
					# cv2.imwrite(opt.test_images+'lbl_%s_%s_%s'%(ii, i, k), lbl_)

					loss = criterion(predicts, lbl_[:, 0, :, :, :].long())
					test_loss.append(float(loss))
					# loss.backward()
					# optimizer.step()

					predicts = F.softmax(predicts, dim=1)
					predicts = (predicts[:, 1, :, :, :] > 0.5).long()
					d = dice(predicts, lbl_[:, 0, :, :, :].long())
					test_dice.append(d)
		"""
		
		# **************** save loss for one batch ****************
		print('train_loss ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
		# print('test_loss ' + str(sum(test_loss) / (len(test_loss) * 1.0)))
		print('train_dice ' + str(sum(train_dice) / (len(train_dice) * 1.0)))
		# print('test_dice ' + str(sum(test_dice) / (len(test_dice) * 1.0)))

		# if sum(test_dice) / (len(test_dice) * 1.0) > best_dice:
		# 	best_dice = sum(test_dice) / (len(test_dice) * 1.0)
		# 	best_epoch = epoch

		# **************** save model ****************
		if epoch % 5 == 0:
			torch.save(model.state_dict(), os.path.join(save_dir, 'detect_train_0.pth'))

def help():
	"""
	打印帮助信息：pythonfile.py help
	"""
	print("""
	usage: python file.py <function> [--args=value]
	<function> := train | test | help
	example:
			python {0} train --lr=0.01 --env='env0701'
			python {0} test --dataset='path/to/dataset/root/'
			python {0} help
	avaiable args:
	""".format(__file__))
	from inspect import getsource
	source = (getsource(opt.__class__))
	print(source)

def moduletest():
	net = models.UNet3D(4, 2, degree=16)
	print('total parameter: ', str(netSize(net)))

	x = torch.randn(4, 4, 16, 192, 192)
	print('input data:')
	print(x.shape)

	if torch.cuda.is_available():
		print('Used:Cuda')
		net = net.cuda()
		x = x.cuda()
	
	y = net(x)
	print('output data:')
	print(y.shape)

def datatest():
	# dataset = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=True, step=1)
	# image, label, index_min, index_max = dataset[89]
	# print(image.shape)
	# print(label.shape)
	# print(index_min)
	# print(index_max)
	dataset = BraTS2019(opt.local_root_path, opt.val_root_path, is_train=True)
	image, label = dataset[2]
	print(image.shape)
	plt.imshow(image[5][0][6], cmap='RdPu')
	plt.show()

def brats2019_train(**kwargs):
	device_ids = [0, 1]
	opt._parse(kwargs)

	# 配置模型
	model = getattr(models, opt.model)()
	save_dir = 'ckpt_' + opt.model + '/'

	device = t.device('cuda') if opt.use_gpu else t.device('cpu')

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	print('Start load DataParallels')
	model = nn.DataParallel(model)
	model = model.cuda(device=device_ids[0])
	print('Loading DataParallel finished.')
	if opt.load_model_path:
		model.load_state_dict(t.load('./' + save_dir + opt.load_model_path))
		print('load model')

	# pytorch数据处理
	train_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=True)
	# test_data = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=False)
	print('train_data and test_data load finished.')

	train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
	# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
	print('train and test dataloader load finished.')

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
	best_dice = -1
	best_epoch = -1
	print('criterion and optimizer is finished.')

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		model.train()
		for ii, (image, label) in enumerate(train_dataloader):
			img = image.cuda()
			lbl = label.cuda()
			"""
			img: [batch_size, 4, 144, 192, 192]
			lbl: [batch_size, 1, 144, 192, 192]
			"""

			optimizer.zero_grad()
			predicts = model(img)
			print('predicts.shape(): ', predicts.shape)
			loss = criterion(predicts, lbl)
			loss.backward()
			optimizer.step()





			return



def othertest():
	a = np.array([49, 54, 58])
	b = np.array([121, 129, 116])
	c = b-a
	print(c)

def trash():

	img = plt.imread('./trash/img_0_0_5_0.png')
	lbl = plt.imread('./trash/lbl_0_0_5_0.png')
	# print(img)
	# plt.imshow(img, cmap='RdPu')
	plt.imshow(lbl, cmap='RdPu')
	plt.show()

if __name__ == '__main__':
	import fire

	fire.Fire()

