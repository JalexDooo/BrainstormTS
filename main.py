# coding=utf-8
import os
import gzip
import glob
import math
import numpy as np
import pandas as pd
import nibabel as nib

import cv2
import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.image as Image

from data.dataset import *

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

	for path in val_file:
		print(path)
		g_file = gzip.GzipFile(path)
		name = path.replace(".gz", "")
		open(name, 'wb').write(g_file.read())
		g_file.close()
	print('VAL文件解压完毕。')

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
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0, 1]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'



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
	train_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=True, task=opt.task)
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
	print(model.eval())

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		model.train()
		for ii, (image, label) in enumerate(train_dataloader):
			img = image.cuda()
			lbl = label.cuda()
			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				lbl_ = lbl[:, :, :, 96 * k:96 * k + 96]
				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				optimizer.zero_grad()
				predicts = model(img_)
				# print('predicts.shape(): ', predicts.shape)
				# print('lbl.shape(): ', lbl_.shape)
				loss = criterion(predicts, lbl_.long())
				train_loss.append(float(loss))
				# print('loss is : ', loss)
				loss.backward()
				optimizer.step()

				predicts = F.softmax(predicts, dim=1)
				predicts = (predicts[:, 1, :, :, :] > 0.5).long()
				d = dice(predicts, lbl_.long())
				train_dice.append(d)


		print('train_loss : ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
		print('train_dice : ' + str(sum(train_dice) / (len(train_dice) * 1.0)))

		torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__epoch_%s.pth'%(opt.task, epoch)))

def brats2019_test(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0, 1]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'



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
	train_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=True, task=opt.task)
	# test_data = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=False)
	print('train_data and test_data load finished.')

	train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
	# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
	print('train and test dataloader load finished.')

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
	best_dice = -1.0
	best_epoch = -1.0
	print('criterion and optimizer is finished.')
	print(model.eval())


	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		model.eval()
		for ii, (image, label) in enumerate(train_dataloader):
			img = image.cuda()
			lbl = label.cuda()
			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				lbl_ = lbl[:, :, :, 96 * k:96 * k + 96]
				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				optimizer.zero_grad()
				predicts = model(img_)
				# print('predicts.shape(): ', predicts.shape)
				# print('lbl.shape(): ', lbl_.shape)
				loss = criterion(predicts, lbl_.long())
				train_loss.append(float(loss))
				# print('loss is : ', loss)
				# loss.backward()
				# optimizer.step()

				predicts = F.softmax(predicts, dim=1)
				predicts = (predicts[:, 1, :, :, :] > 0.5).long()
				d = dice(predicts, lbl_.long())
				train_dice.append(d)


		print('train_loss : ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
		print('train_dice : ' + str(sum(train_dice) / (len(train_dice) * 1.0)))
		if best_dice < (sum(train_dice) / (len(train_dice) * 1.0)):
			best_dice = (sum(train_dice) / (len(train_dice) * 1.0))
			best_epoch = epoch

	print('The best epoch is %s, the best dice is %s.' % (best_epoch, best_dice))

def brats2019_predict(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0, 1]
	opt._parse(kwargs)

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

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
	train_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=True, predict=True, task=opt.task)
	# test_data = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=False)
	print('train_data and test_data load finished.')

	train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
	# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
	print('train and test dataloader load finished.')

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
	best_dice = -1
	best_epoch = -1
	print('criterion and optimizer is finished.')
	print(model.eval())

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------' % (epoch))
		train_loss = []
		train_dice = []
		model.eval()
		for ii, (image, label) in enumerate(train_dataloader):
			img = image.cuda()
			lbl = label.cuda()

			output_predict = []
			output_label = []

			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				lbl_ = lbl[:, :, :, 96 * k:96 * k + 96]
				"""
                img: [batch_size, 4, 144, 192, 192]
                lbl: [batch_size, 144, 192, 192]
                """

				optimizer.zero_grad()
				predicts = model(img_)

				predicts = F.softmax(predicts, dim=1)
				predicts = (predicts[:, 1, :, :, :] > 0.5).int()

				output_predict.append(predicts[0])
				output_predict.append(predicts[1])
				output_label.append(lbl_[0])
				output_label.append(lbl_[1])

				# d = dice(predicts, lbl_.long())
				# train_dice.append(d)

				# test = nib.Nifti1Image(predicts[0, 1, :, :, :].data.cpu().numpy(), np.eye(4))
				# path = opt.predict_nibable_path
				# nib.save(test, path + '/test.nii.gz')



			output_predict1 = t.cat((output_predict[0], output_predict[2]), 2).data.cpu().numpy()
			# output_predict2 = t.cat((output_predict[1], output_predict[3]), 2).data.cpu().numpy()
			output_label1 = t.cat((output_label[0], output_label[2]), 2).data.cpu().numpy()
			# output_label2 = t.cat((output_label[1], output_label[3]), 2).data.cpu().numpy()

			print(output_predict1.shape)
			# print(output_predict2.shape)
			print('----------------------')
			print(output_label1.shape)
			# print(output_label2.shape)

			path = opt.predict_nibable_path

			output_predict1 = nib.Nifti1Image(output_predict1, np.eye(4))
			# output_predict2 = nib.Nifti1Image(output_predict2, np.eye(4))
			output_label1 = nib.Nifti1Image(output_label1, np.eye(4))
			# output_label2 = nib.Nifti1Image(output_label2, np.eye(4))

			nib.save(output_label1, path + '/label_%s_.nii.gz'%(opt.task))
			# nib.save(output_label2, path + '/label_%s_2_.nii.gz'%(opt.task))
			nib.save(output_predict1, path + '/predict_%s_.nii.gz'%(opt.task))
			# nib.save(output_predict2, path + '/predict_%s_2_.nii.gz'%(opt.task))




			print('output_predict: ', output_predict[0].shape, ', ', len(output_predict))

			return

def brats2019_single_image_predict(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0, 1]
	opt._parse(kwargs)

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

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
	dataset = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=True, predict=True, task=opt.task)

	best_dice = -1
	best_epoch = -1
	print('criterion and optimizer is finished.')
	print(model.eval())

	image1, label1 = dataset[50]
	image2, label2 = dataset[150]
	# print(image1.shape)
	# print(label1.shape)
	#
	# flair, t1, t1ce, t2 = image1.int().numpy()
	#
	# flair = nib.Nifti1Image(flair, np.eye(4))
	# t1 = nib.Nifti1Image(t1, np.eye(4))
	# t1ce = nib.Nifti1Image(t1ce, np.eye(4))
	# t2 = nib.Nifti1Image(t2, np.eye(4))
	# path = opt.predict_nibable_path
	#
	# nib.save(flair, path + '/image_flair.nii.gz')
	# nib.save(t1, path + '/image_t1.nii.gz')
	# nib.save(t1ce, path + '/image_t1ce.nii.gz')
	# nib.save(t2, path + '/image_t2.nii.gz')
	# print('finished')
	#
	# return
	image = []
	image.append(image1.numpy())
	image.append(image2.numpy())
	image = np.array(image)

	label = []
	label.append(label1.numpy())
	label.append(label2.numpy())
	label = np.array(label)

	image = Variable(t.Tensor(image))
	label = Variable(t.Tensor(label))

	img = image.cuda()
	lbl = label.cuda()

	output_predict = []
	output_label = []

	for k in range(2):
		img_ = img[:, :, :, :, 96 * k:96 * k + 96]
		lbl_ = lbl[:, :, :, 96 * k:96 * k + 96]
		"""
        img: [batch_size, 4, 144, 192, 192]
        lbl: [batch_size, 144, 192, 192]
        """

		predicts = model(img_)
		# print(predicts)
		# print('--------------------------------------------------------------')
		# if opt.task == 'TC':
		# 	predicts = F.softmax(predicts, dim=1)
		values, predicts = t.max(predicts, dim=1)
		predicts = predicts.int()
		# print(predicts)
		# predicts = (predicts[:, 1, :, :, :] > 0.5).int()

		output_predict.append(predicts[0])
		output_predict.append(predicts[1])
		output_label.append(lbl_[0])
		output_label.append(lbl_[1])

	# d = dice(predicts, lbl_.long())
	# train_dice.append(d)

	# test = nib.Nifti1Image(predicts[0, 1, :, :, :].data.cpu().numpy(), np.eye(4))
	# path = opt.predict_nibable_path
	# nib.save(test, path + '/test.nii.gz')

	output_predict1 = t.cat((output_predict[0], output_predict[2]), 2).data.cpu().numpy()
	output_predict2 = t.cat((output_predict[1], output_predict[3]), 2).data.cpu().numpy()
	output_label1 = t.cat((output_label[0], output_label[2]), 2).data.cpu().numpy()
	output_label2 = t.cat((output_label[1], output_label[3]), 2).data.cpu().numpy()

	print(output_predict1.shape)
	print(output_predict2.shape)
	print('----------------------')
	print(output_label1.shape)
	print(output_label2.shape)

	path = opt.predict_nibable_path

	output_predict1 = nib.Nifti1Image(output_predict1, np.eye(4))
	output_predict2 = nib.Nifti1Image(output_predict2, np.eye(4))
	output_label1 = nib.Nifti1Image(output_label1, np.eye(4))
	output_label2 = nib.Nifti1Image(output_label2, np.eye(4))

	nib.save(output_label1, path + '/label_%s_t.nii.gz' % (opt.task))
	nib.save(output_label2, path + '/label_%s_2_t.nii.gz' % (opt.task))
	nib.save(output_predict1, path + '/predict_%s_t.nii.gz' % (opt.task))
	nib.save(output_predict2, path + '/predict_%s_2_t.nii.gz' % (opt.task))

	print('output_predict: ', output_predict[0].shape, ', ', len(output_predict))

def othertest():
	# a = Variable(t.tensor([[1, 1, 3, 0, 0]]))
	# b = Variable(t.tensor([[0, 1, 1, 1, 1]]))
	# print(score(a, b))



	loss = nn.CrossEntropyLoss()
	input = Variable(t.tensor([
		[ # batch1
			[[[0.1, 0.1], [0.1, 0.1]], [[0.1, 0.1], [0.1, 0.9]]],
			[[[0.2, 0.2], [0.2, 0.2]], [[0.2, 0.2], [0.2, 0.9]]],
			[[[0.3, 0.3], [0.3, 0.3]], [[0.3, 0.3], [0.3, 0.9]]],
			[[[0.4, 0.4], [0.4, 0.4]], [[0.4, 0.4], [0.4, 0.9]]]
		],
		[ # batch2
			[[[0.1, 0.1], [0.1, 0.1]], [[0.1, 0.1], [0.1, 0.9]]],
			[[[0.2, 0.2], [0.2, 0.2]], [[0.2, 0.2], [0.2, 0.9]]],
			[[[0.3, 0.3], [0.3, 0.3]], [[0.3, 0.3], [0.3, 0.9]]],
			[[[0.4, 0.4], [0.4, 0.4]], [[0.4, 0.4], [0.4, 0.9]]]
		]
	]))
	target = Variable(t.tensor([
		[[[1, 2], [3, 0]], [[1, 2], [3, 0]]], # batch1
		[[[1, 2], [3, 0]], [[1, 2], [3, 0]]]  # batch2
	]))
	input = t.reshape(input, (2, 4, -1))
	target = t.reshape(target, (2, -1))
	print(input.shape)
	print(target.shape)
	output = loss(input, target)

	print(output)

def trash():

	img = plt.imread('./trash/img_0_0_5_0.png')
	lbl = plt.imread('./trash/lbl_0_0_5_0.png')
	# print(img)
	# plt.imshow(img, cmap='RdPu')
	plt.imshow(lbl, cmap='RdPu')
	plt.show()

def exercise(**kwargs):
	device = t.device('cuda') if opt.use_gpu else t.device('cpu')

	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0, 1]
	opt._parse(kwargs)

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	print('Start load DataParallels')
	model = nn.DataParallel(model)
	model = model.cuda(device=device_ids[0])
	print('Loading DataParallel finished.')
	if opt.load_model_path:
		model.load_state_dict(t.load('./' + save_dir + opt.load_model_path))
		print('load model')

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

	image = Variable(t.tensor([
		[ # batch1
			[[[0.1, 0.1], [0.1, 0.1]], [[0.1, 0.1], [0.1, 0.9]]],
			[[[0.2, 0.2], [0.2, 0.2]], [[0.2, 0.2], [0.2, 0.9]]],
			[[[0.3, 0.3], [0.3, 0.3]], [[0.3, 0.3], [0.3, 0.9]]],
			[[[0.4, 0.4], [0.4, 0.4]], [[0.4, 0.4], [0.4, 0.9]]]
		],
		[ # batch2
			[[[0.1, 0.1], [0.1, 0.1]], [[0.1, 0.1], [0.1, 0.9]]],
			[[[0.2, 0.2], [0.2, 0.2]], [[0.2, 0.2], [0.2, 0.9]]],
			[[[0.3, 0.3], [0.3, 0.3]], [[0.3, 0.3], [0.3, 0.9]]],
			[[[0.4, 0.4], [0.4, 0.4]], [[0.4, 0.4], [0.4, 0.9]]]
		]
	]))

	label = Variable(t.tensor([
		[[[1, 2], [3, 0]], [[1, 2], [3, 0]]], # batch1
		[[[1, 2], [3, 0]], [[1, 2], [3, 0]]]  # batch2
	]))
	image = image.cuda()
	label = label.cuda()

	for i in range(100):
		optimizer.zero_grad()
		predicts = model(image)
		loss = criterion(predicts, label.long())
		print('loss: ', float(loss))
		loss.backward()
		optimizer.step()

	torch.save(model.state_dict(), os.path.join(save_dir, 'exercise_test_loss.pth'))




	image = t.reshape(image, (2, 4, -1))
	label = t.reshape(label, (2, -1))

	print(image.shape)
	print(image.shape)
	predict = F.softmax(image, dim=1)
	print(predict)


	print(predict.shape)
	print(label.shape)

def ModuleTest1_train(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0, 1]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

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
	train_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=True, task=opt.task)
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
	print(model.eval())

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		model.train()
		for ii, (image, label) in enumerate(train_dataloader):
			img = image.cuda()
			lbl = label.cuda()
			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				lbl_ = lbl[:, :, :, 96 * k:96 * k + 96]
				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				optimizer.zero_grad()
				predicts = model(img_)
				# print('predicts.shape(): ', predicts.shape)
				# print('lbl.shape(): ', lbl_.shape)
				loss = criterion(predicts, lbl_.long())
				train_loss.append(float(loss))
				# print('loss is : ', loss)
				loss.backward()
				optimizer.step()

				# predicts = F.softmax(predicts, dim=1)
				value, tmp = t.max(predicts, dim=1)
				# print('tmp.shape: ', tmp.shape)
				# print('tmp:   ', tmp)
				# predicts = (predicts[:, 1, :, :, :] > 0.5).long()
				d = dice(tmp, lbl_.long())
				# print('d: ', d)
				train_dice.append(d)


		print('train_loss : ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
		print('train_dice : ' + str(sum(train_dice) / (len(train_dice) * 1.0)))
		if not epoch % 5:
			torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__epoch_%s.pth'%(opt.task, epoch)))

	torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__final_epoch.pth' % (opt.task)))

def ModuleTest1_predict(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	task = ['WT', 'TC', 'ET']
	task_model = ['task_WT__final_epoch.pth', 'task_TC__final_epoch.pth', 'task_ET__final_epoch.pth']
	device_ids = [0]
	opt._parse(kwargs)

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'
	print('Start load DataParallels')
	model = nn.DataParallel(model)
	model = model.cuda(device=device_ids[0])
	predictss = [[], [], []]
	predicts_names = []

	save_nii_head_path = opt.val_root_path+'/BraTS19_CBICA_AAM_1//BraTS19_CBICA_AAM_1_flair.nii.gz'
	head_image = nib.load(save_nii_head_path)
	affine = head_image.affine
	box_size = [240, 240, 155]


	for kkk in range(3):
		opt.task = task[kkk]
		opt.load_model_path = task_model[kkk]

		print('Loading DataParallel finished.')
		if opt.load_model_path:
			model.load_state_dict(t.load('./' + save_dir + opt.load_model_path))
			print('load model')

		# pytorch数据处理
		val_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=False, task=opt.task)
		print('train_data and test_data load finished.')
		val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

		print(model.eval())
		i = 0
		for image, name, box_min, box_max in val_dataloader:
			img = image.cuda()
			out_predict = []
			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""
				predicts = model(img_)
				value, tmp = t.max(predicts, dim=1)
				out_predict.append(tmp.int())
			predict = t.cat((out_predict[0], out_predict[1]), 3).data.cpu().numpy()

			pp = np.zeros((155, 240, 240))
			x, y, z = box_min
			pp[x:x+144, y:y+192, z:z+192] = predict[0]
			predict = np.transpose(pp, [2, 1, 0])
			predictss[kkk].append(predict)
			predicts_names.append(name[0])

	predictss = np.array(predictss)
	predicts_names = np.array(predicts_names)
	path = opt.predict_nibable_path
	for i in range(len(predictss[0])):
		multi_predict = (predictss[0][i, :, :, :]>0)*(predictss[1][i, :, :, :]<=0)*(predictss[2][i, :, :, :]<=0)*1.0 + (predictss[1][i, :, :, :]>0)*(predictss[2][i, :, :, :]<=0)*2.0 + (predictss[2][i, :, :, :]>0)*3.0
		if not i:
			print(multi_predict)
		output = nib.Nifti1Image(multi_predict, affine)
		nib.save(output, path + predicts_names[i] + '.nii.gz')

def ModuleTest1_multi_train(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0, 1, 2]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

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
	train_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=True, task=opt.task)
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
	print(model.eval())

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		model.train()
		for ii, (image, label) in enumerate(train_dataloader):
			img = image.cuda()
			lbl = label.cuda()
			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				lbl_ = lbl[:, :, :, 96 * k:96 * k + 96]
				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				optimizer.zero_grad()
				predicts = model(img_)
				# print('predicts.shape(): ', predicts.shape)
				# print('lbl.shape(): ', lbl_.shape)
				loss = criterion(predicts, lbl_.long())
				train_loss.append(float(loss))
				# print('loss is : ', loss)
				loss.backward()
				optimizer.step()

				# predicts = F.softmax(predicts, dim=1)
				value, tmp = t.max(predicts, dim=1)
				# print('tmp.shape: ', tmp.shape)
				# print('tmp:   ', tmp)
				# predicts = (predicts[:, 1, :, :, :] > 0.5).long()
				d = dice(tmp, lbl_.long())
				# print('d: ', d)
				train_dice.append(d)


		print('train_loss : ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
		print('train_dice : ' + str(sum(train_dice) / (len(train_dice) * 1.0)))
		if not epoch % 5:
			torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__epoch_%s.pth'%(opt.task, epoch)))

	torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__final_epoch.pth' % (opt.task)))

def ModuleTest1_multi_predict(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

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
	val_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=False, task=opt.task)
	# test_data = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=False)
	print('train_data and test_data load finished.')

	val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
	# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
	print('train and test dataloader load finished.')

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
	best_dice = -1
	best_epoch = -1
	print('criterion and optimizer is finished.')
	print(model.eval())

	predictss = []
	predicts_names = []

	save_nii_head_path = opt.val_root_path + '/BraTS19_CBICA_AAM_1//BraTS19_CBICA_AAM_1_flair.nii.gz'
	head_image = nib.load(save_nii_head_path)
	affine = head_image.affine

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		for image, name, box_min, box_max in val_dataloader:
			img = image.cuda()
			out_predict = []
			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				predicts = model(img_)

				value, tmp = t.max(predicts, dim=1)
				out_predict.append(tmp.int())
			predict = t.cat((out_predict[0], out_predict[1]), 3).data.cpu().numpy()

			pp = np.zeros((155, 240, 240))
			x, y, z = box_min
			pp[x:x + 144, y:y + 192, z:z + 192] = predict[0]
			predict = np.transpose(pp, [2, 1, 0])
			predictss.append(predict)
			predicts_names.append(name[0])

	predictss = np.array(predictss)
	predicts_names = np.array(predicts_names)
	path = opt.predict_nibable_path
	for i in range(len(predictss)):
		output = nib.Nifti1Image(predictss[i], affine)
		nib.save(output, path + predicts_names[i] + '.nii.gz')

def unet3d_multiclass_train(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0, 1, 2, 3]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

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
	train_data = BraTS2019_Multi(opt.train_root_path, opt.val_root_path, is_train=True, task=opt.task)
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
	print(model.eval())

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		model.train()
		for ii, (image, label) in enumerate(train_dataloader):
			img = image.cuda()
			lbl = label.cuda()
			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				lbl_ = lbl[:, :, :, 96 * k:96 * k + 96]
				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				optimizer.zero_grad()
				predicts = model(img_)
				# print('predicts.shape(): ', predicts.shape)
				# print('lbl.shape(): ', lbl_.shape)
				loss = criterion(predicts, lbl_.long())
				train_loss.append(float(loss))
				# print('loss is : ', loss)
				loss.backward()
				optimizer.step()

				# predicts = F.softmax(predicts, dim=1)
				value, tmp = t.max(predicts, dim=1)
				# print('tmp.shape: ', tmp.shape)
				# print('tmp:   ', tmp)
				# predicts = (predicts[:, 1, :, :, :] > 0.5).long()
				d = dice(tmp, lbl_.long())
				# print('d: ', d)
				train_dice.append(d)


		print('train_loss : ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
		print('train_dice : ' + str(sum(train_dice) / (len(train_dice) * 1.0)))
		if not epoch % 5:
			torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__epoch_%s.pth'%(opt.task, epoch)))

	torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__final_epoch.pth' % (opt.task)))

def unet3d_multiclass_predict(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

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
	val_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=False, task=opt.task)
	# test_data = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=False)
	print('train_data and test_data load finished.')

	val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
	# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
	print('train and test dataloader load finished.')

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
	best_dice = -1
	best_epoch = -1
	print('criterion and optimizer is finished.')
	print(model.eval())

	predictss = []
	predicts_names = []

	save_nii_head_path = opt.val_root_path + '/BraTS19_CBICA_AAM_1//BraTS19_CBICA_AAM_1_flair.nii.gz'
	head_image = nib.load(save_nii_head_path)
	affine = head_image.affine

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		for image, name, box_min, box_max in val_dataloader:
			img = image.cuda()
			out_predict = []
			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				predicts = model(img_)

				value, tmp = t.max(predicts, dim=1)
				out_predict.append(tmp.int())
			predict = t.cat((out_predict[0], out_predict[1]), 3).data.cpu().numpy()

			pp = np.zeros((155, 240, 240))
			x, y, z = box_min
			pp[x:x + 144, y:y + 192, z:z + 192] = predict[0]
			predict = np.transpose(pp, [2, 1, 0])
			predictss.append(predict)
			predicts_names.append(name[0])

	predictss = np.array(predictss)
	predicts_names = np.array(predicts_names)
	path = opt.predict_nibable_path
	for i in range(len(predictss)):
		output = nib.Nifti1Image(predictss[i], affine)
		nib.save(output, path + predicts_names[i] + '.nii.gz')

def ModuleTest2_multi_train(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0, 1, 2, 3]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

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
	train_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=True, task=opt.task)
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
	print(model.eval())

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		model.train()
		for ii, (image, label) in enumerate(train_dataloader):
			img = image.cuda()
			lbl = label.cuda()
			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				lbl_ = lbl[:, :, :, 96 * k:96 * k + 96]
				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				optimizer.zero_grad()
				predicts = model(img_)
				# print('predicts.shape(): ', predicts.shape)
				# print('lbl.shape(): ', lbl_.shape)
				loss = criterion(predicts, lbl_.long())
				train_loss.append(float(loss))
				# print('loss is : ', loss)
				loss.backward()
				optimizer.step()

				# predicts = F.softmax(predicts, dim=1)
				value, tmp = t.max(predicts, dim=1)
				# print('tmp.shape: ', tmp.shape)
				# print('tmp:   ', tmp)
				# predicts = (predicts[:, 1, :, :, :] > 0.5).long()
				d = dice(tmp, lbl_.long())
				# print('d: ', d)
				train_dice.append(d)


		print('train_loss : ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
		print('train_dice : ' + str(sum(train_dice) / (len(train_dice) * 1.0)))
		if not epoch % 5:
			torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__epoch_%s.pth'%(opt.task, epoch)))

	torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__final_epoch.pth' % (opt.task)))

def ModuleTest2_multi_predict(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

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
	val_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=False, task=opt.task)
	# test_data = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=False)
	print('train_data and test_data load finished.')

	val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
	# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
	print('train and test dataloader load finished.')

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
	best_dice = -1
	best_epoch = -1
	print('criterion and optimizer is finished.')
	print(model.eval())

	predictss = []
	predicts_names = []

	save_nii_head_path = opt.val_root_path + '/BraTS19_CBICA_AAM_1//BraTS19_CBICA_AAM_1_flair.nii.gz'
	# save_nii_head_path = opt.val_root_path + '/Brats17_CBICA_AAM_1//Brats17_CBICA_AAM_1_flair.nii.gz'
	head_image = nib.load(save_nii_head_path)
	affine = head_image.affine

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		for image, name, box_min, box_max in val_dataloader:
			img = image.cuda()
			out_predict = []
			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				predicts = model(img_)

				value, tmp = t.max(predicts, dim=1)
				out_predict.append(tmp.int())
			predict = t.cat((out_predict[0], out_predict[1]), 3).data.cpu().numpy()

			pp = np.zeros((155, 240, 240))
			x, y, z = box_min
			pp[x:x + 144, y:y + 192, z:z + 192] = predict[0]
			predict = np.transpose(pp, [2, 1, 0])
			predictss.append(predict)
			predicts_names.append(name[0])

	predictss = np.array(predictss)
	predicts_names = np.array(predicts_names)
	path = opt.predict_nibable_path
	for i in range(len(predictss)):
		output = nib.Nifti1Image(predictss[i], affine)
		nib.save(output, path + predicts_names[i] + '.nii.gz')

def ModuleTest3_multi_train(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0, 1, 2, 3]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

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
	train_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=True, task=opt.task)
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
	print(model.eval())

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		model.train()
		for ii, (image, label) in enumerate(train_dataloader):
			img = image.cuda()
			lbl = label.cuda()
			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				lbl_ = lbl[:, :, :, 96 * k:96 * k + 96]
				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				optimizer.zero_grad()
				predicts = model(img_)
				# print('predicts.shape(): ', predicts.shape)
				# print('lbl.shape(): ', lbl_.shape)
				loss = criterion(predicts, lbl_.long())
				train_loss.append(float(loss))
				# print('loss is : ', loss)
				loss.backward()
				optimizer.step()

				# predicts = F.softmax(predicts, dim=1)
				value, tmp = t.max(predicts, dim=1)
				# print('tmp.shape: ', tmp.shape)
				# print('tmp:   ', tmp)
				# predicts = (predicts[:, 1, :, :, :] > 0.5).long()
				d = dice(tmp, lbl_.long())
				# print('d: ', d)
				train_dice.append(d)


		print('train_loss : ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
		print('train_dice : ' + str(sum(train_dice) / (len(train_dice) * 1.0)))
		if not epoch % 5:
			torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__epoch_%s.pth'%(opt.task, epoch)))

	torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__final_epoch.pth' % (opt.task)))

def ModuleTest3_multi_predict(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

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
	val_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=False, task=opt.task)
	# test_data = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=False)
	print('train_data and test_data load finished.')

	val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
	# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
	print('train and test dataloader load finished.')

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
	best_dice = -1
	best_epoch = -1
	print('criterion and optimizer is finished.')
	print(model.eval())

	predictss = []
	predicts_names = []

	save_nii_head_path = opt.val_root_path + '/BraTS19_CBICA_AAM_1//BraTS19_CBICA_AAM_1_flair.nii.gz'
	# save_nii_head_path = opt.val_root_path + '/Brats17_CBICA_AAM_1//Brats17_CBICA_AAM_1_flair.nii.gz'
	head_image = nib.load(save_nii_head_path)
	affine = head_image.affine

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		for image, name, box_min, box_max in val_dataloader:
			img = image.cuda()
			out_predict = []
			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				predicts = model(img_)

				value, tmp = t.max(predicts, dim=1)
				out_predict.append(tmp.int())
			predict = t.cat((out_predict[0], out_predict[1]), 3).data.cpu().numpy()

			pp = np.zeros((155, 240, 240))
			x, y, z = box_min
			pp[x:x + 144, y:y + 192, z:z + 192] = predict[0]
			predict = np.transpose(pp, [2, 1, 0])
			predictss.append(predict)
			predicts_names.append(name[0])

	predictss = np.array(predictss)
	predicts_names = np.array(predicts_names)
	path = opt.predict_nibable_path
	for i in range(len(predictss)):
		output = nib.Nifti1Image(predictss[i], affine)
		nib.save(output, path + predicts_names[i] + '.nii.gz')

def ModuleTest4_multi_train(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0, 1, 2, 3]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

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
	train_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=True, task=opt.task)
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
	print(model.eval())

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		model.train()
		for ii, (image, label) in enumerate(train_dataloader):
			img = image.cuda()
			lbl = label.cuda()
			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				lbl_ = lbl[:, :, :, 96 * k:96 * k + 96]
				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				optimizer.zero_grad()
				predicts = model(img_)
				# print('predicts.shape(): ', predicts.shape)
				# print('lbl.shape(): ', lbl_.shape)
				loss = criterion(predicts, lbl_.long())
				train_loss.append(float(loss))
				# print('loss is : ', loss)
				loss.backward()
				optimizer.step()

				# predicts = F.softmax(predicts, dim=1)
				value, tmp = t.max(predicts, dim=1)
				# print('tmp.shape: ', tmp.shape)
				# print('tmp:   ', tmp)
				# predicts = (predicts[:, 1, :, :, :] > 0.5).long()
				d = dice(tmp, lbl_.long())
				# print('d: ', d)
				train_dice.append(d)


		print('train_loss : ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
		print('train_dice : ' + str(sum(train_dice) / (len(train_dice) * 1.0)))
		if not epoch % 5:
			torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__epoch_%s.pth'%(opt.task, epoch)))

	torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__final_epoch.pth' % (opt.task)))

def ModuleTest4_multi_predict(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

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
	val_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=False, task=opt.task)
	# test_data = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=False)
	print('train_data and test_data load finished.')

	val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
	# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
	print('train and test dataloader load finished.')

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
	best_dice = -1
	best_epoch = -1
	print('criterion and optimizer is finished.')
	print(model.eval())

	predictss = []
	predicts_names = []

	save_nii_head_path = opt.val_root_path + '/BraTS19_CBICA_AAM_1//BraTS19_CBICA_AAM_1_flair.nii.gz'
	# save_nii_head_path = opt.val_root_path + '/Brats17_CBICA_AAM_1//Brats17_CBICA_AAM_1_flair.nii.gz'
	head_image = nib.load(save_nii_head_path)
	affine = head_image.affine

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		for image, name, box_min, box_max in val_dataloader:
			img = image.cuda()
			out_predict = []
			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				predicts = model(img_)

				value, tmp = t.max(predicts, dim=1)
				out_predict.append(tmp.int())
			predict = t.cat((out_predict[0], out_predict[1]), 3).data.cpu().numpy()

			pp = np.zeros((155, 240, 240))
			x, y, z = box_min
			pp[x:x + 144, y:y + 192, z:z + 192] = predict[0]
			predict = np.transpose(pp, [2, 1, 0])
			predictss.append(predict)
			predicts_names.append(name[0])

	predictss = np.array(predictss)
	predicts_names = np.array(predicts_names)
	path = opt.predict_nibable_path
	for i in range(len(predictss)):
		output = nib.Nifti1Image(predictss[i], affine)
		nib.save(output, path + predicts_names[i] + '.nii.gz')

def ModuleTest5_multi_train(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0, 1, 2, 3]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

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
	train_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=True, task=opt.task)
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
	print(model.eval())

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		model.train()
		for ii, (image, label) in enumerate(train_dataloader):
			img = image.cuda()
			lbl = label.cuda()
			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				lbl_ = lbl[:, :, :, 96 * k:96 * k + 96]
				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				optimizer.zero_grad()
				predicts = model(img_)
				# print('predicts.shape(): ', predicts.shape)
				# print('lbl.shape(): ', lbl_.shape)
				loss = criterion(predicts, lbl_.long())
				train_loss.append(float(loss))
				# print('loss is : ', loss)
				loss.backward()
				optimizer.step()

				# predicts = F.softmax(predicts, dim=1)
				value, tmp = t.max(predicts, dim=1)
				# print('tmp.shape: ', tmp.shape)
				# print('tmp:   ', tmp)
				# predicts = (predicts[:, 1, :, :, :] > 0.5).long()
				d = dice(tmp, lbl_.long())
				# print('d: ', d)
				train_dice.append(d)


		print('train_loss : ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
		print('train_dice : ' + str(sum(train_dice) / (len(train_dice) * 1.0)))
		if not epoch % 5:
			torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__epoch_%s.pth'%(opt.task, epoch)))

	torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__final_epoch.pth' % (opt.task)))

def ModuleTest5_multi_predict(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

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
	val_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=False, task=opt.task)
	# test_data = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=False)
	print('train_data and test_data load finished.')

	val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
	# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
	print('train and test dataloader load finished.')

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
	best_dice = -1
	best_epoch = -1
	print('criterion and optimizer is finished.')
	print(model.eval())

	predictss = []
	predicts_names = []

	save_nii_head_path = opt.val_root_path + '/BraTS19_CBICA_AAM_1//BraTS19_CBICA_AAM_1_flair.nii.gz'
	# save_nii_head_path = opt.val_root_path + '/Brats17_CBICA_AAM_1//Brats17_CBICA_AAM_1_flair.nii.gz'
	head_image = nib.load(save_nii_head_path)
	affine = head_image.affine

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		for image, name, box_min, box_max in val_dataloader:
			img = image.cuda()
			out_predict = []
			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				predicts = model(img_)

				value, tmp = t.max(predicts, dim=1)
				out_predict.append(tmp.int())
			predict = t.cat((out_predict[0], out_predict[1]), 3).data.cpu().numpy()

			pp = np.zeros((155, 240, 240))
			x, y, z = box_min
			pp[x:x + 144, y:y + 192, z:z + 192] = predict[0]
			predict = np.transpose(pp, [2, 1, 0])
			predictss.append(predict)
			predicts_names.append(name[0])

	predictss = np.array(predictss)
	predicts_names = np.array(predicts_names)
	path = opt.predict_nibable_path
	for i in range(len(predictss)):
		output = nib.Nifti1Image(predictss[i], affine)
		nib.save(output, path + predicts_names[i] + '.nii.gz')

def ModuleTest_multi_train(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0, 1, 2, 3]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

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
	train_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=True, task=opt.task)
	# print(opt.train_root_path)
	# print(len(train_data))
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
	print(model.eval())

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		model.train()
		for ii, (image, label) in enumerate(train_dataloader):
			img = image.cuda()
			lbl = label.cuda()
			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				lbl_ = lbl[:, :, :, 96 * k:96 * k + 96]
				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				optimizer.zero_grad()
				predicts = model(img_)
				# print('predicts.shape(): ', predicts.shape)
				# print('lbl.shape(): ', lbl_.shape)
				loss = criterion(predicts, lbl_.long())
				train_loss.append(float(loss))
				# print('loss is : ', loss)
				loss.backward()
				optimizer.step()

				# predicts = F.softmax(predicts, dim=1)
				value, tmp = t.max(predicts, dim=1)
				# print('tmp.shape: ', tmp.shape)
				# print('tmp:   ', tmp)
				# predicts = (predicts[:, 1, :, :, :] > 0.5).long()
				d = dice(tmp, lbl_.long())
				# print('d: ', d)
				train_dice.append(d)

		print('train_loss : ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
		print('train_dice : ' + str(sum(train_dice) / (len(train_dice) * 1.0)))
		if not epoch % 5:
			torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__epoch_%s.pth'%(opt.task, epoch)))

	torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__final_epoch.pth' % (opt.task)))

def ModuleTest_multi_predict(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

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
	val_data = BraTS2019(opt.train_root_path, opt.val_root_path, is_train=False, task=opt.task)
	# test_data = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=False)
	print('train_data and test_data load finished.')

	val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
	# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
	print('train and test dataloader load finished.')

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
	best_dice = -1
	best_epoch = -1
	print('criterion and optimizer is finished.')
	print(model.eval())

	predictss = []
	predicts_names = []

	save_nii_head_path = opt.val_root_path + '/BraTS19_CBICA_AAM_1//BraTS19_CBICA_AAM_1_flair.nii.gz'
	# save_nii_head_path = opt.val_root_path + '/Brats17_CBICA_AAM_1//Brats17_CBICA_AAM_1_flair.nii.gz'
	head_image = nib.load(save_nii_head_path)
	affine = head_image.affine

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		for image, name, box_min, box_max in val_dataloader:
			img = image.cuda()
			out_predict = []
			for k in range(2):
				img_ = img[:, :, :, :, 96 * k:96 * k + 96]
				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				predicts = model(img_)

				value, tmp = t.max(predicts, dim=1)
				out_predict.append(tmp.int())
			predict = t.cat((out_predict[0], out_predict[1]), 3).data.cpu().numpy()

			pp = np.zeros((155, 240, 240))
			x, y, z = box_min
			pp[x:x + 144, y:y + 192, z:z + 192] = predict[0]
			predict = np.transpose(pp, [2, 1, 0])
			predictss.append(predict)
			predicts_names.append(name[0])

	predictss = np.array(predictss)
	predicts_names = np.array(predicts_names)
	path = opt.predict_nibable_path
	for i in range(len(predictss)):
		output = nib.Nifti1Image(predictss[i], affine)
		nib.save(output, path + predicts_names[i] + '.nii.gz')

def Aneu_train(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0, 1, 2, 3]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)(1, 2)
	print('model is : ', opt.model)
	save_dir = 'ckpt_aneu_' + opt.model + '/'

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	print('Start load DataParallels')
	model = nn.DataParallel(model)
	model = model.cuda(device=device_ids[0])
	print('Loading DataParallel finished.')
	if opt.load_model_path:
		model.load_state_dict(t.load('./' + save_dir + opt.load_model_path))
		print('load model')

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------' % (epoch))

		# pytorch数据处理
		train_data = AneuMulti(opt.aneu_path)
		# test_data = AneuMulti(opt.aneu_val_path)
		print('train_data and test_data load finished.')

		train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
		# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
		print('train and test dataloader load finished.')

		criterion = nn.CrossEntropyLoss()
		print('lr: ', opt.lr)
		optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

		print('criterion and optimizer is finished.')
		# print(model.eval())

		train_loss = []
		train_dice = []
		model.train()
		for ii, (image, label, name) in enumerate(train_dataloader):
			for k in range(2):
				img = image[:, k, :, :, :, :]
				lbl = label[:, k, :, :, :]

				for fck in range(7):
					img_ = img[:, :, :, :, 64*fck:64*(fck+1)]
					lbl_ = lbl[:, :, :, 64*fck:64*(fck+1)]
					img_ = img_.cuda()
					lbl_ = lbl_.cuda()

					optimizer.zero_grad()
					predicts = model(img_)
					loss = criterion(predicts, lbl_.long())
					train_loss.append(float(loss))
					loss.backward()
					optimizer.step()

					value, tmp = t.max(predicts, dim=1)
					d = dice(tmp, lbl_.long())
					train_dice.append(d)
		print('train_loss : ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
		print('train_dice : ' + str(sum(train_dice) / (len(train_dice) * 1.0)))

		# test_loss = []
		# test_dice = []
		# model.eval()
		# with t.no_grad():
		# 	for ii, (image, label, name) in enumerate(test_dataloader):
		# 		for k in range(2):
		# 			img = image[:, k, :, :, :, :]
		# 			lbl = label[:, k, :, :, :]
		#
		# 			for fck in range(7):
		# 				img_ = img[:, :, :, :, 64 * fck:64 * fck + 64]
		# 				lbl_ = lbl[:, :, :, 64 * fck:64 * fck + 64]
		# 				img_ = img_.cuda()
		# 				lbl_ = lbl_.cuda()
		#
		# 				predicts = model(img_)
		# 				loss = criterion(predicts, lbl_.long())
		# 				test_loss.append(float(loss))
		#
		# 				value, tmp = t.max(predicts, dim=1)
		# 				d = dice(tmp, lbl_.long())
		# 				test_dice.append(d)
		#
		# 				print('ii_{}, k_{}, fck_{}, test_dice: {} '.format(ii, k, fck, d))
		# 	print('test_loss : ' + str(sum(test_loss) / (len(test_loss) * 1.0)))
		# 	print('test_dice : ' + str(sum(test_dice) / (len(test_dice) * 1.0)))

		torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__epoch_%s.pth'%(opt.task, epoch)))

	torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__final_epoch.pth' % (opt.task)))

def Aneu_predict(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)(in_data=1, out_data=2)
	print('model is : ', opt.model)
	save_dir = 'ckpt_aneu_' + opt.model + '/'

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	path = opt.aneu_output_path
	path += '_' + opt.model + '_' + opt.load_model_path[:-4]

	if not os.path.exists(path):
		os.mkdir(path)

	print('Start load DataParallels')
	model = nn.DataParallel(model)
	model = model.cuda(device=device_ids[0])
	print('Loading DataParallel finished.')
	if opt.load_model_path:
		model.load_state_dict(t.load('./' + save_dir + opt.load_model_path))
		print('load model')

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))

		# pytorch数据处理
		train_data = AneuMulti(opt.aneu_path, is_train=False, val_path=opt.aneu_val_path)
		print('train_data and test_data load finished.')

		train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

		train_loss = []
		train_dice = []
		for ii, (image, label, img_shape, index_min, index_max, affine, name) in enumerate(train_dataloader):
			predict = []
			iimage = []
			for k in range(2):
				img = image[:, k, :, :, :, :]
				img = img.cuda()

				for fck in range(7):
					img_ = img[:, :, :, :, 64*fck:64*fck+64]

					predicts = model(img_)

					value, tmp = t.max(predicts, dim=1)
					predict.append(tmp.int().data.cpu())
					iimage.append(img_.data.cpu())

			predict1 = t.cat((predict[0], predict[1]), dim=3)
			for i in range(2, 7):
				predict1 = t.cat((predict1, predict[i]), dim=3)
			predict2 = t.cat((predict[7], predict[8]), dim=3)
			for i in range(9, 14):
				predict2 = t.cat((predict2, predict[i]), dim=3)
			out_predict = t.cat((predict1, predict2), dim=1).numpy()

			image1 = t.cat((iimage[0], iimage[1]), dim=4)
			for i in range(2, 7):
				image1 = t.cat((image1, iimage[i]), dim=4)
			image2 = t.cat((iimage[7], iimage[8]), dim=4)
			for i in range(9, 14):
				image2 = t.cat((image2, iimage[i]), dim=4)
			out_image = t.cat((image1, image2), dim=2).numpy()
			out_image = out_image[:, 0, :, :, :]

			# [128, 448, 448]
			# print('shape: ', out_image[0].shape)
			out_image = crop_with_box(out_image[0], np.array([0, 0, 0]), img_shape)
			out_predict = crop_with_box(out_predict[0], np.array([0, 0, 0]), img_shape)
			# print('shape_: ', out_image.shape)

			out_image = np.transpose(out_image, [2, 1, 0])
			out_predict = np.transpose(out_predict, [2, 1, 0])

			# affine [0]: batch size
			output = nib.Nifti1Image(out_predict, affine[0])
			nib.save(output, path + '/' + name[0] + '_predict.nii.gz')
			im = nib.Nifti1Image(out_image, affine[0])
			nib.save(im, path + '/' + name[0] + '_image.nii.gz')

def Aneu_test(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)(in_data=1, out_data=2)
	print('model is : ', opt.model)
	save_dir = 'ckpt_aneu_' + opt.model + '/'

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	print('Start load DataParallels')
	model = nn.DataParallel(model)
	model = model.cuda(device=device_ids[0])
	print('Loading DataParallel finished.')
	if opt.load_model_path:
		model.load_state_dict(t.load('./' + save_dir + opt.load_model_path))
		print('load model')

	score = []
	score.append(['Name', 'Dice', 'Sensitivity', 'Specificity'])

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))

		# pytorch数据处理
		train_data = AneuMulti(opt.aneu_path, is_train=True, val_path=opt.aneu_val_path)
		print('train_data and test_data load finished.')

		train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

		for ii, (image, label, name) in enumerate(train_dataloader):
			test_dice = []
			test_sensitivity = []
			test_specificity = []
			predict = []
			iimage = []
			for k in range(2):
				img = image[:, k, :, :, :, :]
				lbl = label[:, k, :, :, :]
				img = img.cuda()
				lbl = lbl.cuda()

				for fck in range(7):
					img_ = img[:, :, :, :, 64*fck:64*fck+64]
					lbl_ = lbl[:, :, :, 64*fck:64*fck+64]
					predicts = model(img_)

					value, tmp = t.max(predicts, dim=1)
					d = dice(tmp, lbl_.long())
					se = sensitivity(tmp, lbl_.long())
					sp = specificity(tmp, lbl_.long())
					predict.append(tmp.int().data.cpu())
					iimage.append(img_.data.cpu())
					test_dice.append(d)
					test_sensitivity.append(se)
					test_specificity.append(sp)
			lbl_dice = sum(test_dice)/len(test_dice)
			lbl_se = sum(test_sensitivity)/len(test_sensitivity)
			lbl_sp = sum(test_specificity)/len(test_specificity)
			score.append([name, str(lbl_dice), str(lbl_se), str(lbl_sp)])
			# if lbl_dice < 0.7:
			print('test dice: {} -->  {}'.format(name, lbl_dice))
			print('test sensitivity: {} -->  {}'.format(name, lbl_se))
			print('test specificity: {} -->  {}'.format(name, lbl_sp))
			# return
	save = pd.DataFrame(score, columns=['Name', 'Dice', 'Sensitivity', 'Specificity'])
	save.to_csv('./' + opt.model + '_' + opt.load_model_path[:-4] + '_multi_score.csv', index=False, header=False)


def hhhhh():
	model = models.Liangliang_Liu(in_data=1, out_data=2)
	data = torch.randn(2, 1, 16, 16, 64)
	out = model(data)

def ModuleTest_multi_train_random(**kwargs):

	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return
	device_ids = [0, 1, 2, 3]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))
	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	print('Start load DataParallels')
	model = nn.DataParallel(model)
	model = model.cuda(device=device_ids[0])
	print('Loading DataParallel finished.')
	if opt.load_model_path:
		model.load_state_dict(t.load('./' + save_dir + opt.load_model_path))
		print('load model')


	best_dice = -1
	best_epoch = -1
	lr = opt.lr
	print('criterion and optimizer is finished.')
	# print(model.eval())

	for kkepoch in range(opt.random_epoch):
		print('----------------------kkepoch %d--------------------' % (kkepoch))

		print('lr: ', lr)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))
		lr *= opt.lr_decay
		# pytorch数据处理
		train_data = BraTS2019_Random(opt.train_root_path, opt.val_root_path, is_train=True, task=opt.task)
		print('train_data and test_data load finished.')

		train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
		# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
		print('train and test dataloader load finished.')

		train_loss = []
		train_dice = []
		model.train()
		for ii, (image, label) in enumerate(train_dataloader):
			# img = image.cuda()
			# lbl = label.cuda()
			for i in range(9):
				img_ = image[:, i, :, :, :, :]
				lbl_ = label[:, i, :, :, :]
				img_ = img_.cuda()
				lbl_ = lbl_.cuda()

				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				optimizer.zero_grad()
				predicts = model(img_)
				# print('predicts.shape(): ', predicts.shape)
				# print('lbl.shape(): ', lbl_.shape)
				loss = criterion(predicts, lbl_.long())
				train_loss.append(float(loss))
				# print('loss is : ', loss)
				loss.backward()
				optimizer.step()

				# predicts = F.softmax(predicts, dim=1)
				value, tmp = t.max(predicts, dim=1)

				d = dice(tmp, lbl_.long())
				# print('d: ', d)
				train_dice.append(d)

		print('train_loss : ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
		print('train_dice : ' + str(sum(train_dice) / (len(train_dice) * 1.0)))
		if not kkepoch % 5:
			torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__epoch_%s.pth'%(opt.task, kkepoch)))

		torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__final_epoch.pth' % (opt.task)))

def ModuleTest_multi_val_random(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

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
	val_data = BraTS2019_Random(opt.train_root_path, opt.val_root_path, is_train=False, task=opt.task)
	# test_data = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=False)
	print('train_data and test_data load finished.')

	val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
	# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
	print('train and test dataloader load finished.')

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
	best_dice = -1
	best_epoch = -1
	print('criterion and optimizer is finished.')
	print(model.eval())

	predictss = []
	predicts_names = []

	save_nii_head_path = opt.val_root_path + '/BraTS19_CBICA_AAM_1/BraTS19_CBICA_AAM_1_flair.nii.gz'
	# save_nii_head_path = opt.val_root_path + '/BraTS19_2013_0_1/BraTS19_2013_0_1_flair.nii.gz'
	head_image = nib.load(save_nii_head_path)
	affine = head_image.affine

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		for image, name, box_min, box_max in val_dataloader:
			img = image.cuda()
			out_predict = []

			for k in range(9):
				img_ = img[:, k, :, :, :]
				predicts = model(img_)
				value, tmp = t.max(predicts, dim=1)
				out_predict.append(tmp.int())
			predict = t.cat((out_predict[0], out_predict[1]), dim=1)
			for k in range(2, 9):
				predict = t.cat((predict, out_predict[k]), dim=1)
			predict = predict.data.cpu().numpy()
			# predict = out_precessing(predict)

			# for k in range(2):
			# 	img_ = img[:, :, :, :, 96 * k:96 * k + 96]
			# 	"""
			# 	img: [batch_size, 4, 144, 192, 192]
			# 	lbl: [batch_size, 144, 192, 192]
			# 	"""
			#
			# 	predicts = model(img_)
			#
			# 	value, tmp = t.max(predicts, dim=1)
			# 	out_predict.append(tmp.int())
			#
			#
			# predict = t.cat((out_predict[0], out_predict[1]), 3).data.cpu().numpy()

			pp = np.zeros((155, 240, 240))
			x, y, z = box_min
			pp[x:x + 144, y:y + 192, z:z + 192] = predict[0]
			predict = np.transpose(pp, [2, 1, 0])
			predictss.append(predict)
			predicts_names.append(name[0])

	predictss = np.array(predictss)
	predicts_names = np.array(predicts_names)
	path = opt.predict_nibable_path
	for i in range(len(predictss)):
		output = nib.Nifti1Image(predictss[i], affine)
		nib.save(output, path + predicts_names[i] + '.nii.gz')

def ModuleTest_multi_train_random_dataarg(**kwargs):

	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return
	device_ids = [0, 1, 2, 3]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))
	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	print('Start load DataParallels')
	model = nn.DataParallel(model)
	model = model.cuda(device=device_ids[0])
	print('Loading DataParallel finished.')
	if opt.load_model_path:
		model.load_state_dict(t.load('./' + save_dir + opt.load_model_path))
		print('load model')

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
	best_dice = -1
	best_epoch = -1
	print('criterion and optimizer is finished.')
	# print(model.eval())

	for kkepoch in range(opt.random_epoch):
		print('----------------------kkepoch %d--------------------' % (kkepoch))
		# pytorch数据处理
		train_data = BraTS2019_Random_DataArg(opt.train_root_path, opt.val_root_path, is_train=True, task=opt.task)
		# print(opt.train_root_path)
		# print(len(train_data))
		# test_data = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=False)
		print('train_data and test_data load finished.')

		train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
		# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
		print('train and test dataloader load finished.')

		# for epoch in range(opt.max_epoch):

		train_loss = []
		train_dice = []
		model.train()
		for ii, (image, label) in enumerate(train_dataloader):
			# img = image.cuda()
			# lbl = label.cuda()
			for i in range(24):
				img_ = image[:, i, :, :, :, :]
				lbl_ = label[:, i, :, :, :]
				img_ = img_.cuda()
				lbl_ = lbl_.cuda()

				"""
				img: [batch_size, 4, 144, 192, 192]
				lbl: [batch_size, 144, 192, 192]
				"""

				optimizer.zero_grad()
				predicts = model(img_)
				# print('predicts.shape(): ', predicts.shape)
				# print('lbl.shape(): ', lbl_.shape)
				loss = criterion(predicts, lbl_.long())
				train_loss.append(float(loss))
				# print('loss is : ', loss)
				loss.backward()
				optimizer.step()

				# predicts = F.softmax(predicts, dim=1)
				value, tmp = t.max(predicts, dim=1)
				# print('tmp.shape: ', tmp.shape)
				# print('tmp:   ', tmp)
				# predicts = (predicts[:, 1, :, :, :] > 0.5).long()
				d = dice(tmp, lbl_.long())
				# print('d: ', d)
				train_dice.append(d)


		print('train_loss : ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
		print('train_dice : ' + str(sum(train_dice) / (len(train_dice) * 1.0)))
			# if not epoch % 5:
			# 	torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__epoch_%s.pth'%(opt.task, epoch)))

		torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__final_epoch.pth' % (opt.task)))


def dataset_clean(**kwargs):
	opt._parse(kwargs)
	# pytorch数据处理
	train_data = AneuMulti(opt.aneu_path)
	train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
	print('train and test dataloader load finished.')
	for ii, (image, label) in enumerate(train_dataloader):
		print('image: ', ii)


def made():
	data = np.array([[1, 2, 3], [1, 2, 2], [1, 2, 33]])
	data.append([3, 4, 5])

	save = pd.DataFrame(data, columns=['l1', 'l2', 'l3'])
	save.to_csv('./multi_score.csv', index=False, header=False)


def restest():
	print(64//4 * 448//4 * 64 // 4)
	# a = t.autograd.Variable(t.randn([2, 1, 64, 448, 64]))
	# model = models.Resnet()
	# model(a)


if __name__ == '__main__':
	import fire

	fire.Fire()

