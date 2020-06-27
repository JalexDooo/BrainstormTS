#!/bin/bash

# date1=`date +%s`

# traing
#train_loss : 0.003706652016554523
#train_dice : 0.9049327418473818
#cd /home/sunjindong/BrainstormTS && python main.py brats2019_train --task='WT' --model='MultiscaleUNet3D' --batch_size=2 --max_epoch=61 --load_model_path='task_WT__epoch_13.pth'
#cd /home/sunjindong/BrainstormTS && python main.py brats2019_test --task='WT' --model='MultiscaleUNet3D' --batch_size=2 --max_epoch=1 --load_model_path='task_WT__epoch_13.pth'

# 0.8693697297274539
#cd /home/sunjindong/BrainstormTS && python main.py brats2019_train --task='TC' --model='MultiscaleUNet3D' --batch_size=2 --max_epoch=61 --load_model_path='task_TC__epoch_9.pth'
#cd /home/sunjindong/BrainstormTS && python main.py brats2019_test --task='TC' --model='MultiscaleUNet3D' --batch_size=2 --max_epoch=1 --load_model_path='task_TC__epoch_9.pth'

#train_loss : 0.0015702718355108708
#train_dice : 0.8392257473288413
#cd /home/sunjindong/BrainstormTS && python main.py brats2019_train --task='ET' --model='MultiscaleUNet3D' --batch_size=2 --max_epoch=20 --load_model_path='task_ET__epoch_19.pth'
#cd /home/sunjindong/BrainstormTS && python main.py brats2019_test --task='ET' --model='MultiscaleUNet3D' --batch_size=2 --max_epoch=1 --load_model_path='task_ET__epoch_19.pth'

#cd /home/sunjindong/BrainstormTS && python main.py brats2019_train --task='NCR' --model='MultiscaleUNet3D' --batch_size=2 --max_epoch=61 #--load_model_path='task_NCR__epoch_60.pth'
#cd /home/sunjindong/BrainstormTS && python main.py brats2019_predict --model='MultiscaleUNet3D' --batch_size=2 --max_epoch=1 --load_model_path='score_epoch_75.pth'

# predicting
#cd /home/sunjindong/BrainstormTS && python main.py brats2019_predict --task='TC' --model='MultiscaleUNet3D' --batch_size=2 --max_epoch=1 --load_model_path='task_TC__epoch_9.pth'
#cd /home/sunjindong/BrainstormTS && python main.py brats2019_predict --model='MultiscaleUNet3D' --batch_size=2 --max_epoch=1 --load_model_path='score_epoch_75.pth'
#cd /home/sunjindong/BrainstormTS && python main.py brats2019_predict --model='MultiscaleUNet3D' --batch_size=2 --max_epoch=1 --load_model_path='score_epoch_75.pth'

#cd /home/sunjindong/BrainstormTS && python main.py brats2019_single_image_predict --task='WT' --model='MultiscaleUNet3D' --load_model_path='task_WT__epoch_20.pth'
#cd /home/sunjindong/BrainstormTS && python main.py brats2019_single_image_predict --task='TC' --model='MultiscaleUNet3D' --load_model_path='task_TC__epoch_9.pth'
#cd /home/sunjindong/BrainstormTS && python main.py brats2019_single_image_predict --task='ET' --model='MultiscaleUNet3D' --load_model_path='task_ET__epoch_19.pth'
#cd /home/sunjindong/BrainstormTS && python main.py brats2019_single_image_predict --task='WT' --model='MultiscaleUNet3D' --load_model_path='task_WT__epoch_13.pth'

# task 2173 2217 2220 2234 #############
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest1_multi_train --task='ALL' --model='ModuleTest1' --max_epoch=151 --batch_size=3 --load_model_path='task_ALL__final_epoch.pth'

#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest1_multi_predict --task='ALL' --model='ModuleTest1' --max_epoch=1 --batch_size=1 --load_model_path='task_ALL__final_epoch.pth' --predict_nibable_path='./brats2019_val_multitest1/'


# 90.4 task 1573
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest1_train --task='WT' --model='ModuleTest1' --max_epoch=71 --load_model_path='task_WT__final_epoch.pth'

# 85.9 task 1605
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest1_train --task='TC' --model='ModuleTest1' --max_epoch=71 --load_model_path='task_TC__final_epoch.pth'

# 84.7 task 1664 and task 1679
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest1_train --task='ET' --model='ModuleTest1' --max_epoch=71 --load_model_path='task_ET__final_epoch.pth'

#cd /home/sunjindong/BrainstormTS && python main.py brats2019_single_image_predict --task='WT' --model='ModuleTest1' --load_model_path='task_WT__final_epoch.pth'

# the-state-of-art task 1963 1977 166
#cd /home/sunjindong/BrainstormTS && python main.py unet3d_multiclass_train --task='ALL' --model='UNet3D' --max_epoch=101 --load_model_path='task_ALL__final_epoch.pth'
#''' UNet3D
#----------------------epoch 70--------------------
#train_loss : 0.01513150871579412
#train_dice : 2.163642768419485
#'''

# 预测 ModuleTest1
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest1_predict --model='ModuleTest1' --max_epoch=1 --batch_size=1 --predict_nibable_path='./brats2019_validation/'

# 预测 Unet3d
#cd /home/sunjindong/BrainstormTS && python main.py unet3d_multiclass_predict --task='ALL' --model='UNet3D' --max_epoch=1 --batch_size=1 --load_model_path='task_ALL__final_epoch.pth' --predict_nibable_path='./brats2019_val_unet3d/'



# ModuleTest2 training 修改,添加Maxpooling组件
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest2_multi_train --task='ALL' --model='ModuleTest2' --max_epoch=151 --batch_size=8 --load_model_path='task_ALL__final_epoch.pth'
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest2_multi_predict --task='ALL' --model='ModuleTest2' --max_epoch=1 --batch_size=1 --load_model_path='task_ALL__final_epoch.pth' --predict_nibable_path='./brats2019_val_moduletest2/'

# ModuleTest3 training 修改反卷积层Transconvolutional
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest3_multi_train --task='ALL' --model='ModuleTest3' --max_epoch=101 --batch_size=8 --load_model_path='task_ALL__final_epoch.pth'
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest3_multi_predict --task='ALL' --model='ModuleTest3' --max_epoch=1 --batch_size=1 --load_model_path='task_ALL__final_epoch.pth' --predict_nibable_path='./brats2019_val_moduletest3/'


# ModuleTest4 training 修改特征融合方式 task 1107
#cd /home/test/BrainstormTS && python main.py ModuleTest4_multi_train --task='ALL' --model='ModuleTest4' --max_epoch=151 --batch_size=4 --load_model_path='task_ALL__final_epoch.pth' --train_root_path = '/home/test/dataset/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training'
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest4_multi_predict --task='ALL' --model='ModuleTest4' --max_epoch=1 --batch_size=1 --load_model_path='task_ALL__final_epoch.pth' --predict_nibable_path='./brats2019_val_moduletest4/'
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest4_multi_predict --task='ALL' --model='ModuleTest4' --max_epoch=1 --batch_size=1 --load_model_path='task_ALL__epoch_35.pth' --predict_nibable_path='./brats2019_val_moduletest4/'


# ModuleTest5 training 删除Dilated convolution
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest5_multi_train --task='ALL' --model='ModuleTest5' --max_epoch=101 --batch_size=4 --load_model_path='task_ALL__final_epoch.pth'
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest5_multi_predict --task='ALL' --model='ModuleTest5' --max_epoch=1 --batch_size=1 --load_model_path='task_ALL__final_epoch.pth' --predict_nibable_path='./brats2019_val_moduletest5/'

# ModuleTest6 training ModuleTest6 task 1521 1752
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest_multi_train --task='ALL' --model='ModuleTest6' --max_epoch=101 --batch_size=4 --load_model_path='task_ALL__final_epoch.pth'
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest_multi_predict --task='ALL' --model='ModuleTest6' --max_epoch=1 --batch_size=1 --load_model_path='task_ALL__final_epoch.pth' --predict_nibable_path='./brats2019_val_moduletest6/'
# Unet-101 task 1661
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest_multi_train --task='ALL' --model='UNet3D' --max_epoch=101 --batch_size=4 #--load_model_path='task_ALL__final_epoch.pth'

#ModuleTest6 test task 1818
#cd /home/test/BrainstormTS && python main.py ModuleTest_multi_train --task='ALL' --model='ModuleTest6' --max_epoch=101 --batch_size=4 --load_model_path='task_ALL__final_epoch.pth' --train_root_path='/home/test/dateset/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training'
#cd /home/test/BrainstormTS && python main.py ModuleTest_multi_predict --task='ALL' --model='ModuleTest6' --max_epoch=1 --batch_size=1 --load_model_path='task_ALL__epoch_90.pth' --predict_nibable_path='./brats2019_val_moduletest6/' --val_root_path='/home/test/dateset/MICCAI_BraTS_2019_Data_Validation/MICCAI_BraTS_2019_Data_Validation'
# ModuleTest5 training test
#cd /home/test/BrainstormTS && python main.py ModuleTest_multi_train_random --task='Random' --model='ModuleTest5' --random_epoch=30 --max_epoch=1 --batch_size=20 --train_root_path='/home/test/dateset/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training' --load_model_path='task_Random__final_epoch.pth'
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest_multi_val_random --task='Random' --model='ModuleTest5' --max_epoch=1 --batch_size=1  --load_model_path='task_Random__final_epoch.pth' --predict_nibable_path='./brats2019_train_moduletest5_random/' --val_root_path='/home/sunjindong/dataset/MICCAI_BraTS19_Training_Val'

# ModuleTest6 training sunjindong
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest_multi_train_random --task='Random' --model='ModuleTest6' --random_epoch=40 --max_epoch=1 --batch_size=32  --load_model_path='task_Random__final_epoch.pth'
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest_multi_val_random --task='Random' --model='ModuleTest6' --max_epoch=1 --batch_size=1  --load_model_path='task_Random__final_epoch.pth' --predict_nibable_path='./brats2019_val_moduletest6_random/'

# Aneu_traing
cd /home/aneu/Brainstorm && python main.py Aneu_train --task='Random' --model='ModuleTest6' --max_epoch=11 --batch_size=8 --lr=0.001 --load_model_path='task_Random__final_epoch.pth'
#cd /home/aneu/Brainstorm && python main.py Aneu_predict --task='Random' --model='ModuleTest6' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__final_epoch.pth'


# ModuleTest2 training
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest_multi_train_random --task='Random' --model='ModuleTest2' --random_epoch=160 --max_epoch=1 --batch_size=32  --load_model_path='task_Random__final_epoch.pth'
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest_multi_val_random --task='Random' --model='ModuleTest2' --max_epoch=1 --batch_size=1  --load_model_path='task_Random__final_epoch.pth' --predict_nibable_path='./brats2019_val_moduletest2_random/'

# ModuleTest3 training
#cd /home/test/BrainstormTS && python main.py ModuleTest_multi_train_random --task='Random' --model='ModuleTest3' --random_epoch=160 --max_epoch=1 --batch_size=32 --train_root_path='/home/test/dateset/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training' --load_model_path='task_Random__final_epoch.pth'
#cd /home/test/BrainstormTS && python main.py ModuleTest_multi_val_random --task='Random' --model='ModuleTest3' --max_epoch=1 --batch_size=1  --load_model_path='task_Random__final_epoch.pth' --predict_nibable_path='./brats2019_val_moduletest3_random/' --val_root_path='/home/test/dateset/MICCAI_BraTS_2019_Data_Validation/MICCAI_BraTS_2019_Data_Validation'

#Module ResUnet
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest_multi_train_random_dataarg --task='Random' --model='ResUnet' --random_epoch=50 --max_epoch=1 --batch_size=12 --lr=0.002  --load_model_path='task_Random__final_epoch.pth'
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest_multi_val_random --task='Random' --model='ResUnet' --max_epoch=1 --batch_size=1  --load_model_path='task_Random__final_epoch.pth' --predict_nibable_path='./brats2019_val_resunet_random/'

# dataarg
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest_multi_train_random_dataarg --task='Random' --model='ModuleTest5' --random_epoch=100 --max_epoch=1 --batch_size=32 --lr=0.002  #--load_model_path='task_Random__final_epoch.pth'
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest_multi_val_random --task='Random' --model='ModuleTest5' --max_epoch=1 --batch_size=1  --load_model_path='task_Random__final_epoch.pth' --predict_nibable_path='./brats2019_val_moduletest5_random/'

#----------------------epoch 100--------------------
#train_loss : 0.008762094631276255
#train_dice : 2.294680844825654

#'''
#----------------------epoch 0--------------------
#train_loss : 1.2705800973233723
#train_dice : 0.09219573420141638
#----------------------epoch 1--------------------
#train_loss : 1.0654016445790018
#train_dice : 0.13380535554229966
#----------------------epoch 2--------------------
#train_loss : 0.7977348129664149
#train_dice : 0.3762021396783491
#----------------------epoch 3--------------------
#train_loss : 0.5933932896171298
#train_dice : 0.6056086931365016
#----------------------epoch 4--------------------
#train_loss : 0.4615835974968615
#train_dice : 0.7845062489695461
#----------------------epoch 5--------------------
#train_loss : 0.3695848530956677
#train_dice : 0.9385944183530825
#----------------------epoch 6--------------------
#train_loss : 0.30017769673750516
#train_dice : 1.0465725001502653
#----------------------epoch 7--------------------
#train_loss : 0.2493245762196325
#train_dice : 1.2137247546209935
#----------------------epoch 8--------------------
#train_loss : 0.20988648242893673
#train_dice : 1.2676977551350526
#----------------------epoch 9--------------------
#train_loss : 0.182906730898789
#train_dice : 1.3436272539632388
#----------------------epoch 10--------------------
#train_loss : 0.15606935065062272
#train_dice : 1.438694378930143
#----------------------epoch 11--------------------
#train_loss : 0.13708214306583008
#train_dice : 1.5126779820300964
#----------------------epoch 12--------------------
#train_loss : 0.12125610697659708
#train_dice : 1.605982662756667
#----------------------epoch 13--------------------
#train_loss : 0.10753031886581864
#train_dice : 1.633648556545792
#----------------------epoch 14--------------------
#train_loss : 0.09647925481909797
#train_dice : 1.6543583330909928
#----------------------epoch 15--------------------
#train_loss : 0.08668223422552858
#train_dice : 1.735876295071792
#----------------------epoch 16--------------------
#train_loss : 0.07905381518815245
#train_dice : 1.7086563708987446
#----------------------epoch 17--------------------
#train_loss : 0.07094333529294956
#train_dice : 1.7327626455837475
#----------------------epoch 18--------------------
#train_loss : 0.06593162670642846
#train_dice : 1.7724091561939677
#----------------------epoch 19--------------------
#train_loss : 0.060656071574028046
#train_dice : 1.774985129225244
#----------------------epoch 20--------------------
#train_loss : 0.056363138902400224
#train_dice : 1.8299529277149587
#'''

# date2=`date +%s`
# ((timer=date2-date1))
# echo "use time: $timer"

