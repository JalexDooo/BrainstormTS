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

# Aneu_traing # 89 # 80
#cd /home/aneu/Brainstorm && python main.py Aneu_train --task='Random' --model='ModuleTest6' --max_epoch=11 --batch_size=4 --lr=0.0001 --load_model_path='task_Random__epoch_14.pth'
#cd /home/aneu/Brainstorm && python main.py Aneu_train --task='Random' --model='ModuleTest6' --max_epoch=11 --batch_size=8 --lr=0.0001 --load_model_path='task_Random__final_epoch.pth' --aneu_path='/home/aneu/dataset/dongmai_dark'
#cd /home/aneu/Brainstorm && python main.py Aneu_train --task='Random' --model='UNet3D' --max_epoch=51 --batch_size=12 --lr=0.0001 # --load_model_path='task_Random__final_epoch.pth' --aneu_path='/home/aneu/dataset/dongmai_dark'
#cd /home/aneu/Brainstorm && python main.py Aneu_train --task='Random' --model='UNet3D_Aneu' --max_epoch=51 --batch_size=12 --lr=0.0001 # --load_model_path='task_Random__final_epoch.pth' --aneu_path='/home/aneu/dataset/dongmai_dark'
#cd /home/aneu/Brainstorm && python main.py Aneu_train --task='Random' --model='MultiscaleUNet3D' --max_epoch=51 --batch_size=8 --lr=0.0001 # --load_model_path='task_Random__final_epoch.pth' --aneu_path='/home/aneu/dataset/dongmai_dark'

#cd /home/aneu/Brainstorm && python main.py Aneu_predict --task='Random' --model='ModuleTest6' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__final_epoch.pth'
#cd /home/aneu/Brainstorm && python main.py Aneu_predict --task='Random' --model='UNet3D' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__final_epoch.pth'
#cd /home/aneu/Brainstorm && python main.py Aneu_predict --task='Random' --model='UNet3D_Aneu' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__final_epoch.pth'
#cd /home/aneu/Brainstorm && python main.py Aneu_predict --task='Random' --model='MultiscaleUNet3D' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__final_epoch.pth'
#cd /home/aneu/Brainstorm && python main.py Aneu_predict --task='Random' --model='MultiscaleUNet3D' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__epoch_45.pth'

#cd /home/aneu/Brainstorm && python main.py Aneu_test --task='Random' --model='ModuleTest6' --max_epoch=1 --batch_size=1 --aneu_path='/home/aneu/dataset/final_val_dataset' --load_model_path='task_Random__final_epoch.pth'
#cd /home/aneu/Brainstorm && python main.py Aneu_test --task='Random' --model='UNet3D' --max_epoch=1 --batch_size=1 --aneu_path='/home/aneu/dataset/final_val_dataset' --load_model_path='task_Random__final_epoch.pth'
#cd /home/aneu/Brainstorm && python main.py Aneu_test --task='Random' --model='UNet3D_Aneu' --max_epoch=1 --batch_size=1 --aneu_path='/home/aneu/dataset/final_val_dataset' --load_model_path='task_Random__final_epoch.pth'
#cd /home/aneu/Brainstorm && python main.py Aneu_test --task='Random' --model='MultiscaleUNet3D' --max_epoch=1 --batch_size=1 --aneu_path='/home/aneu/dataset/final_val_dataset' --load_model_path='task_Random__final_epoch.pth'
#cd /home/aneu/Brainstorm && python main.py Aneu_test --task='Random' --model='MultiscaleUNet3D' --max_epoch=1 --batch_size=1 --aneu_path='/home/aneu/dataset/final_val_dataset' --load_model_path='task_Random__epoch_45.pth'

# trainset 3169 test 3170 val 3184

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

# --------------------------------------- The Second Paper --------------------------------------------- epoch 160
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest_multi_train_random --task='Random' --model='NewResUnet1' --random_epoch=80 --max_epoch=1 --batch_size=16 --lr=0.001  --load_model_path='task_Random__final_epoch.pth'
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest_multi_val_random --task='Random' --model='NewResUnet1' --random_epoch=1 --max_epoch=1 --batch_size=1 --lr=0.001  --load_model_path='task_Random__final_epoch.pth' --predict_nibable_path='./brats2019_val_NewResUnet1_random/'

# --------------------------------------- My brain aneu ---------------------------------------------
#cd /home/aneu/Brainstorm && python main.py Aneu_train --task='Random' --model='NewResUnet1' --max_epoch=21 --batch_size=4 --lr=0.001 #--load_model_path='task_Random__epoch_14.pth'
#cd /home/aneu/Brainstorm && python main.py dataset_clean --task='Random' --model='NewResUnet1' --max_epoch=21 --batch_size=4 --lr=0.001 #--load_model_path='task_Random__epoch_14.pth'

# --------------------------------------- Neurocomputing fix ---------------------------------------------
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest_multi_train_random --task='Random' --model='ModuleTest5' --random_epoch=100 --max_epoch=1 --batch_size=8
cd /home/sunjindong/BrainstormTS && python main.py ModuleTest_multi_val_random --task='Random' --model='ModuleTest5' --max_epoch=1 --batch_size=1  --load_model_path='task_Random__final_epoch.pth' --predict_nibable_path='./brats2019_val_moduletest5_random/'
#cd /home/sunjindong/BrainstormTS && python main.py ModuleTest_multi_train_random --task='Random' --model='ModuleTest4' --random_epoch=100 --max_epoch=1 --batch_size=4

# date2=`date +%s`
# ((timer=date2-date1))
# echo "use time: $timer"

