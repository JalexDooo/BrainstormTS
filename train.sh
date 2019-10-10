#!/bin/bash

# date1=`date +%s`

# cd /home/sunjindong/BrainstormTS && python main.py unzip
#cd /home/sunjindong/BrainstormTS && python main.py train --max_epoch=30
# cd /home/sunjindong/BrainstormTS && python main.py test --load_model_path='epoch_0.pth' --max_epoch=1
# cd /home/sunjindong/BrainstormTS && python main.py datatest
# cd /home/sunjindong/BrainstormTS && python main.py detection_and_train --load_model_path='detect_train_0.pth' --max_epoch=60
# cd /home/sunjindong/BrainstormTS && python main.py detection_and_train --max_epoch=1

cd /home/sunjindong/BrainstormTS && python main.py brats2019_train --model='MultiscaleUNet3D' --batch_size=2 --max_epoch=5 --load_model_path='epoch_5.pth'
# date2=`date +%s`
# ((timer=date2-date1))
# echo "use time: $timer"

