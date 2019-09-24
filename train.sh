#!/bin/bash

date1=`date +%s`

# cd /home/sunjindong/BrainstormTS && python main.py train --load_model_path='epoch_0.pth' --max_epoch=1
cd /home/sunjindong/BrainstormTS && python main.py test --load_model_path='epoch_0.pth' --max_epoch=1

date2=`date +%s`
((timer=date2-date1))
echo "use time: $timer"

