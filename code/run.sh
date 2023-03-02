
nohup python train.py --gpu 0 --save Replaced_with_depthconv_01 --pre_train False 2>&1 &

nohup python train.py --gpu 1 --save Replaced_with_depthconv_02 --pre_train False 2>&1 &

nohup python train.py --gpu 2 --save Replaced_with_depthconv_03 --pre_train False 2>&1 &