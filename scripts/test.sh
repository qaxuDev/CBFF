#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# # use torch.distributed.launch
# switch to current folder
# sh scripts/train.sh <num_gpu> <port>

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['levir', 'whu']
# method: ['fixmatch_CbffDecoder', 'supervised'] 
# exp: just for specifying the 'save_path'
# split: ['5%', '10%', '20%', '40%']

dataset='whu'
method='fixmatch_CbffDecoder'  
exp='deeplabv3plus_r50'
split='5%'

config=configs/whu.yaml

labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt

save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    inference.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log



