#!/bin/bash

python=python3
dataset=FamilyIC
exp_dir=exps_$dataset
exp_name=$dataset
batch_size=32
max_epoch=20
data_dir=./data/$dataset/
relation_file=$data_dir/relations.txt
gpu_id=0
cat $relation_file | while read line
do
$python -u main.py \
        --data_dir $data_dir \
        --exps_dir ./$exp_dir/ \
        --exp_name $exp_name \
        --target_relation $line \
        --batch_size $batch_size \
        --max_epoch $max_epoch \
        --with_constrain \
        --use_gpu \
        --gpu_id $gpu_id \
        --tau_1 10 \
        --tau_2 0.2 \
        --length 20
$python predict.py ./$exp_dir/$exp_name-$line-ori $batch_size 0 $gpu_id

$python predict.py ./$exp_dir/$exp_name-$line-ori $batch_size 1 $gpu_id
done
$python -u evaluate.py $data_dir $exp_dir $exp_name