#!/bin/bash

python=python3
dataset=YAGO26K906
exp_dir=exps_$dataset
exp_name=$dataset
batch_size=16
max_epoch=50
data_dir=./data/$dataset/
relation_file=$data_dir/relations.dict
gpu_id=0
cat $relation_file | awk -F "\t" '{print $2}' | while read line
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
	      --learning_rate 0.1 \
        --length 50 	
$python predict.py ./$exp_dir/$exp_name-${line//\//\|}-ori $batch_size 0 $gpu_id

$python predict.py ./$exp_dir/$exp_name-${line//\//\|}-ori $batch_size 1 $gpu_id
done
$python -u evaluate_no_valid.py $data_dir $exp_dir $exp_name
