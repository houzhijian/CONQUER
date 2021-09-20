#!/usr/bin/env bash
# run at project root dir


model_dir=$1
tasks=(VR)
tasks+=(SVMR)
tasks+=(VCMR)
device_id=$2

CUDA_VISIBLE_DEVICES=${device_id} python inference.py \
--dataset_config config/tvr_data_config.json \
--model_config config/model_config.json \
--max_vcmr_video 10 \
--eval_query_bsz 5 \
--model_dir ${model_dir} \
--tasks ${tasks[@]} \
${@:3}