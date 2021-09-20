#!/usr/bin/env bash
# run at project root dir

exp_id=$1
device_id=$2

CUDA_VISIBLE_DEVICES=${device_id} python train.py \
--dataset_config config/tvr_data_config.json \
--model_config config/model_config.json \
--stop_task VCMR \
--eval_tasks_at_training VCMR SVMR VR \
--use_interal_vr_scores \
--bsz 64 \
--use_extend_pool 500 \
--neg_video_num 0 \
--exp_id ${exp_id} \
--eval_query_bsz 5 \
--max_vcmr_video 10 \
--num_workers 8 \
${@:3}