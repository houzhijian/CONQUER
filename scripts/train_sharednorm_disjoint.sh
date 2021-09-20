#!/usr/bin/env bash
# run at project root dir

exp_id=$1
device_id=$2

CUDA_VISIBLE_DEVICES=${device_id} python train.py \
--dataset_config config/tvr_data_config.json \
--model_config config/model_config.json \
--stop_task VCMR \
--eval_tasks_at_training VCMR SVMR VR \
--bsz 16 \
--use_extend_pool 1000 \
--neg_video_num 3 \
--exp_id ${exp_id} \
--eval_query_bsz 5 \
--max_vcmr_video 10 \
--num_workers 8 \
--similarity_measure disjoint \
${@:3}