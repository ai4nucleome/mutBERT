#!/bin/bash

cuda=$1
lr=3e-5

echo "Use mutbert."

for seed in 42
do

    for data in 0 1 2 3 4
    do 
        CUDA_VISIBLE_DEVICES=${cuda} python train.py \
            --model_name_or_path JadenLong/MutBERT \
            --data_path  data/GUE/tf/${data} \
            --kmer 1 \
            --run_name tf_${data} \
            --model_max_length 102 \
            --factor 1.0 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 3 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/mutbert \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --tb_name our_tf_${data} \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done
done
