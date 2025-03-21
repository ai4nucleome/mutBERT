#!/bin/bash

cuda=$1


echo "Use: mutbert, The provided kmer is: 1"


for seed in 42
do
    for data in H2AFZ H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K9ac H3K9me3 H4K20me1
    do
        CUDA_VISIBLE_DEVICES=${cuda} python train.py \
            --model_name_or_path JadenLong/MutBERT \
            --kmer 1 \
            --run_name ${data} \
            --model_max_length 1024 \
            --factor 2.0 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-4 \
            --max_train_steps 10000 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/mutbert \
            --eval_strategy steps \
            --eval_steps 200 \
            --warmup_steps 100 \
            --logging_steps 200 \
            --tb_name ${data} \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done

    for data in enhancers
    do
        CUDA_VISIBLE_DEVICES=${cuda} python train.py \
            --model_name_or_path JadenLong/MutBERT \
            --kmer 1 \
            --run_name ${data} \
            --model_max_length 402 \
            --factor 1.0 \
            --num_labels 2 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-4 \
            --max_train_steps 10000 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/mutbert \
            --eval_strategy steps \
            --eval_steps 200 \
            --warmup_steps 100 \
            --logging_steps 200 \
            --tb_name ${data} \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done

    for data in enhancers_types
    do
        CUDA_VISIBLE_DEVICES=${cuda} python train.py \
            --model_name_or_path JadenLong/MutBERT \
            --kmer 1 \
            --run_name ${data} \
            --model_max_length 402 \
            --factor 1.0 \
            --num_labels 3 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-4 \
            --max_train_steps 10000 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/mutbert \
            --eval_strategy steps \
            --eval_steps 200 \
            --warmup_steps 100 \
            --logging_steps 200 \
            --tb_name ${data} \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done

    for data in promoter_all promoter_no_tata promoter_tata
    do
        CUDA_VISIBLE_DEVICES=${cuda} python train.py \
            --model_name_or_path JadenLong/MutBERT \
            --kmer 1 \
            --run_name ${data} \
            --model_max_length 302 \
            --factor 1.0 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-4 \
            --max_train_steps 10000 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/mutbert \
            --eval_strategy steps \
            --eval_steps 200 \
            --warmup_steps 100 \
            --logging_steps 200 \
            --tb_name ${data} \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done

    for data in splice_sites_acceptors splice_sites_donors
    do
        CUDA_VISIBLE_DEVICES=${cuda} python train.py \
            --model_name_or_path JadenLong/MutBERT \
            --kmer 1 \
            --run_name ${data} \
            --model_max_length 768 \
            --factor 1.5 \
            --num_labels 2 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-4 \
            --max_train_steps 10000 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/mutbert \
            --eval_strategy steps \
            --eval_steps 200 \
            --warmup_steps 100 \
            --logging_steps 200 \
            --tb_name ${data} \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done

    for data in splice_sites_all
    do
        CUDA_VISIBLE_DEVICES=${cuda} python train.py \
            --model_name_or_path JadenLong/MutBERT \
            --kmer 1 \
            --run_name ${data} \
            --model_max_length 768 \
            --factor 1.5 \
            --num_labels 3 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-4 \
            --max_train_steps 10000 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/mutbert \
            --eval_strategy steps \
            --eval_steps 200 \
            --warmup_steps 100 \
            --logging_steps 200 \
            --tb_name ${data} \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done
done