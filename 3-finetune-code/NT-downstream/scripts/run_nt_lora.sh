#!/bin/bash

m=$1
cuda=$2


if [ "$m" -eq 0 ]; then
    model=InstaDeepAI/nucleotide-transformer-500m-1000g
    run_name=NT_500_1000g
elif [ "$m" -eq 1 ]; then
    model=InstaDeepAI/nucleotide-transformer-500m-human-ref
    run_name=NT_500_human
elif [ "$m" -eq 2 ]; then
    model=InstaDeepAI/nucleotide-transformer-2.5b-1000g
    run_name=NT_2500_1000g
elif [ "$m" -eq 3 ]; then
    model=InstaDeepAI/nucleotide-transformer-2.5b-multi-species
    run_name=NT_2500_multi
elif [ "$m" -eq 4 ]; then
    model=InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
    run_name=NT_50_multi
elif [ "$m" -eq 5 ]; then
    model=InstaDeepAI/nucleotide-transformer-v2-100m-multi-species
    run_name=NT_100_multi
elif [ "$m" -eq 6 ]; then
    model=InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
    run_name=NT_500_multi
elif [ "$m" -eq 7 ]; then
    model=InstaDeepAI/nucleotide-transformer-v2-250m-multi-species
    run_name=NT_250_multi
else
    echo "Wrong argument"
    exit 1
fi
echo "Use: $model, The provided kmer is: 6"


for seed in 42
do
    for data in H2AFZ H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K9ac H3K9me3 H4K20me1
    do
        CUDA_VISIBLE_DEVICES=${cuda} python train.py \
            --model_name_or_path ${model} \
            --kmer -1 \
            --run_name ${data} \
            --model_max_length 200 \
            --use_lora \
            --lora_r 8 \
            --lora_dropout 0.05 \
            --lora_alpha 16 \
            --lora_target_modules 'query,value,key,dense' \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-4 \
            --max_train_steps 10000 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/${run_name}_lora \
            --eval_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
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
            --model_name_or_path ${model} \
            --kmer -1 \
            --run_name ${data} \
            --model_max_length 70 \
            --num_labels 2 \
            --use_lora \
            --lora_r 8 \
            --lora_dropout 0.05 \
            --lora_alpha 16 \
            --lora_target_modules 'query,value,key,dense' \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-4 \
            --max_train_steps 10000 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/${run_name}_lora \
            --eval_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
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
            --model_name_or_path ${model} \
            --kmer -1 \
            --run_name ${data} \
            --model_max_length 70 \
            --num_labels 3 \
            --use_lora \
            --lora_r 8 \
            --lora_dropout 0.05 \
            --lora_alpha 16 \
            --lora_target_modules 'query,value,key,dense' \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-4 \
            --max_train_steps 10000 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/${run_name}_lora \
            --eval_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
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
            --model_name_or_path ${model} \
            --kmer -1 \
            --run_name ${data} \
            --model_max_length 60 \
            --use_lora \
            --lora_r 8 \
            --lora_dropout 0.05 \
            --lora_alpha 16 \
            --lora_target_modules 'query,value,key,dense' \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-4 \
            --max_train_steps 10000 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/${run_name}_lora \
            --eval_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
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
            --model_name_or_path ${model} \
            --kmer -1 \
            --run_name ${data} \
            --model_max_length 110 \
            --num_labels 2 \
            --use_lora \
            --lora_r 8 \
            --lora_dropout 0.05 \
            --lora_alpha 16 \
            --lora_target_modules 'query,value,key,dense' \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-4 \
            --max_train_steps 10000 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/${run_name}_lora \
            --eval_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
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
            --model_name_or_path ${model} \
            --kmer -1 \
            --run_name ${data} \
            --model_max_length 110 \
            --num_labels 3 \
            --use_lora \
            --lora_r 8 \
            --lora_dropout 0.05 \
            --lora_alpha 16 \
            --lora_target_modules 'query,value,key,dense' \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-4 \
            --max_train_steps 10000 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/${run_name}_lora \
            --eval_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 200 \
            --tb_name ${data} \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done
done