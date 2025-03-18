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
    for data in 0 1 2 3 4
    do 
        CUDA_VISIBLE_DEVICES=${cuda} python train.py \
            --model_name_or_path ${model} \
            --data_path  data/GUE/tf/$data \
            --kmer -1 \
            --run_name tf_${data} \
            --model_max_length 30 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 3 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/${run_name} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --tb_name tf_${data} \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done
done