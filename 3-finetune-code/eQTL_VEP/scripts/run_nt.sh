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


CUDA_VISIBLE_DEVICES=${cuda} python vep_embedding.py \
  --seq_len 12000 \
  --bp_per_token 6 \
  --model_name_or_path ${model} \
  --downstream_save_dir output \
  --name ${run_name}_seqlen=12k \
  --embed_dump_batch_size 20