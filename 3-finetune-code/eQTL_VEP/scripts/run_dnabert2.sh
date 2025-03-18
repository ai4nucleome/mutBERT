#!/bin/bash

cuda=$1

CUDA_VISIBLE_DEVICES=${cuda} python vep_embedding.py \
  --seq_len 2048 \
  --bp_per_token 1 \
  --model_name_or_path zhihan1996/DNABERT-2-117M \
  --downstream_save_dir output \
  --name dnabert2_seqlen=2k \
  --embed_dump_batch_size 32