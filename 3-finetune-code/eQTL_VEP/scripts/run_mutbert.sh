CUDA_VISIBLE_DEVICES=1 python vep_embedding.py \
  --seq_len 2048 \
  --bp_per_token 1 \
  --model_name_or_path JadenLong/MutBERT \
  --downstream_save_dir output \
  --name mutbert_seqlen=2k \
  --embed_dump_batch_size 128