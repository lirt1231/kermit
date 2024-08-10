#!/usr/bin/env bash

TOKENIZERS_PARALLELISM=true python src/main.py \
--dataset FB15k237 \
--pretrained_model google-bert/bert-base-uncased \
--max_num_tokens 150 \
--additive_margin 0.02 \
--epochs 10 \
--batch_size 1024 \
--tau 0.04 \
--lr 1e-5 \
--workers 4 \
--use_neighbor_names \
--pooling mean \
--experiment_name FB15k237 \
--do_test \
--neighbor_weight 0.05 \
--rerank_n_hop 2 \
--eval_batch_size 1024

# --project ${wandb_project} \
# --entity ${wandb_entity}
