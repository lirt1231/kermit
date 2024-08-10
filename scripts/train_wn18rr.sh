#!/usr/bin/env bash

TOKENIZERS_PARALLELISM=true python src/main.py \
--dataset WN18RR \
--pretrained_model google-bert/bert-base-uncased \
--tau 0.03 \
--max_num_tokens 150 \
--additive_margin 0.02 \
--epochs 65 \
--batch_size 1024 \
--lr 8e-5 \
--use_neighbor_names \
--pooling mean \
--workers 4 \
--experiment_name wn18rr \
--do_test \
--neighbor_weight 0.05 \
--rerank_n_hop 5 \
--eval_batch_size 1024

# --project ${wandb_project} \
# --entity ${wandb_entity}
