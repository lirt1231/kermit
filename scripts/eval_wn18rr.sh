#!/usr/bin/env bash

python src/evaluate.py \
--checkpoint_path output/WN18RR/checkpoints/model_last.pth \
--batch_size 1024 \
--neighbor_weight 0.05 \
--rerank_n_hop 5 \
--device cuda:0
