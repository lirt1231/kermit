#!/usr/bin/env bash

python src/evaluate.py \
--checkpoint_path output/FB15k237/checkpoints/model_last.pth \
--batch_size 1024 \
--neighbor_weight 0.05 \
--rerank_n_hop 2 \
--device cuda:0
