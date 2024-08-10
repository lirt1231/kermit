import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
from simple_parsing import ArgumentParser


__all__ = ("TrainerArguments", "ModelArguments", "get_args")


@dataclass
class TrainerArguments:
    # dataset arguments
    dataset: str = "WN18RR"  # ["WN18RR", "FB15k237"]
    max_num_tokens: int = 128  # max length of text sequence
    # trainer arguments
    epochs: int = 50  # number of max epochs
    batch_size: int = 128  # batch size
    save_freq: int = 5  # model saving frequency per training epochs
    log_step: int = 10  # logging frequency per training steps
    seed: int = 24213  # random state
    lr: float = 2e-5  # initial learning rate for AdamW
    weight_decay: float = 1e-4  # weight decay for AdamW
    use_amp: bool = True  # use torch.amp
    workers: int = 1  # number of data loading workers
    grad_clip: float = 10.0  # gradient clipping
    warmup: int = 400  # warmup steps

    output_path: str = "output/"
    # wandb config
    experiment_name: Optional[str] = None  # wandb run name
    project: Optional[str] = None  # wandb project name
    entity: Optional[str] = None  # wandb entity

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = self.dataset
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        # create directories
        self.data_dir = os.path.join("./data", self.dataset)
        self.train_path = os.path.join(self.data_dir, "train.json")
        self.valid_path = os.path.join(self.data_dir, "valid.json")
        self.test_path = os.path.join(self.data_dir, "test.json")

        self.log_dir = os.path.join(self.output_path, "logs")
        self.log_file_path = os.path.join(self.log_dir, f"{self.experiment_name}.log")
        self.experiment_path = os.path.join(self.output_path, self.experiment_name)
        self.model_save_dir = os.path.join(self.experiment_path, "checkpoints")
        self.last_model_path = os.path.join(self.model_save_dir, "model_last.pth")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir, exist_ok=True)

        self.use_wandb = self.project is not None


@dataclass
class ModelArguments:
    pretrained_model: str = "bert-base-uncased"
    use_neighbor_names: bool = False
    tau: float = 0.05  # temperature parameter
    finetune_tau: bool = False  # make temperature as a trainable parameter or not
    additive_margin: float = 0.0  # additive margin for InfoNCE loss function
    pooling: str = "mean"  # pooling method: ["mean", "cls"]


@dataclass
class EvaluationArguments:
    do_test: bool = False
    neighbor_weight: float = 0.0  # weight for neighbor loss
    rerank_n_hop: int = 0  # number of hops for reranking
    device = "cuda"  # device for evaluation
    eval_batch_size: int = 128  # batch size for evaluation


def get_args():
    parser = ArgumentParser()
    parser.add_arguments(TrainerArguments, dest="train_args")
    parser.add_arguments(ModelArguments, dest="model_args")
    parser.add_arguments(EvaluationArguments, dest="eval_args")
    args = parser.parse_args()
    train_args: TrainerArguments = args.train_args
    model_args: ModelArguments = args.model_args
    eval_args: EvaluationArguments = args.eval_args

    with open(os.path.join(train_args.experiment_path, "args.json"), "w") as file:
        json.dump(
            {
                "train_args": asdict(train_args),
                "model_args": asdict(model_args),
                "eval_args": asdict(eval_args),
            },
            file,
            indent=4,
        )

    return train_args, model_args, eval_args
