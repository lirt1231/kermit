import gc

import torch
import wandb

import evaluate
import logger
from dataset import Dataset, init as dataset_init
from model import Kermit
from args import get_args
from trainer import Trainer
from utils import report_model_parameters


if __name__ == "__main__":
    train_args, model_args, eval_args = get_args()
    logger.create_logger(train_args, model_args, eval_args)
    dataset_init(
        train_args.data_dir,
        model_args.pretrained_model,
        [train_args.train_path],
        train_args.train_path,
    )
    train_dataset = Dataset(
        train_args.train_path,
        train_args.max_num_tokens,
        model_args.use_neighbor_names,
        True,
        True,
    )
    valid_dataset = Dataset(
        train_args.valid_path,
        train_args.max_num_tokens,
        model_args.use_neighbor_names,
        True,
        True,
    )
    model = Kermit(model_args)
    trainer = Trainer(train_args, model, train_dataset, valid_dataset)

    num_all, num_trainable = report_model_parameters(model)
    logger.log(
        f"Trainable number of parameters: {(num_trainable / 1e6):.1f}M/{(num_all / 1e6):.1f}M"
    )
    logger.log(repr(model))
    # train model
    trainer.train_loop()

    # testing
    if eval_args.do_test:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        fw_metrics, bw_metrics, avg_metrics = evaluate.main(
            train_args.last_model_path,
            eval_args.neighbor_weight,
            eval_args.rerank_n_hop,
            eval_args.device,
            eval_args.eval_batch_size,
        )
        if train_args.use_wandb:
            for k, v in fw_metrics.items():
                wandb.summary[f"test/forward_{k}"] = v
            for k, v in bw_metrics.items():
                wandb.summary[f"test/backward_{k}"] = v
            for k, v in avg_metrics.items():
                wandb.summary[f"test/avg_{k}"] = v
            wandb.finish()

    logger.close()
