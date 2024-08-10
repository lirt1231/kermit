import os

import torch
from torch.cuda import amp
from torch.utils.data import DataLoader

import dataset
import logger
from args import TrainerArguments
from logger import AvgMetricTracker, MetricTracker, BestMetricTracker
from model import Kermit, ModelOutput
from utils import compute_hits, move_to_cuda, get_model_obj, get_linear_schedule_with_warmup


class Trainer:
    def __init__(
        self,
        train_args: TrainerArguments,
        model: Kermit,
        train_dataset: dataset.Dataset,
        valid_dataset: dataset.Dataset,
    ) -> None:
        self.args = train_args
        self.global_step = 0
        self.epoch_start = 0

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(model).cuda()
        elif torch.cuda.is_available():
            self.model = model.cuda()
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_args.batch_size,
            shuffle=True,
            collate_fn=dataset.collate,
            pin_memory=True,
            num_workers=train_args.workers,
        )
        self.valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=train_args.batch_size * 2,
            shuffle=False,
            collate_fn=dataset.collate,
            pin_memory=True,
            num_workers=train_args.workers,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=train_args.lr, weight_decay=train_args.weight_decay
        )
        self.num_steps_per_epoch = len(self.train_dataloader)
        self.num_training_steps = train_args.epochs * self.num_steps_per_epoch
        warmup = min(train_args.warmup, self.num_training_steps // 10)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup, self.num_training_steps
        )
        if train_args.use_amp:
            self.scaler = amp.GradScaler()

    def train_loop(self) -> None:
        self.best_metric_tracker = BestMetricTracker("best_Hit@1", "valid")
        for epoch in range(self.epoch_start, self.args.epochs + self.epoch_start):
            # train for one epoch
            self.train_epoch()
            self.eval()
            if (epoch + 1) % self.args.save_freq == 0:
                save_path = os.path.join(self.args.model_save_dir, f"model_epoch{epoch + 1}.pth")
                model = get_model_obj(self.model)
                torch.save(model.state_dict(), save_path)
        # save the last model
        model = get_model_obj(self.model)
        torch.save(model.state_dict(), self.args.last_model_path)

    def train_epoch(self) -> None:
        loss_tracker = AvgMetricTracker("loss", "train")
        lr_tracker = MetricTracker("lr", "train")
        invt_tracker = MetricTracker("inv_t", "train")

        for _, inputs in enumerate(self.train_dataloader):
            self.global_step += 1
            model = self.model.train()
            inputs.pop("triples")
            inputs = move_to_cuda(inputs)
            if self.args.use_amp:
                with amp.autocast():
                    model_output = ModelOutput(**model(**inputs))
            else:
                model_output = ModelOutput(**model(**inputs))
            loss_dict = get_model_obj(model).compute_loss(
                model_output.hr_vector,
                model_output.tail_vector,
                inputs["positive_mask"],
            )
            loss = loss_dict["loss"]
            self.backward(loss)

            batch_size = model_output.hr_vector.size(0)
            loss_tracker.update(loss.item(), batch_size)
            lr_tracker.update(self.scheduler.get_last_lr()[0])
            invt_tracker.update(loss_dict["inv_t"].item())

            if self.global_step % self.args.log_step == 0:
                epoch_prop = self.global_step / self.num_steps_per_epoch
                logger.log(
                    f"Step: {self.global_step}/{self.num_training_steps}, "
                    + f"Epoch: {epoch_prop:.2f}/{self.args.epochs}"
                )
                logger.log_metric(loss_tracker, lr_tracker, invt_tracker, step=self.global_step)

    @torch.inference_mode()
    def eval(self) -> None:
        model = self.model.eval()
        dataset.set_eval_mode(True)
        loss_tracker = AvgMetricTracker("loss_recon", "valid")
        hit1_tracker = AvgMetricTracker("Hit@1", "valid")
        hit3_tracker = AvgMetricTracker("Hit@3", "valid")
        hit10_tracker = AvgMetricTracker("Hit@10", "valid")

        for _, inputs in enumerate(self.valid_dataloader):
            inputs.pop("triples")
            inputs = move_to_cuda(inputs)
            model_output = ModelOutput(**model(**inputs))
            loss_dict = get_model_obj(model).compute_loss(
                model_output.hr_vector,
                model_output.tail_vector,
                inputs["positive_mask"],
            )
            logits, loss = loss_dict["logits"], loss_dict["loss"]
            hit1, hit3, hit10 = compute_hits(logits, inputs["labels"], topk=(1, 3, 10))

            batch_size = logits.size(0)
            loss_tracker.update(loss.item(), batch_size)
            hit1_tracker.update(hit1, batch_size)
            hit3_tracker.update(hit3, batch_size)
            hit10_tracker.update(hit10, batch_size)
        self.best_metric_tracker.update(hit1_tracker.value, self.global_step)

        logger.log_metric(
            loss_tracker,
            hit1_tracker,
            hit3_tracker,
            hit10_tracker,
            self.best_metric_tracker,
            step=self.global_step,
        )
        dataset.set_eval_mode(False)

    def backward(self, loss: torch.FloatTensor) -> None:
        self.optimizer.zero_grad()
        if self.args.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
        self.scheduler.step()
