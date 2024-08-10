import logging
import sys
from collections import ChainMap
from typing import Any, Callable, Dict, List, Tuple

import wandb

from args import TrainerArguments, ModelArguments, EvaluationArguments


__all__ = ("MetricTracker", "BestMetricTracker", "create_logger", "log", "log_metric")


class MetricTracker:
    name: str
    prefix: str
    value: Any

    def __init__(self, name: str, prefix: str = None) -> None:
        self.prefix = prefix
        self.name = name

    def update(self, val: Any) -> None:
        self.value = val

    def format_dict(self) -> Dict[str, Any]:
        if self.prefix:
            return {f"{self.prefix}/{self.name}": self.value}
        else:
            return {f"{self.name}": self.value}


class AvgMetricTracker(MetricTracker):
    def __init__(self, name: str, prefix: str = None) -> None:
        super().__init__(name, prefix)
        self.value = 0
        self.sum = 0
        self.count = 0

    def update(self, val_avg: Any, num: int) -> None:
        self.sum += val_avg * num
        self.count += num
        self.value = self.sum / self.count


class BestMetricTracker(MetricTracker):
    metric: "BestMetric"
    best_fn: Callable[[Any, Any], bool]

    class BestMetric:
        def __init__(self, value, step) -> None:
            self.value = value
            self.step = step

        def __eq__(self, other: "BestMetricTracker.BestMetric") -> bool:
            return self.value == other.value

        def __lt__(self, other: "BestMetricTracker.BestMetric") -> bool:
            return BestMetricTracker.best_fn(other.value, self.value)

        def get(self, round_digits: int = 4) -> Tuple[int, int]:
            return int(self.value * (10**round_digits)), self.step

    def __init__(
        self,
        name: str,
        prefix: str = None,
        value: float = 0.0,
        best_fn: Callable[[Any, Any], bool] = lambda x, y: x > y,
    ) -> None:
        super().__init__(name, prefix)
        BestMetricTracker.best_fn = best_fn
        self.step = 0
        self.metric = BestMetricTracker.BestMetric(value, 0)

    def update(self, val: float, step: int) -> None:
        if BestMetricTracker.best_fn(val, self.value):
            self.metric = BestMetricTracker.BestMetric(val, step)

    @property
    def value(self) -> float:
        return self.metric.value


class Logger:
    def log(self, msg: str) -> None:
        pass

    def log_metric(self, *metrics: Tuple[MetricTracker], **kwargs) -> None:
        metric_dict = dict(ChainMap(*[metric.format_dict() for metric in metrics]))
        self._log_metric(metric_dict, **kwargs)

    def _log_metric(self, metric_dict: Dict[str, Any], **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


class WandbLogger(Logger):
    def __init__(self) -> None:
        pass

    def _log_metric(self, metric_dict: Dict[str, Any], **kwargs) -> None:
        wandb.log(metric_dict, **kwargs)

    def close(self) -> None:
        wandb.finish()


class SysLogger(Logger):
    def __init__(self, log_path: str) -> None:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s | %(levelname)s |  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(filename=log_path, mode="w"),
                logging.StreamHandler(stream=sys.stdout),
            ],
        )

        self.logger = logging.getLogger(__name__)

    def log(self, msg: str) -> None:
        self.logger.info(msg)

    def _log_metric(self, metric_dict: Dict[str, Any], **kwargs) -> None:
        self.logger.info(metric_dict)


loggers: List[Logger] = []


def create_logger(
    train_args: TrainerArguments, model_args: ModelArguments, eval_args: EvaluationArguments
):
    global loggers
    loggers.append(SysLogger(train_args.log_file_path))
    if train_args.use_wandb:
        wandb.init(
            dir=train_args.log_dir,
            config={"train_args": train_args, "model_args": model_args, "eval_args": eval_args},
            project=train_args.project,
            entity=train_args.entity,
            name=train_args.experiment_name,
        )
        loggers.append(WandbLogger())


def close():
    for logger in loggers:
        logger.close()


def log(msg: str):
    for logger in loggers:
        logger.log(msg)


def log_metric(*metrics: Tuple[MetricTracker], **kwargs):
    for logger in loggers:
        logger.log_metric(*metrics, **kwargs)
