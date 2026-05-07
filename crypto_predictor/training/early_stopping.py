import copy
import torch
import torch.nn as nn

from utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        metric: str = "f1",
        delta: float = 1e-4,
        checkpoint_path: str = "best_model.pth",
        verbose: bool = True,
    ):
        assert metric in ("f1", "loss"), "metric 只支持 'f1' 或 'loss'"

        self.patience = patience
        self.metric = metric
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def _to_score(self, value: float) -> float:
        return -value if self.metric == "loss" else value

    def __call__(self, metric_value: float, model: nn.Module) -> None:
        score = self._to_score(metric_value)

        if self.best_score is None:
            self.best_score = score
            self._save(metric_value, model)
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping [{self.metric}]: {self.counter}/{self.patience} 次未提升 "
                    f"(当前={metric_value:.4f}, 最优={self._to_score(self.best_score):.4f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("触发早停，停止训练")
            return

        self.best_score = score
        self._save(metric_value, model)
        self.counter = 0

    def _save(self, metric_value: float, model: nn.Module) -> None:
        if self.verbose:
            prev = self._to_score(self.best_score) if self.best_score is not None else float("nan")
            logger.info(f"[{self.metric}] 提升: {prev:.4f} -> {metric_value:.4f}，保存模型")

        torch.save(model.state_dict(), self.checkpoint_path)
        self.best_model_state = copy.deepcopy(model.state_dict())

    def load_best(self, model: nn.Module) -> nn.Module:
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            logger.info("已恢复最优模型权重")
        return model