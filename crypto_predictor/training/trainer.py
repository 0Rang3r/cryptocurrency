import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional

from config.config import TrainConfig
from training.early_stopping import EarlyStopping
from training.metrics import compute_classification_metrics, log_metrics
from utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        cfg: TrainConfig = None,
        pos_weight_value: Optional[float] = None,
    ):
        self.model = model.to(device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.cfg = cfg or TrainConfig()

        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(self.cfg.results_dir, exist_ok=True)

        pos_weight = self._estimate_pos_weight(pos_weight_value)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=device) if pos_weight is not None else None
        )

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        self.scheduler = self._build_scheduler()

        ckpt_path = os.path.join(self.cfg.checkpoint_dir, f"best_{model_name}.pth")
        self.early_stopping = EarlyStopping(
            patience=self.cfg.early_stopping_patience,
            metric=self.cfg.early_stopping_metric,
            checkpoint_path=ckpt_path,
        )

    def run(self) -> Dict:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"开始训练: {self.model_name}")
        logger.info(f"{'=' * 60}")

        self._train_loop()
        return self._test()

    def _estimate_pos_weight(self, override: Optional[float]) -> Optional[float]:
        if override is not None:
            logger.info(f"使用手动指定的 pos_weight: {override:.3f}")
            return override

        all_labels = []
        for _, _, y in self.train_loader:
            all_labels.extend(y.numpy())

        labels = np.array(all_labels)
        pos = labels.sum()
        neg = len(labels) - pos

        if pos == 0 or neg == 0:
            logger.warning("训练集中只有单一类别，跳过 pos_weight")
            return None

        pos_weight = neg / pos
        logger.info(f"自动估算 pos_weight: {pos_weight:.3f} (正样本: {int(pos)}, 负样本: {int(neg)})")
        return float(pos_weight)

    def _build_scheduler(self):
        if not self.cfg.use_lr_scheduler:
            return None

        if self.cfg.lr_scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.epochs,
                eta_min=self.cfg.lr_min,
            )

        if self.cfg.lr_scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                verbose=False,
            )

        logger.warning(f"未知调度器类型: {self.cfg.lr_scheduler_type}，跳过")
        return None

    def _train_one_epoch(self) -> float:
        self.model.train()
        losses = []

        for x_price, x_emotion, y in self.train_loader:
            x_price = x_price.to(self.device)
            x_emotion = x_emotion.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x_price, x_emotion)
            loss = self.criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.cfg.grad_clip_norm,
            )

            self.optimizer.step()
            losses.append(loss.item())

        return float(np.mean(losses))

    def _validate(self) -> Dict:
        self.model.eval()
        val_losses = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for x_price, x_emotion, y in self.val_loader:
                x_price = x_price.to(self.device)
                x_emotion = x_emotion.to(self.device)
                y_dev = y.to(self.device)

                logits = self.model(x_price, x_emotion)
                loss = self.criterion(logits, y_dev)
                val_losses.append(loss.item())

                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y.numpy())

        metrics = compute_classification_metrics(
            np.array(all_labels),
            np.array(all_probs),
        )
        metrics["val_loss"] = round(float(np.mean(val_losses)), 4)
        return metrics

    def _train_loop(self) -> None:
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._train_one_epoch()
            val_metrics = self._validate()

            val_loss = val_metrics["val_loss"]
            val_f1 = val_metrics["F1"]

            if self.scheduler is not None:
                if self.cfg.lr_scheduler_type == "plateau":
                    self.scheduler.step(val_f1)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch [{epoch:3d}/{self.cfg.epochs}] "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
                f"| val_F1={val_f1:.4f} | lr={current_lr:.2e}"
            )

            monitor_value = val_f1 if self.cfg.early_stopping_metric == "f1" else val_loss
            self.early_stopping(monitor_value, self.model)

            if self.early_stopping.early_stop:
                break

        self.early_stopping.load_best(self.model)

    def _test(self) -> Dict:
        logger.info(f"\n[{self.model_name}] 测试集评估")
        self.model.eval()

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for x_price, x_emotion, y in self.test_loader:
                x_price = x_price.to(self.device)
                x_emotion = x_emotion.to(self.device)

                logits = self.model(x_price, x_emotion)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y.numpy())

        metrics = compute_classification_metrics(
            np.array(all_labels),
            np.array(all_probs),
        )
        metrics["model"] = self.model_name
        log_metrics(metrics, prefix=self.model_name)
        return metrics