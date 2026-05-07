import os
import sys
import copy
import random
import warnings
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CRYPTO_ROOT = os.path.join(PROJECT_ROOT, "crypto_predictor")
if CRYPTO_ROOT not in sys.path:
    sys.path.insert(0, CRYPTO_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    balanced_accuracy_score,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from dataset_loader import CryptoFixedWindowDataset
from dataset_adaptive import AdaptiveWindowDataset
from models.baselines.lstm_gru_transformer import BaselineLSTM, VanillaTransformer
from models.innovative.dual_attention_model import InnovativeDualAttentionModel
from models.innovative.dual_attention_model_m5_only import InnovativeDualAttentionModelM5Only

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CACHE_DIR = "./dataset_cache"
FIXED_CACHE_NAME = "fixed_dataset.pt"
ADAPTIVE_CACHE_NAME_M4 = "adaptive_dataset_v2.pt"
ADAPTIVE_CACHE_NAME_M5 = "adaptive_dataset_m5plus.pt"

BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 12
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
SEED = 42
NUM_WORKERS = 0

EARLY_STOP_MIN_DELTA = 1e-5
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 4
SCHEDULER_MIN_LR = 1e-6

M5_STAGE1_EPOCHS = 8
M5_STAGE2_EPOCHS = 40
M5_STAGE1_LR = 1e-4
M5_STAGE2_LR = 5e-5
M5_WEIGHT_DECAY = 5e-4
M5_RESIDUAL_ALPHA = 0.08
M5_UNFREEZE_LAST_N = 1
M5_STAGE1_DISTILL_WEIGHT = 0.20
M5_STAGE2_DISTILL_WEIGHT = 0.05


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStoppingLoss:
    def __init__(self, patience=12, min_delta=1e-5, path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model, extra_state=None):
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            payload = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "best_val_loss": val_loss,
            }
            if extra_state is not None:
                payload.update(extra_state)
            torch.save(payload, self.path)
            return

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True


def to_1d_numpy(x):
    return np.asarray(x).reshape(-1)


def safe_auc(y_true, y_prob):
    y_true = to_1d_numpy(y_true).astype(int)
    y_prob = to_1d_numpy(y_prob).astype(float)
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return 0.5


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = to_1d_numpy(y_true).astype(int)
    y_prob = to_1d_numpy(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except ValueError:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "AUC": safe_auc(y_true, y_prob),
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred),
        "preds_class": y_pred,
        "fpr": fpr,
        "tpr": tpr,
    }


def print_cache_meta(cache_name, cache_obj):
    print(f"\n[{cache_name}]")
    for key in [
        "target_col",
        "task_horizon",
        "seq_len",
        "p_dim",
        "e_dim",
        "cache_version",
        "emotion_lag",
    ]:
        if key in cache_obj:
            print(f"  {key}: {cache_obj[key]}")
    if "train" in cache_obj:
        print(f"  train size: {len(cache_obj['train'])}")
    if "val" in cache_obj:
        print(f"  val size:   {len(cache_obj['val'])}")
    if "test" in cache_obj:
        print(f"  test size:  {len(cache_obj['test'])}")


def choose_guarded_threshold(y_true, y_prob):
    y_true = to_1d_numpy(y_true).astype(int)
    y_prob = to_1d_numpy(y_prob).astype(float)

    base_t = 0.50
    base_metrics = compute_metrics(y_true, y_prob, threshold=base_t)
    best_t = base_t
    best_metrics = base_metrics

    for t in np.arange(0.45, 0.551, 0.01):
        metrics = compute_metrics(y_true, y_prob, threshold=float(t))
        pos_rate = metrics["preds_class"].mean() if len(metrics["preds_class"]) > 0 else 0.0

        if pos_rate < 0.35 or pos_rate > 0.65:
            continue
        if metrics["F1-Score"] + 1e-9 < base_metrics["F1-Score"]:
            continue
        if metrics["BalancedAcc"] > best_metrics["BalancedAcc"] + 1e-9:
            best_t = float(t)
            best_metrics = metrics

    return best_t, best_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []

    for x_price, x_emotion, y in loader:
        x_price = x_price.to(device).float()
        x_emotion = x_emotion.to(device).float()
        y = y.to(device).float().view(-1)

        optimizer.zero_grad()
        outputs = model(x_price, x_emotion).view(-1)
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for x_price, x_emotion, y in loader:
            x_price = x_price.to(device).float()
            x_emotion = x_emotion.to(device).float()
            y = y.to(device).float().view(-1)

            outputs = model(x_price, x_emotion).view(-1)
            loss = criterion(outputs, y)
            losses.append(loss.item())

    return float(np.mean(losses)) if losses else np.inf


def extract_teacher_emotion_from_m5plus(x_emotion_m5: torch.Tensor) -> torch.Tensor:
    keep_idx = []
    for i in range(5):
        base = i * 4
        keep_idx.extend([base, base + 1])
    return x_emotion_m5[:, :, keep_idx]


def predict_probabilities(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x_price, x_emotion, y in loader:
            x_price = x_price.to(device).float()
            x_emotion = x_emotion.to(device).float()
            y = y.to(device).float().view(-1)

            outputs = model(x_price, x_emotion).view(-1)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            labels = y.detach().cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels)

    if len(all_probs) == 0:
        return np.array([]), np.array([])

    return np.concatenate(all_probs), np.concatenate(all_labels)


def train_and_evaluate(
    model,
    model_name,
    train_loader,
    val_loader,
    test_loader,
    device,
    lr=None,
    weight_decay=None,
):
    print(f"\n[Run] {model_name}")

    model = model.to(device)
    weight_path = f"weights_{model_name}.pth"

    actual_lr = LEARNING_RATE if lr is None else lr
    actual_weight_decay = WEIGHT_DECAY if weight_decay is None else weight_decay
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_auc": [], "lr": []}

    if os.path.exists(weight_path):
        print(f"  -> 已有权重 {weight_path}，跳过训练")
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.AdamW(
            model.parameters(),
            lr=actual_lr,
            weight_decay=actual_weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=SCHEDULER_MIN_LR,
        )

        early_stopping = EarlyStoppingLoss(
            patience=PATIENCE,
            min_delta=EARLY_STOP_MIN_DELTA,
            path=weight_path,
        )

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate_loss(model, val_loader, criterion, device)
            val_probs, val_labels = predict_probabilities(model, val_loader, device)
            val_auc = safe_auc(val_labels, val_probs)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            history["epoch"].append(epoch)
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["val_auc"].append(float(val_auc))
            history["lr"].append(float(current_lr))

            print(
                f"  Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_auc={val_auc:.4f} | lr={current_lr:.6f}"
            )

            early_stopping(
                val_loss,
                model,
                extra_state={
                    "val_auc": val_auc,
                    "used_lr": actual_lr,
                    "used_weight_decay": actual_weight_decay,
                },
            )

            if early_stopping.early_stop:
                print(f"  -> stop at epoch {epoch}")
                break

        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    val_probs, val_labels = predict_probabilities(model, val_loader, device)
    chosen_threshold, chosen_val_metrics = choose_guarded_threshold(val_labels, val_probs)

    test_probs, test_labels = predict_probabilities(model, test_loader, device)
    test_metrics = compute_metrics(test_labels, test_probs, threshold=chosen_threshold)

    return {
        "model_name": model_name,
        "best_threshold": chosen_threshold,
        "val_loss_best": float(checkpoint.get("best_val_loss", np.nan)),
        "val_auc_best": float(checkpoint.get("val_auc", np.nan)),
        "val_bal_acc": chosen_val_metrics["BalancedAcc"],
        "used_lr": float(checkpoint.get("used_lr", actual_lr)),
        "used_weight_decay": float(checkpoint.get("used_weight_decay", actual_weight_decay)),
        "used_emotion_scale": 1.0,
        "teacher_copied_count": np.nan,
        "Accuracy": test_metrics["Accuracy"],
        "Precision": test_metrics["Precision"],
        "Recall": test_metrics["Recall"],
        "F1-Score": test_metrics["F1-Score"],
        "AUC": test_metrics["AUC"],
        "BalancedAcc": test_metrics["BalancedAcc"],
        "fpr": test_metrics["fpr"],
        "tpr": test_metrics["tpr"],
        "preds_prob": to_1d_numpy(test_probs),
        "preds_class": test_metrics["preds_class"],
        "labels": to_1d_numpy(test_labels).astype(int),
        "trained_model": model,
        "weight_path": weight_path,
        "history": history,
    }


def build_optimizer_for_trainable(model, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def train_m5_epoch_distill(model, teacher_model, loader, optimizer, device, distill_weight=0.2):
    model.train()
    teacher_model.eval()
    bce_criterion = nn.BCEWithLogitsLoss()

    total_losses = []
    bce_losses = []
    kd_losses = []

    for x_price, x_emotion, y in loader:
        x_price = x_price.to(device).float()
        x_emotion = x_emotion.to(device).float()
        y = y.to(device).float().view(-1)

        optimizer.zero_grad()
        student_logits = model(x_price, x_emotion).view(-1)

        with torch.no_grad():
            teacher_emotion = extract_teacher_emotion_from_m5plus(x_emotion)
            teacher_logits = teacher_model(x_price, teacher_emotion).view(-1)

        bce_loss = bce_criterion(student_logits, y)
        kd_loss = model.distill_loss_from_teacher_logits(
            student_logits,
            teacher_logits,
            weight=distill_weight,
        )
        loss = bce_loss + kd_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        total_losses.append(loss.item())
        bce_losses.append(bce_loss.item())
        kd_losses.append(kd_loss.item())

    return (
        float(np.mean(total_losses)) if total_losses else 0.0,
        float(np.mean(bce_losses)) if bce_losses else 0.0,
        float(np.mean(kd_losses)) if kd_losses else 0.0,
    )


def evaluate_m5_loss_distill(model, teacher_model, loader, device, distill_weight=0.2):
    model.eval()
    teacher_model.eval()
    bce_criterion = nn.BCEWithLogitsLoss()

    total_losses = []
    bce_losses = []
    kd_losses = []

    with torch.no_grad():
        for x_price, x_emotion, y in loader:
            x_price = x_price.to(device).float()
            x_emotion = x_emotion.to(device).float()
            y = y.to(device).float().view(-1)

            student_logits = model(x_price, x_emotion).view(-1)
            teacher_emotion = extract_teacher_emotion_from_m5plus(x_emotion)
            teacher_logits = teacher_model(x_price, teacher_emotion).view(-1)

            bce_loss = bce_criterion(student_logits, y)
            kd_loss = model.distill_loss_from_teacher_logits(
                student_logits,
                teacher_logits,
                weight=distill_weight,
            )
            loss = bce_loss + kd_loss

            total_losses.append(loss.item())
            bce_losses.append(bce_loss.item())
            kd_losses.append(kd_loss.item())

    total = float(np.mean(total_losses)) if total_losses else np.inf
    bce = float(np.mean(bce_losses)) if bce_losses else np.inf
    kd = float(np.mean(kd_losses)) if kd_losses else np.inf
    return total, bce, kd


def train_m5_with_teacher(
    model,
    teacher_model,
    train_loader,
    val_loader,
    test_loader,
    device,
    run_suffix="",
):
    display_name = f"M5_Ultimate_Adaptive_Emotion{run_suffix}"
    print(f"\n[Run] {display_name}_M5Plus")

    model = model.to(device)
    best_path = f"weights_M5_Ultimate_Adaptive_Emotion_M5Plus{run_suffix}.pth"
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_auc": [], "lr": [], "stage": []}
    copied_count = 0

    if os.path.exists(best_path):
        print(f"  -> 已有权重 {best_path}，跳过训练")
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        copied_count = int(checkpoint.get("teacher_copied_count", 0))
    else:
        teacher_model = teacher_model.to(device)
        teacher_model.eval()

        model.set_residual_alpha(M5_RESIDUAL_ALPHA)
        copy_info = model.load_backbone_from_teacher_model(teacher_model, verbose=True)
        copied_count = copy_info["copied_count"]

        early_stopping = EarlyStoppingLoss(
            patience=PATIENCE,
            min_delta=EARLY_STOP_MIN_DELTA,
            path=best_path,
        )

        if copied_count > 0:
            print(
                f"[Stage 1] copied={copied_count}, "
                f"input_proj_weight_copied={copy_info['input_proj_weight_copied']}"
            )
            model.freeze_backbone_by_copy_info(copy_info)

        else:
            print("[Stage 1] no copied tensors, run warmup without freezing backbone")

        optimizer = build_optimizer_for_trainable(
            model,
            lr=M5_STAGE1_LR,
            weight_decay=M5_WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=SCHEDULER_MIN_LR,
        )

        for epoch in range(1, M5_STAGE1_EPOCHS + 1):
            train_total, train_bce, train_kd = train_m5_epoch_distill(
                model,
                teacher_model,
                train_loader,
                optimizer,
                device,
                distill_weight=M5_STAGE1_DISTILL_WEIGHT,
            )
            val_total, val_bce, val_kd = evaluate_m5_loss_distill(
                model,
                teacher_model,
                val_loader,
                device,
                distill_weight=M5_STAGE1_DISTILL_WEIGHT,
            )
            val_probs, val_labels = predict_probabilities(model, val_loader, device)
            val_auc = safe_auc(val_labels, val_probs)
            scheduler.step(val_total)
            current_lr = optimizer.param_groups[0]["lr"]

            history["epoch"].append(epoch)
            history["train_loss"].append(float(train_total))
            history["val_loss"].append(float(val_total))
            history["val_auc"].append(float(val_auc))
            history["lr"].append(float(current_lr))
            history["stage"].append("stage1")

            print(
                f"  [Stage1 {epoch:02d}/{M5_STAGE1_EPOCHS}] "
                f"train_total={train_total:.4f} (bce={train_bce:.4f}, kd={train_kd:.4f}) | "
                f"val_total={val_total:.4f} (bce={val_bce:.4f}, kd={val_kd:.4f}) | "
                f"val_auc={val_auc:.4f} | lr={current_lr:.6f}"
            )

            early_stopping(
                val_total,
                model,
                extra_state={
                    "stage": "stage1",
                    "val_auc": val_auc,
                    "used_lr": M5_STAGE1_LR,
                    "used_weight_decay": M5_WEIGHT_DECAY,
                    "used_residual_alpha": M5_RESIDUAL_ALPHA,
                    "teacher_copied_count": copied_count,
                    "teacher_copy_info": copy_info,
                    "stage1_distill_weight": M5_STAGE1_DISTILL_WEIGHT,
                    "stage2_distill_weight": M5_STAGE2_DISTILL_WEIGHT,
                },
            )

            if early_stopping.early_stop:
                print("  -> stop in stage 1")
                break

        print(f"[Stage 2] unfreeze last {M5_UNFREEZE_LAST_N} backbone layer(s)")
        model.unfreeze_backbone_last_layers(n_last_layers=M5_UNFREEZE_LAST_N)

        optimizer = build_optimizer_for_trainable(
            model,
            lr=M5_STAGE2_LR,
            weight_decay=M5_WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=SCHEDULER_MIN_LR,
        )

        early_stopping.early_stop = False
        early_stopping.counter = 0

        for epoch in range(1, M5_STAGE2_EPOCHS + 1):
            train_total, train_bce, train_kd = train_m5_epoch_distill(
                model,
                teacher_model,
                train_loader,
                optimizer,
                device,
                distill_weight=M5_STAGE2_DISTILL_WEIGHT,
            )
            val_total, val_bce, val_kd = evaluate_m5_loss_distill(
                model,
                teacher_model,
                val_loader,
                device,
                distill_weight=M5_STAGE2_DISTILL_WEIGHT,
            )
            val_probs, val_labels = predict_probabilities(model, val_loader, device)
            val_auc = safe_auc(val_labels, val_probs)
            scheduler.step(val_total)
            current_lr = optimizer.param_groups[0]["lr"]

            history["epoch"].append(M5_STAGE1_EPOCHS + epoch)
            history["train_loss"].append(float(train_total))
            history["val_loss"].append(float(val_total))
            history["val_auc"].append(float(val_auc))
            history["lr"].append(float(current_lr))
            history["stage"].append("stage2")

            print(
                f"  [Stage2 {epoch:02d}/{M5_STAGE2_EPOCHS}] "
                f"train_total={train_total:.4f} (bce={train_bce:.4f}, kd={train_kd:.4f}) | "
                f"val_total={val_total:.4f} (bce={val_bce:.4f}, kd={val_kd:.4f}) | "
                f"val_auc={val_auc:.4f} | lr={current_lr:.6f}"
            )

            early_stopping(
                val_total,
                model,
                extra_state={
                    "stage": "stage2",
                    "val_auc": val_auc,
                    "used_lr": M5_STAGE2_LR,
                    "used_weight_decay": M5_WEIGHT_DECAY,
                    "used_residual_alpha": M5_RESIDUAL_ALPHA,
                    "teacher_copied_count": copied_count,
                    "teacher_copy_info": copy_info,
                    "stage1_distill_weight": M5_STAGE1_DISTILL_WEIGHT,
                    "stage2_distill_weight": M5_STAGE2_DISTILL_WEIGHT,
                },
            )

            if early_stopping.early_stop:
                print("  -> stop in stage 2")
                break

        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    val_probs, val_labels = predict_probabilities(model, val_loader, device)
    chosen_threshold, chosen_val_metrics = choose_guarded_threshold(val_labels, val_probs)

    test_probs, test_labels = predict_probabilities(model, test_loader, device)
    test_metrics = compute_metrics(test_labels, test_probs, threshold=chosen_threshold)

    return {
        "model_name": display_name,
        "best_threshold": chosen_threshold,
        "val_loss_best": float(checkpoint.get("best_val_loss", np.nan)),
        "val_auc_best": float(checkpoint.get("val_auc", np.nan)),
        "val_bal_acc": chosen_val_metrics["BalancedAcc"],
        "used_lr": float(checkpoint.get("used_lr", M5_STAGE2_LR)),
        "used_weight_decay": float(checkpoint.get("used_weight_decay", M5_WEIGHT_DECAY)),
        "used_emotion_scale": M5_RESIDUAL_ALPHA,
        "teacher_copied_count": int(checkpoint.get("teacher_copied_count", copied_count)),
        "Accuracy": test_metrics["Accuracy"],
        "Precision": test_metrics["Precision"],
        "Recall": test_metrics["Recall"],
        "F1-Score": test_metrics["F1-Score"],
        "AUC": test_metrics["AUC"],
        "BalancedAcc": test_metrics["BalancedAcc"],
        "fpr": test_metrics["fpr"],
        "tpr": test_metrics["tpr"],
        "preds_prob": to_1d_numpy(test_probs),
        "preds_class": test_metrics["preds_class"],
        "labels": to_1d_numpy(test_labels).astype(int),
        "trained_model": model,
        "weight_path": best_path,
        "history": history,
    }


def plot_comprehensive_metrics_bar(results):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC", "BalancedAcc"]
    models = [res["model_name"].replace("_", "\n") for res in results]

    x = np.arange(len(models))
    width = 0.13
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

    for i, metric in enumerate(metrics):
        values = [res[metric] for res in results]
        ax.bar(
            x + i * width,
            values,
            width,
            label=metric,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_ylabel("Score")
    ax.set_title("Metrics Comparison", fontweight="bold", pad=20)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(models, rotation=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=6,
        frameon=True,
        shadow=True,
    )
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig("fig1_metrics_comparison_bar_m5plus.png", dpi=300)
    print("保存: fig1_metrics_comparison_bar_m5plus.png")
    plt.close(fig)


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), dpi=300)
    fig.suptitle("Confusion Matrices", fontsize=18, fontweight="bold", y=0.98)
    axes = axes.flatten()

    for i, res in enumerate(results):
        cm = confusion_matrix(res["labels"], res["preds_class"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i],
            cbar=False,
            annot_kws={"size": 14, "weight": "bold"},
        )
        axes[i].set_title(
            f"{res['model_name'].split('_', 1)[-1]}\n(th={res['best_threshold']:.2f})",
            fontweight="bold",
        )
        axes[i].set_xlabel("Predicted Label")
        axes[i].set_ylabel("True Label")
        axes[i].set_yticklabels(["Down (0)", "Up (1)"], rotation=0)
        axes[i].set_xticklabels(["Down (0)", "Up (1)"])

    for j in range(len(results), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("fig2_confusion_matrices_m5plus.png", dpi=300)
    print("保存: fig2_confusion_matrices_m5plus.png")
    plt.close(fig)


def plot_enhanced_roc(results):
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    line_styles = [":", "--", "-.", "-", "-"]

    for i, res in enumerate(results):
        lw = 2.5 if ("Ultimate" in res["model_name"] or "Adaptive" in res["model_name"]) else 1.5
        ax.plot(
            res["fpr"],
            res["tpr"],
            color=colors[i],
            linestyle=line_styles[i],
            linewidth=lw,
            label=f"{res['model_name']} (AUC = {res['AUC']:.4f})",
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guessing")
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, edgecolor="black", fancybox=True)

    plt.savefig("fig3_enhanced_roc_m5plus.png", dpi=300)
    print("保存: fig3_enhanced_roc_m5plus.png")
    plt.close(fig)


def plot_results_summary_table(results):
    df = pd.DataFrame(
        [
            {
                "Model": r["model_name"].replace("_", " "),
                "Threshold": r["best_threshold"],
                "Accuracy": r["Accuracy"],
                "Precision": r["Precision"],
                "Recall": r["Recall"],
                "F1": r["F1-Score"],
                "AUC": r["AUC"],
                "BalancedAcc": r["BalancedAcc"],
            }
            for r in results
        ]
    )
    for c in ["Threshold", "Accuracy", "Precision", "Recall", "F1", "AUC", "BalancedAcc"]:
        df[c] = df[c].map(lambda x: f"{x:.4f}")

    fig, ax = plt.subplots(figsize=(15, 3.6), dpi=300)
    ax.axis("off")
    ax.set_title("Results Summary", fontsize=14, fontweight="bold", pad=14)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAF2")
        elif row % 2 == 1:
            cell.set_facecolor("#F8F8FB")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig("fig0_results_summary_table.png", dpi=300, bbox_inches="tight")
    print("保存: fig0_results_summary_table.png")
    plt.close(fig)


def calculate_max_drawdown(equity_curve):
    equity_curve = np.asarray(equity_curve, dtype=float)
    if len(equity_curve) == 0:
        return np.nan

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def compute_backtest_metrics(prices, signals, transaction_cost=0.0):
    prices = np.asarray(prices, dtype=float)
    signals = np.asarray(signals, dtype=int)

    n = min(len(prices), len(signals))
    prices = prices[:n]
    signals = signals[:n]

    if n < 2:
        raise ValueError("回测数据长度不足")

    market_returns = prices[1:] / prices[:-1] - 1.0
    positions = signals[:-1].astype(float)
    position_changes = np.abs(np.diff(signals)).astype(float)

    strategy_returns = positions * market_returns
    if transaction_cost > 0:
        strategy_returns = strategy_returns - position_changes * transaction_cost

    buy_hold_returns = market_returns

    strategy_equity = np.concatenate([[1.0], np.cumprod(1.0 + strategy_returns)])
    buy_hold_equity = np.concatenate([[1.0], np.cumprod(1.0 + buy_hold_returns)])

    return {
        "strategy": "M5_long_cash",
        "strategy_total_return": float(strategy_equity[-1] - 1.0),
        "buy_hold_total_return": float(buy_hold_equity[-1] - 1.0),
        "strategy_max_drawdown": calculate_max_drawdown(strategy_equity),
        "buy_hold_max_drawdown": calculate_max_drawdown(buy_hold_equity),
        "number_of_trades": int(position_changes.sum()),
        "transaction_cost": float(transaction_cost),
        "n_test_points": int(n),
    }, strategy_equity, buy_hold_equity


def plot_trading_strategy_backtest(
    results,
    test_loader=None,
    csv_path="btc_multimodal_hourly_dataset.csv",
    threshold_override=None,
    transaction_cost=0.0,
    plot_last_n=240,
):
    best_result = next((res for res in results if "M5" in res["model_name"]), results[-1])

    probs = to_1d_numpy(best_result["preds_prob"])
    threshold = best_result["best_threshold"] if threshold_override is None else float(threshold_override)
    signals = (probs >= threshold).astype(int)

    prices = None
    dates = None

    if test_loader is not None:
        dataset = test_loader.dataset
        if hasattr(dataset, "sample_close"):
            prices = np.asarray(dataset.sample_close, dtype=float)
        if hasattr(dataset, "sample_dates"):
            dates = np.asarray(dataset.sample_dates)

    if prices is None or len(prices) == 0:
        raw_df = pd.read_csv(csv_path).dropna(subset=["close"]).reset_index(drop=True)
        n = len(signals)
        valid_df = raw_df.tail(n).copy()
        prices = valid_df["close"].values

        if "date_hour" in valid_df.columns:
            dates = valid_df["date_hour"].astype(str).values
        elif "timestamp" in valid_df.columns:
            dates = valid_df["timestamp"].astype(str).values
        else:
            dates = np.arange(len(valid_df)).astype(str)

    n = min(len(prices), len(signals))
    prices = prices[:n]
    signals = signals[:n]

    if dates is None or len(dates) == 0:
        dates = np.arange(n)
    else:
        dates = dates[:n]

    try:
        dates_plot = pd.to_datetime(dates)
    except Exception:
        dates_plot = np.arange(n)

    metrics, strategy_equity, buy_hold_equity = compute_backtest_metrics(
        prices=prices,
        signals=signals,
        transaction_cost=transaction_cost,
    )

    pd.DataFrame([metrics]).to_csv("fig4_backtest_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "date": dates[: len(strategy_equity)],
            "price": prices[: len(strategy_equity)],
            "signal": signals[: len(strategy_equity)],
            "strategy_equity": strategy_equity,
            "buy_hold_equity": buy_hold_equity,
        }
    ).to_csv("fig4_backtest_equity.csv", index=False, encoding="utf-8-sig")

    print("保存: fig4_backtest_metrics.csv")
    print("保存: fig4_backtest_equity.csv")

    start = max(0, n - plot_last_n)
    plot_dates = dates_plot[start:n]
    plot_prices = prices[start:n]
    plot_signals = signals[start:n]

    fig, ax1 = plt.subplots(figsize=(15, 6), dpi=300)
    ax1.plot(plot_dates, plot_prices, color="black", linewidth=1.5, label="BTC Price")

    diffs = np.diff(plot_signals, prepend=plot_signals[0])
    buy_idx = np.where(diffs == 1)[0]
    sell_idx = np.where(diffs == -1)[0]

    if len(plot_prices) > 0:
        ax1.scatter(
            plot_dates[buy_idx],
            plot_prices[buy_idx] * 0.98,
            marker="^",
            color="#2ca02c",
            s=120,
            label="BUY",
            zorder=5,
        )
        ax1.scatter(
            plot_dates[sell_idx],
            plot_prices[sell_idx] * 1.02,
            marker="v",
            color="#d62728",
            s=120,
            label="SELL / CASH",
            zorder=5,
        )
        ax1.fill_between(
            plot_dates,
            plot_prices.min() * 0.95,
            plot_prices.max() * 1.05,
            where=(plot_signals == 1),
            facecolor="green",
            alpha=0.08,
            label="Long",
        )
        ax1.fill_between(
            plot_dates,
            plot_prices.min() * 0.95,
            plot_prices.max() * 1.05,
            where=(plot_signals == 0),
            facecolor="red",
            alpha=0.06,
            label="Cash",
        )
        ax1.set_ylim(plot_prices.min() * 0.98, plot_prices.max() * 1.02)

    ax1.set_ylabel("BTC Price")
    ax1.set_title(f"Backtest Signal Visualization ({best_result['model_name']})", fontweight="bold", pad=15)

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper left", ncol=2)

    if len(plot_dates) > 0 and isinstance(plot_dates[0], pd.Timestamp):
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("fig4_trading_backtest.png", dpi=300, bbox_inches="tight")
    print("保存: fig4_trading_backtest.png")
    plt.close(fig)


def plot_temporal_representation_similarity_heatmap(model, device, test_loader):
    model.eval()
    target_x_price = None
    target_x_emotion = None

    for x_price, x_emotion, y in test_loader:
        x_price = x_price.float()
        volatility = x_price[:, :, -1]
        max_vol_idx = torch.argmax(volatility.mean(dim=1))
        target_x_price = x_price[max_vol_idx : max_vol_idx + 1].to(device)
        target_x_emotion = x_emotion[max_vol_idx : max_vol_idx + 1].float().to(device)
        break

    if target_x_price is None:
        print("跳过图5：没有测试样本")
        return

    with torch.no_grad():
        if hasattr(model, "build_teacher_backbone") and hasattr(model, "emotion_fusion"):
            h_base = model.build_teacher_backbone(target_x_price, target_x_emotion)
            fused_repr = model.emotion_fusion(h_base, target_x_emotion)
        elif (
            hasattr(model, "price_proj")
            and hasattr(model, "price_transformer")
            and hasattr(model, "emotion_fusion")
        ):
            p_repr = model.price_proj(target_x_price)
            p_repr = model.pos_encoder(p_repr)
            p_repr = model.price_transformer(p_repr)
            if hasattr(model, "price_norm"):
                p_repr = model.price_norm(p_repr)
            fused_repr = model.emotion_fusion(p_repr, target_x_emotion)
        else:
            print("跳过图5：当前模型不支持导出时间表示")
            return

    fused_norm = torch.nn.functional.normalize(fused_repr, p=2, dim=-1)
    similarity = torch.bmm(fused_norm, fused_norm.transpose(1, 2))[0].cpu().numpy()
    normalized_similarity = (similarity + 1.0) / 2.0

    fig = plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(
        normalized_similarity,
        cmap="magma",
        cbar_kws={"label": "Normalized Representation Similarity"},
        linewidths=0.4,
        linecolor="gray",
    )
    plt.title("Temporal Representation Similarity Heatmap", fontweight="bold", pad=15)
    plt.xlabel("Time Steps")
    plt.ylabel("Time Steps")
    plt.tight_layout()
    plt.savefig("fig5_temporal_representation_similarity_heatmap.png", dpi=300, bbox_inches="tight")
    print("保存: fig5_temporal_representation_similarity_heatmap.png")
    plt.close(fig)

def plot_training_curves(results):
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n), dpi=300, sharex=False)

    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        hist = res.get("history")
        if not hist or len(hist.get("epoch", [])) == 0:
            ax.axis("off")
            continue

        epochs = hist["epoch"]
        ax.plot(epochs, hist["train_loss"], label="Train Loss", linewidth=1.8)
        ax.plot(epochs, hist["val_loss"], label="Validation Loss", linewidth=1.8)

        ax2 = ax.twinx()
        ax2.plot(epochs, hist["val_auc"], label="Validation AUC", linestyle="--", linewidth=1.8)

        ax.set_title(f"Training Curves: {res['model_name']}", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax2.set_ylabel("AUC")
        ax.grid(True, alpha=0.3)

        if "stage" in hist and any(s == "stage2" for s in hist["stage"]):
            stage2_epochs = [e for e, s in zip(hist["epoch"], hist["stage"]) if s == "stage2"]
            if stage2_epochs:
                ax.axvline(stage2_epochs[0] - 0.5, color="gray", linestyle=":", linewidth=1.2)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig("fig6_training_curves.png", dpi=300, bbox_inches="tight")
    print("保存: fig6_training_curves.png")
    plt.close(fig)


if __name__ == "__main__":
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System executing on: {device}")
    print(f"Random seed fixed at: {SEED}")

    fixed_cache_path = os.path.join(CACHE_DIR, FIXED_CACHE_NAME)
    adaptive_cache_m4_path = os.path.join(CACHE_DIR, ADAPTIVE_CACHE_NAME_M4)
    adaptive_cache_m5_path = os.path.join(CACHE_DIR, ADAPTIVE_CACHE_NAME_M5)

    if not os.path.exists(fixed_cache_path):
        raise FileNotFoundError(f"找不到缓存文件: {fixed_cache_path}")
    if not os.path.exists(adaptive_cache_m4_path):
        raise FileNotFoundError(f"找不到缓存文件: {adaptive_cache_m4_path}")
    if not os.path.exists(adaptive_cache_m5_path):
        raise FileNotFoundError(f"找不到缓存文件: {adaptive_cache_m5_path}")

    fixed_cache = torch.load(fixed_cache_path, map_location="cpu", weights_only=False)
    adaptive_cache_m4 = torch.load(adaptive_cache_m4_path, map_location="cpu", weights_only=False)
    adaptive_cache_m5 = torch.load(adaptive_cache_m5_path, map_location="cpu", weights_only=False)

    print_cache_meta(FIXED_CACHE_NAME, fixed_cache)
    print_cache_meta(ADAPTIVE_CACHE_NAME_M4, adaptive_cache_m4)
    print_cache_meta(ADAPTIVE_CACHE_NAME_M5, adaptive_cache_m5)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    tl_f = DataLoader(
        fixed_cache["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=NUM_WORKERS,
    )
    vl_f = DataLoader(
        fixed_cache["val"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    tsl_f = DataLoader(
        fixed_cache["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    tl_a = DataLoader(
        adaptive_cache_m4["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=NUM_WORKERS,
    )
    vl_a = DataLoader(
        adaptive_cache_m4["val"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    tsl_a = DataLoader(
        adaptive_cache_m4["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    tl_m5 = DataLoader(
        adaptive_cache_m5["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=NUM_WORKERS,
    )
    vl_m5 = DataLoader(
        adaptive_cache_m5["val"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    tsl_m5 = DataLoader(
        adaptive_cache_m5["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    p_dim_f = fixed_cache["p_dim"]
    e_dim_f = fixed_cache["e_dim"]
    p_dim_a = adaptive_cache_m4["p_dim"]
    e_dim_a = adaptive_cache_m4["e_dim"]
    p_dim_m5 = adaptive_cache_m5["p_dim"]
    e_dim_m5 = adaptive_cache_m5["e_dim"]

    experiments = []

    model1 = BaselineLSTM(price_dim=p_dim_f)
    experiments.append(train_and_evaluate(model1, "M1_Baseline_LSTM", tl_f, vl_f, tsl_f, device))

    model2 = VanillaTransformer(price_dim=p_dim_f, emotion_dim=e_dim_f)
    experiments.append(train_and_evaluate(model2, "M2_Vanilla_Trans", tl_f, vl_f, tsl_f, device))

    model3 = InnovativeDualAttentionModel(
        price_dim=p_dim_f,
        emotion_dim=e_dim_f,
        residual_alpha=0.10,
    )
    experiments.append(train_and_evaluate(model3, "M3_EmotionAttn_Fixed", tl_f, vl_f, tsl_f, device))

    model4 = VanillaTransformer(price_dim=p_dim_a, emotion_dim=e_dim_a)
    res4 = train_and_evaluate(model4, "M4_Vanilla_Adaptive", tl_a, vl_a, tsl_a, device)
    experiments.append(res4)

    teacher_model = res4["trained_model"]
    teacher_model.eval()

    model5 = InnovativeDualAttentionModelM5Only(
        price_dim=p_dim_m5,
        emotion_dim=e_dim_m5,
        residual_alpha=M5_RESIDUAL_ALPHA,
    )
    res5 = train_m5_with_teacher(model5, teacher_model, tl_m5, vl_m5, tsl_m5, device)
    experiments.append(res5)

    print("\n" + "=" * 148)
    print("实验结果汇总")
    print("=" * 148)
    print(
        f"| {'Model Config':<30} | {'th':<5} | {'lr':<8} | {'wd':<8} | {'emo':<5} | {'copy':<5} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'AUC':<6} | {'BalAcc':<6} |"
    )
    print(
        "|"
        + "-" * 32
        + "|"
        + "-" * 7
        + "|"
        + "-" * 10
        + "|"
        + "-" * 10
        + "|"
        + "-" * 7
        + "|"
        + "-" * 7
        + "|"
        + "-" * 8
        + "|"
        + "-" * 8
        + "|"
        + "-" * 8
        + "|"
        + "-" * 8
        + "|"
        + "-" * 8
        + "|"
        + "-" * 8
        + "|"
    )

    for res in experiments:
        name = res["model_name"][:30]
        copied = res.get("teacher_copied_count", np.nan)
        copied_str = f"{int(copied)}" if copied == copied else "nan"
        print(
            f"| {name:<30} | {res['best_threshold']:.2f}  | "
            f"{res['used_lr']:<8.1e} | {res['used_weight_decay']:<8.1e} | {res['used_emotion_scale']:<5.2f} | {copied_str:<5} | "
            f"{res['Accuracy']:.4f} | {res['Precision']:.4f} | {res['Recall']:.4f} | {res['F1-Score']:.4f} | {res['AUC']:.4f} | {res['BalancedAcc']:.4f} |"
        )
    print("=" * 148)

    pd.DataFrame(
        [
            {
                "model_name": res["model_name"],
                "best_threshold": res["best_threshold"],
                "used_lr": res["used_lr"],
                "used_weight_decay": res["used_weight_decay"],
                "used_emotion_scale": res["used_emotion_scale"],
                "teacher_copied_count": res.get("teacher_copied_count", np.nan),
                "Accuracy": res["Accuracy"],
                "Precision": res["Precision"],
                "Recall": res["Recall"],
                "F1-Score": res["F1-Score"],
                "AUC": res["AUC"],
                "BalancedAcc": res["BalancedAcc"],
                "val_loss_best": res["val_loss_best"],
                "val_auc_best": res["val_auc_best"],
            }
            for res in experiments
        ]
    ).to_csv("ablation_results_teacher_m5_m5plus.csv", index=False, encoding="utf-8-sig")
    print("保存: ablation_results_teacher_m5_m5plus.csv")

    with open("run_config_teacher_m5_m5plus.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "epochs": EPOCHS,
                "patience": PATIENCE,
                "default_lr": LEARNING_RATE,
                "default_weight_decay": WEIGHT_DECAY,
                "adaptive_cache_m4": adaptive_cache_m4_path,
                "adaptive_cache_m5": adaptive_cache_m5_path,
                "m5_stage1_epochs": M5_STAGE1_EPOCHS,
                "m5_stage2_epochs": M5_STAGE2_EPOCHS,
                "m5_stage1_lr": M5_STAGE1_LR,
                "m5_stage2_lr": M5_STAGE2_LR,
                "m5_weight_decay": M5_WEIGHT_DECAY,
                "m5_stage1_distill_weight": M5_STAGE1_DISTILL_WEIGHT,
                "m5_stage2_distill_weight": M5_STAGE2_DISTILL_WEIGHT,
                "m5_residual_alpha": M5_RESIDUAL_ALPHA,
                "m5_unfreeze_last_n": M5_UNFREEZE_LAST_N,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("保存: run_config_teacher_m5_m5plus.json")

    print("\n生成图表...")
    plot_results_summary_table(experiments)
    plot_comprehensive_metrics_bar(experiments)
    plot_confusion_matrices(experiments)
    plot_enhanced_roc(experiments)

    try:
        plot_trading_strategy_backtest(experiments, test_loader=tsl_m5, transaction_cost=0.0)
    except Exception as e:
        print(f"图4生成失败: {e}")

    try:
        plot_temporal_representation_similarity_heatmap(experiments[-1]["trained_model"], device, tsl_m5)
    except Exception as e:
        print(f"图5生成失败: {e}")

    try:
        plot_training_curves(experiments)
    except Exception as e:
        print(f"图6生成失败: {e}")

    print("\n运行结束")