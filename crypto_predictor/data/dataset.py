import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

from config.config import DataConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class CryptoFixedWindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        price_cols: List[str],
        emotion_cols: List[str],
        target_col: str,
    ):
        self.seq_len = seq_len
        self.price_data = df[price_cols].values.astype(np.float32)
        self.emotion_data = df[emotion_cols].values.astype(np.float32)
        self.target_data = df[target_col].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.price_data) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_price = torch.tensor(self.price_data[idx : idx + self.seq_len])
        x_emotion = torch.tensor(self.emotion_data[idx : idx + self.seq_len])
        y = torch.tensor(self.target_data[idx + self.seq_len - 1])
        return x_price, x_emotion, y


def _split_and_scale(
    df: pd.DataFrame,
    price_cols: List[str],
    emotion_cols: List[str],
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    price_scaler = StandardScaler()
    emotion_scaler = StandardScaler()

    train_df[price_cols] = price_scaler.fit_transform(train_df[price_cols])
    val_df[price_cols] = price_scaler.transform(val_df[price_cols])
    test_df[price_cols] = price_scaler.transform(test_df[price_cols])

    train_df[emotion_cols] = emotion_scaler.fit_transform(train_df[emotion_cols])
    val_df[emotion_cols] = emotion_scaler.transform(val_df[emotion_cols])
    test_df[emotion_cols] = emotion_scaler.transform(test_df[emotion_cols])

    return train_df, val_df, test_df


def get_dataloaders(
    csv_path: str,
    cfg: DataConfig = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    if cfg is None:
        cfg = DataConfig()

    logger.info(f"加载数据集: {csv_path}")
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)

    train_df, val_df, test_df = _split_and_scale(
        df,
        cfg.price_cols,
        cfg.emotion_cols,
        cfg.train_ratio,
        cfg.val_ratio,
    )

    train_ds = CryptoFixedWindowDataset(
        train_df, cfg.seq_len, cfg.price_cols, cfg.emotion_cols, cfg.target_col
    )
    val_ds = CryptoFixedWindowDataset(
        val_df, cfg.seq_len, cfg.price_cols, cfg.emotion_cols, cfg.target_col
    )
    test_ds = CryptoFixedWindowDataset(
        test_df, cfg.seq_len, cfg.price_cols, cfg.emotion_cols, cfg.target_col
    )

    logger.info(f"样本数 - train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        len(cfg.price_cols),
        len(cfg.emotion_cols),
    )