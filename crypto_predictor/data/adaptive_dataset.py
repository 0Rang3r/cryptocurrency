import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import Tuple, List

from config.config import DataConfig
from utils.logger import get_logger

logger = get_logger(__name__)

STATE_STEP_MAP = {
    "S3": 1,
    "S2": 2,
    "S1": 4,
}


def assign_market_state(
    df: pd.DataFrame,
    vol_col: str = "Vol_t",
    window: int = 720,
) -> pd.DataFrame:
    df = df.copy()

    rolling_mean = df[vol_col].rolling(window=window, min_periods=1).mean()
    rolling_std = df[vol_col].rolling(window=window, min_periods=1).std().fillna(0)

    conditions = [
        df[vol_col] > (rolling_mean + rolling_std),
        df[vol_col] < (rolling_mean - rolling_std),
    ]
    df["Market_State"] = np.select(conditions, ["S3", "S1"], default="S2")

    state_counts = df["Market_State"].value_counts().to_dict()
    logger.info(f"市场状态分布: {state_counts}")
    return df


class AdaptiveWindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        price_cols: List[str],
        emotion_cols: List[str],
        target_col: str,
        max_step: int = 4,
    ):
        df = df.reset_index(drop=True)

        self.seq_len = seq_len
        self.price_cols = price_cols
        self.emotion_cols = emotion_cols
        self.max_step = max_step

        self.samples_price: List[np.ndarray] = []
        self.samples_emotion: List[np.ndarray] = []
        self.labels: List[float] = []

        safe_start = seq_len * max_step + 10

        logger.info(f"构建自适应窗口样本，候选位置数: {len(df) - safe_start}")
        for idx in tqdm(range(safe_start, len(df) - 1), desc="AdaptiveWindow"):
            seq_p, seq_e = self._build_sequence(df, idx)
            if seq_p is None:
                continue

            self.samples_price.append(seq_p)
            self.samples_emotion.append(seq_e)
            self.labels.append(df.loc[idx, target_col])

        self.samples_price = np.array(self.samples_price, dtype=np.float32)
        self.samples_emotion = np.array(self.samples_emotion, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

        logger.info(f"样本构建完成，有效样本数: {len(self.labels)}")

    def _build_sequence(
        self,
        df: pd.DataFrame,
        start_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        curr = start_idx
        seq_p, seq_e = [], []

        for _ in range(self.seq_len):
            if curr < 0:
                break

            state = df.loc[curr, "Market_State"]
            step = STATE_STEP_MAP.get(state, 2)
            actual_step = min(step, curr + 1)

            window_df = df.loc[curr - actual_step + 1: curr]
            agg_price = window_df[self.price_cols].mean().values
            agg_emotion = window_df[self.emotion_cols].mean().values

            dt_feature = np.array([actual_step / self.max_step], dtype=np.float32)
            agg_price = np.concatenate([agg_price, dt_feature])

            seq_p.append(agg_price)
            seq_e.append(agg_emotion)

            curr -= actual_step

        if len(seq_p) < self.seq_len:
            return None, None

        seq_p.reverse()
        seq_e.reverse()
        return np.array(seq_p), np.array(seq_e)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.samples_price[idx]),
            torch.tensor(self.samples_emotion[idx]),
            torch.tensor(self.labels[idx]),
        )


def get_adaptive_dataloaders(
    csv_path: str,
    cfg: DataConfig = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    if cfg is None:
        cfg = DataConfig()

    logger.info(f"加载数据集: {csv_path}")

    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    df = assign_market_state(df, vol_col="Vol_t", window=cfg.vol_rolling_window)

    n = len(df)
    train_end = int(n * cfg.train_ratio)
    val_end = int(n * (cfg.train_ratio + cfg.val_ratio))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    p_scaler = StandardScaler()
    e_scaler = StandardScaler()

    train_df[cfg.price_cols] = p_scaler.fit_transform(train_df[cfg.price_cols])
    val_df[cfg.price_cols] = p_scaler.transform(val_df[cfg.price_cols])
    test_df[cfg.price_cols] = p_scaler.transform(test_df[cfg.price_cols])

    train_df[cfg.emotion_cols] = e_scaler.fit_transform(train_df[cfg.emotion_cols])
    val_df[cfg.emotion_cols] = e_scaler.transform(val_df[cfg.emotion_cols])
    test_df[cfg.emotion_cols] = e_scaler.transform(test_df[cfg.emotion_cols])

    train_ds = AdaptiveWindowDataset(
        train_df, cfg.seq_len, cfg.price_cols, cfg.emotion_cols, cfg.target_col
    )
    val_ds = AdaptiveWindowDataset(
        val_df, cfg.seq_len, cfg.price_cols, cfg.emotion_cols, cfg.target_col
    )
    test_ds = AdaptiveWindowDataset(
        test_df, cfg.seq_len, cfg.price_cols, cfg.emotion_cols, cfg.target_col
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    price_dim = len(cfg.price_cols) + 1
    emotion_dim = len(cfg.emotion_cols)

    return train_loader, val_loader, test_loader, price_dim, emotion_dim