import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class AdaptiveWindowDataset(Dataset):
    def __init__(
        self,
        df,
        seq_len,
        price_cols,
        emotion_base_cols,
        target_col,
        emotion_lag=1,
    ):
        df = df.reset_index(drop=True)

        self.seq_len = seq_len
        self.price_cols = price_cols
        self.emotion_base_cols = emotion_base_cols
        self.target_col = target_col
        self.emotion_lag = emotion_lag

        self.samples_price = []
        self.samples_emotion = []
        self.labels = []
        self.sample_indices = []
        self.sample_close = []
        self.sample_dates = []

        safe_start_idx = self.seq_len * 4 + 12

        print("构建自适应窗口样本...")
        for idx in tqdm(range(safe_start_idx, len(df))):
            seq_price = []
            seq_emotion = []
            curr_idx = idx

            for _ in range(self.seq_len):
                if curr_idx < 0:
                    break

                state = df.loc[curr_idx, "Market_State"]

                if state == "S3":
                    step = 1
                elif state == "S2":
                    step = 2
                else:
                    step = 4

                if curr_idx - step + 1 < 0:
                    step = curr_idx + 1

                price_window = df.loc[curr_idx - step + 1 : curr_idx]

                agg_price = np.array(
                    [
                        price_window["open"].iloc[0],
                        price_window["high"].max(),
                        price_window["low"].min(),
                        price_window["close"].iloc[-1],
                        price_window["volume"].sum(),
                        price_window["return"].sum(),
                        price_window["Vol_t"].mean(),
                    ],
                    dtype=np.float32,
                )

                dt_feature = np.array([step / 4.0], dtype=np.float32)
                agg_price_with_dt = np.concatenate([agg_price, dt_feature], axis=0)

                emo_end_idx = curr_idx - self.emotion_lag
                if emo_end_idx < 0:
                    break

                emo_start_idx = max(0, emo_end_idx - step + 1)
                emotion_window = df.loc[emo_start_idx:emo_end_idx]

                emotion_features = []

                is_high_info = 1.0 if bool(df.loc[curr_idx, "is_high_info"]) else 0.0
                is_high_vol = 1.0 if bool(df.loc[curr_idx, "is_high_vol"]) else 0.0

                for col in self.emotion_base_cols:
                    mean_v = float(emotion_window[col].mean())
                    last_v = float(emotion_window[col].iloc[-1])

                    if curr_idx > 0:
                        delta_1h = float(df.loc[curr_idx, col] - df.loc[curr_idx - 1, col])
                    else:
                        delta_1h = 0.0

                    hist24_start = max(0, curr_idx - 23)
                    hist24 = df.loc[hist24_start:curr_idx, col].astype(float)
                    mean_24 = float(hist24.mean())
                    std_24 = float(hist24.std(ddof=0)) if len(hist24) > 1 else 0.0
                    zscore_24h = (float(df.loc[curr_idx, col]) - mean_24) / (std_24 + 1e-8)

                    emotion_features.extend([mean_v, last_v, delta_1h, zscore_24h])

                emotion_features.extend([is_high_info, is_high_vol])
                agg_emotion = np.array(emotion_features, dtype=np.float32)

                seq_price.append(agg_price_with_dt)
                seq_emotion.append(agg_emotion)

                curr_idx -= step

            if len(seq_price) == self.seq_len:
                seq_price.reverse()
                seq_emotion.reverse()

                self.samples_price.append(np.array(seq_price, dtype=np.float32))
                self.samples_emotion.append(np.array(seq_emotion, dtype=np.float32))
                self.labels.append(float(df.loc[idx, target_col]))
                self.sample_indices.append(int(idx))
                self.sample_close.append(float(df.loc[idx, "close"]))

                if "date_hour" in df.columns:
                    self.sample_dates.append(str(df.loc[idx, "date_hour"]))
                elif "timestamp" in df.columns:
                    self.sample_dates.append(str(df.loc[idx, "timestamp"]))
                else:
                    self.sample_dates.append(str(idx))

        self.samples_price = np.array(self.samples_price, dtype=np.float32)
        self.samples_emotion = np.array(self.samples_emotion, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.sample_indices = np.array(self.sample_indices, dtype=np.int64)
        self.sample_close = np.array(self.sample_close, dtype=np.float32)
        self.sample_dates = np.array(self.sample_dates)

        print(f"样本构建完成，共 {len(self.labels)} 条有效样本")

    def apply_scalers(self, price_scaler: StandardScaler, emotion_scaler: StandardScaler):
        if len(self.samples_price) == 0:
            return

        n, t, p = self.samples_price.shape
        _, _, e = self.samples_emotion.shape

        price_2d = self.samples_price.reshape(-1, p)
        emotion_2d = self.samples_emotion.reshape(-1, e)

        self.samples_price = price_scaler.transform(price_2d).reshape(n, t, p).astype(np.float32)
        self.samples_emotion = emotion_scaler.transform(emotion_2d).reshape(n, t, e).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.samples_price[idx], dtype=torch.float32),
            torch.tensor(self.samples_emotion[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def _ensure_target_4h(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()

    if target_col in df.columns:
        return df

    if target_col == "target_trend_4h":
        if "close" not in df.columns:
            raise ValueError("找不到 close 列，无法自动构造 target_trend_4h")

        df["future_4h_close"] = df["close"].shift(-4)
        df["ret_4h"] = df["future_4h_close"] / df["close"] - 1.0
        df["target_trend_4h"] = (df["future_4h_close"] > df["close"]).astype(int)
        return df

    raise ValueError(f"目标列 {target_col} 不存在，且当前仅支持自动构造 target_trend_4h")


def _build_market_state(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Vol_mean_720"] = df["Vol_t"].rolling(window=720, min_periods=1).mean()
    df["Vol_std_720"] = df["Vol_t"].rolling(window=720, min_periods=1).std().fillna(0.0)

    df["Info_mean_720"] = df["Info_t"].rolling(window=720, min_periods=1).mean()
    df["Info_std_720"] = df["Info_t"].rolling(window=720, min_periods=1).std().fillna(0.0)

    high_vol = df["Vol_t"] > (df["Vol_mean_720"] + df["Vol_std_720"])
    low_vol = df["Vol_t"] < (df["Vol_mean_720"] - df["Vol_std_720"])

    high_info = df["Info_t"] > (df["Info_mean_720"] + df["Info_std_720"])
    low_info = df["Info_t"] < (df["Info_mean_720"] - df["Info_std_720"])

    conditions = [
        high_vol | high_info,
        low_vol & low_info,
    ]
    choices = ["S3", "S1"]
    df["Market_State"] = np.select(conditions, choices, default="S2")

    df["is_high_info"] = high_info.astype(int)
    df["is_high_vol"] = high_vol.astype(int)

    return df


def _required_columns(price_cols, emotion_base_cols, target_col):
    cols = list(price_cols) + list(emotion_base_cols)
    if target_col not in cols:
        cols.append(target_col)
    cols.extend(["Market_State", "is_high_info", "is_high_vol"])
    return cols


def get_adaptive_dataloaders(
    csv_path,
    seq_len=24,
    batch_size=32,
    target_col="target_trend_4h",
    emotion_lag=1,
):
    df = pd.read_csv(csv_path).reset_index(drop=True)
    df = _ensure_target_4h(df, target_col)
    df = _build_market_state(df)

    price_cols = ["open", "high", "low", "close", "volume", "return", "Vol_t"]
    emotion_base_cols = ["Info_t", "FOMO_m", "FUD_m", "Euphoria_m", "Panic_m"]

    required_cols = _required_columns(price_cols, emotion_base_cols, target_col)
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    train_dataset = AdaptiveWindowDataset(
        train_df,
        seq_len=seq_len,
        price_cols=price_cols,
        emotion_base_cols=emotion_base_cols,
        target_col=target_col,
        emotion_lag=emotion_lag,
    )
    val_dataset = AdaptiveWindowDataset(
        val_df,
        seq_len=seq_len,
        price_cols=price_cols,
        emotion_base_cols=emotion_base_cols,
        target_col=target_col,
        emotion_lag=emotion_lag,
    )
    test_dataset = AdaptiveWindowDataset(
        test_df,
        seq_len=seq_len,
        price_cols=price_cols,
        emotion_base_cols=emotion_base_cols,
        target_col=target_col,
        emotion_lag=emotion_lag,
    )

    if len(train_dataset) == 0:
        raise ValueError("训练集未构造出有效样本，请检查 seq_len、target_col 或原始数据长度")

    price_scaler = StandardScaler()
    emotion_scaler = StandardScaler()

    price_scaler.fit(
        train_dataset.samples_price.reshape(-1, train_dataset.samples_price.shape[-1])
    )
    emotion_scaler.fit(
        train_dataset.samples_emotion.reshape(-1, train_dataset.samples_emotion.shape[-1])
    )

    train_dataset.apply_scalers(price_scaler, emotion_scaler)
    val_dataset.apply_scalers(price_scaler, emotion_scaler)
    test_dataset.apply_scalers(price_scaler, emotion_scaler)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    price_dim = train_dataset.samples_price.shape[-1]
    emotion_dim = train_dataset.samples_emotion.shape[-1]

    return train_loader, val_loader, test_loader, price_dim, emotion_dim