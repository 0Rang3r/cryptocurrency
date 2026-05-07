import os
import re
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

from utils.logger import get_logger

logger = get_logger(__name__)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

EMOTION_KEYWORDS = {
    0: ["fomo", "buy the dip", "pump", "all in", "bullish", "accumulation"],
    1: ["fud", "scam", "dump", "bearish", "bubble", "fake", "ponzi"],
    2: ["to the moon", "ath", "rich", "millionaire", "hype", "x100"],
    3: ["rekt", "liquidated", "crash", "dead", "panic", "bankrupt"],
}

EMOTION_LABEL_NAMES = ["FOMO", "FUD", "Euphoria", "Panic"]


def rule_based_labeling(text: str) -> int:
    text = str(text).lower()
    for label, keywords in EMOTION_KEYWORDS.items():
        for word in keywords:
            if re.search(r"\b" + re.escape(word) + r"\b", text):
                return label
    return -1


class EmotionClassifier:
    MODEL_NAME = "ProsusAI/finbert"
    NUM_LABELS = 4

    def __init__(self, hf_token: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = hf_token
        self.tokenizer = None
        self.model = None

    def load_pretrained(self) -> None:
        logger.info(f"加载预训练模型: {self.MODEL_NAME}")
        kwargs = {"token": self.hf_token} if self.hf_token else {}

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=self.NUM_LABELS,
            ignore_mismatched_sizes=True,
            **kwargs,
        )

    def fine_tune(
        self,
        df_tweets: pd.DataFrame,
        text_col: str = "Content",
        output_dir: str = "./emotion_model",
        epochs: int = 2,
    ) -> None:
        if self.model is None:
            self.load_pretrained()

        logger.info("开始生成伪标签")
        df_tweets = df_tweets.copy()
        df_tweets["pseudo_label"] = df_tweets[text_col].apply(rule_based_labeling)

        train_df = df_tweets[df_tweets["pseudo_label"] != -1].copy()
        logger.info(f"有效伪标签样本数: {len(train_df)}")

        if len(train_df) == 0:
            logger.warning("没有可用伪标签样本，跳过微调，直接使用预训练模型")
            return

        hf_dataset = Dataset.from_pandas(
            train_df[[text_col, "pseudo_label"]].rename(columns={"pseudo_label": "label"})
        )

        def tokenize(examples):
            return self.tokenizer(
                examples[text_col],
                padding="max_length",
                truncation=True,
                max_length=128,
            )

        tokenized = hf_dataset.map(tokenize, batched=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            logging_steps=50,
            save_strategy="no",
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized,
        )

        logger.info("开始微调情绪分类模型")
        trainer.train()
        logger.info("微调结束")

    def predict_batch(self, texts: list, batch_size: int = 128) -> np.ndarray:
        assert self.model is not None, "请先调用 load_pretrained() 或 fine_tune()"

        self.model.to(self.device)
        self.model.eval()

        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
                all_probs.extend(probs)

        return np.array(all_probs)


def extract_hourly_emotions(
    df_tweets: pd.DataFrame,
    classifier: EmotionClassifier,
    text_col: str = "Content",
    date_col: str = "Date",
) -> pd.DataFrame:
    logger.info("开始情绪推理")

    df = df_tweets.copy()
    df["date_hour"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime(
        "%Y-%m-%d %H:00:00"
    )
    df = df.dropna(subset=["date_hour", text_col]).reset_index(drop=True)

    texts = df[text_col].astype(str).tolist()
    probs = classifier.predict_batch(texts)

    prob_df = pd.DataFrame(
        probs,
        columns=["FOMO_prob", "FUD_prob", "Euphoria_prob", "Panic_prob"],
    )
    df = pd.concat([df, prob_df], axis=1)

    logger.info("按小时聚合情绪特征")
    hourly = (
        df.groupby("date_hour")
        .agg(
            Info_t=(text_col, "count"),
            FOMO_m=("FOMO_prob", "mean"),
            FUD_m=("FUD_prob", "mean"),
            Euphoria_m=("Euphoria_prob", "mean"),
            Panic_m=("Panic_prob", "mean"),
        )
        .reset_index()
    )

    logger.info(f"情绪聚合完成，共 {len(hourly)} 条小时记录")
    return hourly


def build_final_dataset(
    price_csv: str,
    twitter_csv: str,
    output_csv: str = "btc_multimodal_hourly_dataset.csv",
    hf_token: str = None,
) -> pd.DataFrame:
    logger.info("开始构建多模态数据集")

    df_price = pd.read_csv(price_csv)
    df_price = df_price.sort_values("date_hour").reset_index(drop=True)
    df_price["return"] = np.log(df_price["close"] / df_price["close"].shift(1))
    df_price["Vol_t"] = df_price["return"].rolling(window=24).std()
    df_price["next_hour_close"] = df_price["close"].shift(-1)
    df_price["target_trend"] = (df_price["next_hour_close"] > df_price["close"]).astype(int)

    df_tweets = pd.read_csv(twitter_csv, on_bad_lines="skip", engine="python")

    classifier = EmotionClassifier(hf_token=hf_token)
    classifier.fine_tune(df_tweets)
    hourly_emotions = extract_hourly_emotions(df_tweets, classifier)

    df_merged = pd.merge(df_price, hourly_emotions, on="date_hour", how="left")
    df_merged["Info_t"] = df_merged["Info_t"].fillna(0)
    df_merged[["FOMO_m", "FUD_m", "Euphoria_m", "Panic_m"]] = df_merged[
        ["FOMO_m", "FUD_m", "Euphoria_m", "Panic_m"]
    ].fillna(0.25)

    df_merged = df_merged.dropna(subset=["Vol_t", "target_trend"]).reset_index(drop=True)
    df_merged.to_csv(output_csv, index=False)

    logger.info(f"数据集构建完成，总行数: {len(df_merged)}，保存路径: {output_csv}")
    logger.info(
        df_merged[
            ["date_hour", "close", "Vol_t", "Info_t", "FOMO_m", "target_trend"]
        ].head().to_string()
    )

    return df_merged


if __name__ == "__main__":
    build_final_dataset(
        price_csv="btc_hourly.csv",
        twitter_csv="dataset/Crypto Tweets/crypto_10k_tweets_(2021_2022Nov).csv",
    )