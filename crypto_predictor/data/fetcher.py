import time
import os
import pandas as pd
import ccxt

from utils.logger import get_logger

logger = get_logger(__name__)


def build_exchange(
    api_key: str,
    secret: str,
    password: str,
    proxy: str = None,
) -> ccxt.okx:
    config = {
        "apiKey": api_key,
        "secret": secret,
        "password": password,
    }
    if proxy:
        config["proxies"] = {"http": proxy, "https": proxy}

    return ccxt.okx(config)


def fetch_ohlcv_hourly(
    exchange: ccxt.okx,
    symbol: str,
    timeframe: str = "1h",
    start: str = "2021-01-01T00:00:00Z",
    end: str = "2023-01-01T00:00:00Z",
    limit_per_request: int = 100,
    sleep_seconds: float = 0.3,
    retry_seconds: float = 5.0,
) -> pd.DataFrame:
    since = exchange.parse8601(start)
    until = exchange.parse8601(end)
    all_data = []

    logger.info(f"开始拉取 {symbol} 数据，时间范围: {start} ~ {end}")

    while since < until:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe,
                since,
                limit=limit_per_request,
            )
            if not ohlcv:
                logger.warning(f"{symbol}: 返回空数据，停止拉取")
                break

            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(sleep_seconds)

        except Exception as e:
            logger.error(f"{symbol} 拉取失败: {e}，{retry_seconds}s 后重试")
            time.sleep(retry_seconds)

    df = pd.DataFrame(
        all_data,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["date_hour"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime(
        "%Y-%m-%d %H:00:00"
    )
    df = df.drop_duplicates(subset="date_hour").reset_index(drop=True)

    if not df.empty:
        logger.info(
            f"{symbol}: 拉取完成，共 {len(df)} 条，"
            f"范围 {df['date_hour'].iloc[0]} ~ {df['date_hour'].iloc[-1]}"
        )
    else:
        logger.info(f"{symbol}: 拉取完成，但结果为空")

    return df


def fetch_all_symbols(
    exchange: ccxt.okx,
    symbols: list = None,
    save_dir: str = ".",
    **kwargs,
) -> dict:
    os.makedirs(save_dir, exist_ok=True)

    if symbols is None:
        symbols = [
            ("BTC/USDT", "btc"),
            ("ETH/USDT", "eth"),
            ("DOGE/USDT", "doge"),
        ]

    results = {}
    for symbol, name in symbols:
        df = fetch_ohlcv_hourly(exchange, symbol, **kwargs)
        save_path = os.path.join(save_dir, f"{name}_hourly.csv")
        df.to_csv(save_path, index=False)
        logger.info(f"已保存: {save_path}")
        results[name] = df

    return results