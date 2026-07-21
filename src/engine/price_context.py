import polars as pl

def calculate_price_context(kline_csv_path: str) -> pl.DataFrame:
    """读取真实的 100 日 K 线 CSV，计算最新的价格位置(PP)、乖离率(Bias)与波动率(ATR)。"""
    try:
        # 🚀 显式声明 Dtypes，强制 Polars 将文本解析为 Float64 双精度浮点数
        df = pl.read_csv(
            kline_csv_path,
            dtypes={
                "code": pl.String,
                "date": pl.String,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "amount": pl.Float64
            }
        )
    except Exception:
        return pl.DataFrame()

    if df.is_empty():
        return pl.DataFrame()

    df = df.sort(["code", "date"])

    df = df.with_columns([
        pl.col("close").rolling_min(window_size=20).over("code").alias("low_20"),
        pl.col("close").rolling_max(window_size=20).over("code").alias("high_20"),
        pl.col("close").rolling_min(window_size=60).over("code").alias("low_60"),
        pl.col("close").rolling_max(window_size=60).over("code").alias("high_60"),
        pl.col("close").rolling_mean(window_size=20).over("code").alias("ma20"),
        (pl.col("high") - pl.col("low")).rolling_mean(window_size=10).over("code").alias("atr_10")
    ])

    df = df.with_columns([
        ((pl.col("close") - pl.col("low_20")) / (pl.col("high_20") - pl.col("low_20") + 1e-5)).alias("pp_20"),
        ((pl.col("close") - pl.col("low_60")) / (pl.col("high_60") - pl.col("low_60") + 1e-5)).alias("pp_60"),
        ((pl.col("close") / (pl.col("ma20") + 1e-5)) - 1.0).alias("bias_20")
    ])

    latest_context = df.group_by("code").last().select([
        "code", "pp_20", "pp_60", "bias_20", "atr_10"
    ])
    return latest_context
