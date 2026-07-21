import polars as pl

def calculate_price_context(kline_csv_path: str) -> pl.DataFrame:
    """读取真实的 100 日 K 线 CSV，计算最新的价格位置(PP)、乖离率(Bias)与波动率(ATR)。"""
    try:
        df = pl.read_csv(kline_csv_path, dtypes={"code": pl.String, "date": pl.String})
    except Exception:
        return pl.DataFrame()

    if df.is_empty():
        return pl.DataFrame()

    # 确保按日期升序，方便滚动计算
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

    # 取每只股票最新的一天（即本次计算目标日）的截面数据
    latest_context = df.group_by("code").last().select([
        "code", "pp_20", "pp_60", "bias_20", "atr_10"
    ])
    return latest_context
