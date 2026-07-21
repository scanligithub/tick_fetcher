import polars as pl

def calculate_price_context(daily_df: pl.DataFrame) -> pl.DataFrame:
    """
    Layer 4.1: 个股日线级别价格位置上下文计算。
    """
    # 期望输入为某只股票历史日K DataFrame：[date, open, high, low, close, volume]
    sorted_df = daily_df.sort("date")

    # 滚动窗口计算
    sorted_df = sorted_df.with_columns([
        pl.col("close").rolling_min(window_size=20).alias("low_20"),
        pl.col("close").rolling_max(window_size=20).alias("high_20"),
        pl.col("close").rolling_min(window_size=60).alias("low_60"),
        pl.col("close").rolling_max(window_size=60).alias("high_60"),
        pl.col("close").rolling_mean(window_size=20).alias("ma20"),
        # ATR 10 计算近似值
        (pl.col("high") - pl.col("low")).rolling_mean(window_size=10).alias("atr_10")
    ])

    # 价格位置计算（PP）
    sorted_df = sorted_df.with_columns([
        ((pl.col("close") - pl.col("low_20")) / (pl.col("high_20") - pl.col("low_20") + 1e-5)).alias("pp_20"),
        ((pl.col("close") - pl.col("low_60")) / (pl.col("high_60") - pl.col("low_60") + 1e-5)).alias("pp_60"),
        ((pl.col("close") / (pl.col("ma20") + 1e-5)) - 1.0).alias("bias_20")
    ])

    return sorted_df.select(["date", "pp_20", "pp_60", "bias_20", "atr_10"])
