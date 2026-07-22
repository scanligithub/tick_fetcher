import polars as pl

def calculate_price_context(kline_csv_path: str) -> pl.DataFrame:
    """
    读取日线K线，计算严格的 T-1 历史上下文指标（通过 .shift(1) 物理隔离避免同日污染）。
    """
    try:
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
    except Exception as e:
        print(f"⚠️ [Price Context] CSV 读取失败 ({kline_csv_path}): {e}", flush=True)
        return pl.DataFrame()

    if df.is_empty():
        print(f"⚠️ [Price Context] CSV 文件为空: {kline_csv_path}", flush=True)
        return pl.DataFrame()

    try:
        # 按股票和日期排序，确保滚动计算的时序正确
        df = df.sort(["code", "date"])

        # 1. 计算日线级基础指标
        df = df.with_columns([
            pl.col("volume").rolling_mean(window_size=20).over("code").alias("adv_20"),
            (pl.col("high") - pl.col("low")).rolling_mean(window_size=10).over("code").alias("atr_10"),
            pl.col("close").rolling_min(window_size=20).over("code").alias("low_20"),
            pl.col("close").rolling_max(window_size=20).over("code").alias("high_20"),
            pl.col("close").rolling_min(window_size=60).over("code").alias("low_60"),
            pl.col("close").rolling_max(window_size=60).over("code").alias("high_60"),
            pl.col("close").rolling_mean(window_size=20).over("code").alias("ma20")
        ])

        # 2. 关键设计：通过 .shift(1) 提取严格昨收历史特征，切断与T日成交信息的物理关联
        df = df.with_columns([
            pl.col("adv_20").shift(1).over("code").alias("adv_20_pre"),
            pl.col("atr_10").shift(1).over("code").alias("atr_10_pre"),
            pl.col("low_20").shift(1).over("code").alias("low_20_pre"),
            pl.col("high_20").shift(1).over("code").alias("high_20_pre"),
            pl.col("low_60").shift(1).over("code").alias("low_60_pre"),
            pl.col("high_60").shift(1).over("code").alias("high_60_pre"),
            pl.col("close").shift(1).over("code").alias("close_pre"),
            pl.col("ma20").shift(1).over("code").alias("ma20_pre")
        ])

        # 3. 派生昨收位置偏离特征（均为 T-1 静态历史状态）
        df = df.with_columns([
            ((pl.col("close_pre") - pl.col("low_20_pre")) / (pl.col("high_20_pre") - pl.col("low_20_pre") + 1e-8))
            .clip(0.0, 1.0).alias("pp_20_pre"),
            ((pl.col("close_pre") - pl.col("low_60_pre")) / (pl.col("high_60_pre") - pl.col("low_60_pre") + 1e-8))
            .clip(0.0, 1.0).alias("pp_60_pre"),
            ((pl.col("close_pre") / (pl.col("ma20_pre") + 1e-8)) - 1.0).alias("bias_20_pre")
        ])

        # 4. 提取序列的最后一个交易日（即T日结算行，但其携带的均是经过 shift 的前一日历史特征）
        latest_context = df.group_by("code").last().select([
            "code", 
            "adv_20_pre", 
            "atr_10_pre", 
            "low_20_pre",
            "high_20_pre",
            "low_60_pre", 
            "high_60_pre", 
            "close_pre",
            "pp_20_pre", 
            "pp_60_pre", 
            "bias_20_pre"
        ])
        return latest_context
    except Exception as e:
        print(f"❌ [Price Context] 滚动算子推导崩溃: {e}", flush=True)
        return pl.DataFrame()
