# FILE: src/engine/price_context.py
import polars as pl

def calculate_price_context(kline_csv_path: str) -> pl.DataFrame:
    """
    读取日线K线，计算严格的 T-1 历史上下文指标（通过 .shift(1) 物理隔离避免同日污染）。
    """
    try:
        # 使用 Polars 推荐的 schema_overrides 规范
        df = pl.read_csv(
            kline_csv_path,
            schema_overrides={
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

        # 1. 在 K 线序列上计算无除权除息污染的真实日收益率 (不 shift，最后一行即为今日 T 日的 K线收益率)
        df = df.with_columns([
            ((pl.col("close") / pl.col("close").shift(1)) - 1.0).alias("daily_return")
        ])

        # 2. 计算日线级基础指标 (V2.2 严格语义：低点算 rolling_min, 高点算 rolling_max)
        df = df.with_columns([
            pl.col("volume").rolling_mean(window_size=20).over("code").alias("adv_20"),
            (pl.col("high") - pl.col("low")).rolling_mean(window_size=10).over("code").alias("atr_10"),
            pl.col("low").rolling_min(window_size=20).over("code").alias("low_20"),
            pl.col("high").rolling_max(window_size=20).over("code").alias("high_20"),
            pl.col("low").rolling_min(window_size=60).over("code").alias("low_60"),
            pl.col("high").rolling_max(window_size=60).over("code").alias("high_60"),
            pl.col("close").rolling_mean(window_size=20).over("code").alias("ma20")
        ])

        # 3. 关键设计：通过 .shift(1) 提取严格昨收历史特征，切断与T日成交信息的物理关联
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

        # 4. 派生昨收位置偏离特征（均为 T-1 静态历史状态）
        df = df.with_columns([
            ((pl.col("close_pre") - pl.col("low_20_pre")) / (pl.col("high_20_pre") - pl.col("low_20_pre") + 1e-8))
            .clip(0.0, 1.0).alias("pp_20_pre"),
            ((pl.col("close_pre") - pl.col("low_60_pre")) / (pl.col("high_60_pre") - pl.col("low_60_pre") + 1e-8))
            .clip(0.0, 1.0).alias("pp_60_pre"),
            ((pl.col("close_pre") / (pl.col("ma20_pre") + 1e-8)) - 1.0).alias("bias_20_pre")
        ])

        # 5. V2.2-Final 核心设计：在 K 线空间直接固化波动率下限 (0.2%) 与有效区间波动下限 (0.5%)，消除下游分母坍缩
        df = df.with_columns([
            # 波动率比例尺下限 (ATR_PCT Floor = 0.2%)
            pl.when((pl.col("atr_10_pre") / (pl.col("close_pre") + 1e-8)) > 0.002)
            .then(pl.col("atr_10_pre") / (pl.col("close_pre") + 1e-8))
            .otherwise(0.002)
            .alias("atr_pct_10_pre"),
            
            # 最低价格有效波动分母下限 (RangeFloor = 0.5% × 昨收)
            pl.when((pl.col("high_60_pre") - pl.col("low_60_pre")) > (pl.col("close_pre") * 0.005))
            .then(pl.col("high_60_pre") - pl.col("low_60_pre"))
            .otherwise(pl.col("close_pre") * 0.005)
            .alias("effective_range_60_pre")
        ])

        # 6. 提取序列的最后一个交易日（今日 T 日，带入已经 shift 的 T-1 特征和今日 K线真实收益率）
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
            "bias_20_pre",
            "daily_return",            # 本日 K 线收益率 (T日)
            "atr_pct_10_pre",          # CST 保护后的 ATR 相对波动率下限 (T-1)
            "effective_range_60_pre"   # CST 保护后的 60日波动区间下限 (T-1)
        ])
        return latest_context
    except Exception as e:
        print(f"❌ [Price Context] 滚动算子推导崩溃: {e}", flush=True)
        return pl.DataFrame()
