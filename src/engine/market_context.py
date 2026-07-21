import polars as pl

def calculate_market_relative_strength(
    stock_daily: pl.DataFrame, 
    index_daily: pl.DataFrame
) -> pl.DataFrame:
    """
    Layer 4.2: 个股对大盘的相对强弱因子 (RS5, RS20)。
    """
    # 规范大盘列名
    idx_df = index_daily.select([
        pl.col("date"),
        pl.col("close").alias("idx_close")
    ])
    
    merged = stock_daily.join(idx_df, on="date", how="left")
    merged = merged.sort("date")

    # 计算 5 日与 20 日收益率
    merged = merged.with_columns([
        (pl.col("close") / pl.col("close").shift(5) - 1.0).alias("stock_ret_5"),
        (pl.col("close") / pl.col("close").shift(20) - 1.0).alias("stock_ret_20"),
        (pl.col("idx_close") / pl.col("idx_close").shift(5) - 1.0).alias("idx_ret_5"),
        (pl.col("idx_close") / pl.col("idx_close").shift(20) - 1.0).alias("idx_ret_20")
    ])

    # 相对超额收益 (RS)
    merged = merged.with_columns([
        (pl.col("stock_ret_5") - pl.col("idx_ret_5")).fill_null(0.0).alias("rs_5"),
        (pl.col("stock_ret_20") - pl.col("idx_ret_20")).fill_null(0.0).alias("rs_20")
    ])

    return merged.select(["date", "rs_5", "rs_20"])
