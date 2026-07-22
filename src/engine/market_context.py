import polars as pl

def calculate_market_relative_strength(kline_csv_path: str, index_csv_path: str) -> pl.DataFrame:
    """
    计算截至 T-1 日，个股相对于大盘的 5 日超额历史收益率 (RS_5_PRE)。
    """
    try:
        kline_dtypes = {
            "code": pl.String, "date": pl.String,
            "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
            "volume": pl.Float64, "amount": pl.Float64
        }
        stock_df = pl.read_csv(kline_csv_path, dtypes=kline_dtypes)
        index_df = pl.read_csv(index_csv_path, dtypes=kline_dtypes)
    except Exception as e:
        print(f"⚠️ [RS Context] 相对强弱计算依赖加载失败: {e}", flush=True)
        return pl.DataFrame()

    if stock_df.is_empty() or index_df.is_empty():
        return pl.DataFrame()

    try:
        # 1. 计算个股 5 日历史回报率，并向后 shift 一日
        stock_df = stock_df.sort(["code", "date"]).with_columns([
            (pl.col("close") / pl.col("close").shift(5).over("code") - 1.0).alias("stock_ret_5")
        ]).with_columns([
            pl.col("stock_ret_5").shift(1).over("code").alias("stock_ret_5_pre")
        ])

        # 2. 计算大盘 5 日历史回报率，并向后 shift 一日
        index_df = index_df.sort("date").with_columns([
            (pl.col("close") / pl.col("close").shift(5) - 1.0).alias("idx_ret_5")
        ]).with_columns([
            pl.col("idx_ret_5").shift(1).alias("idx_ret_5_pre")
        ])

        # 3. 提取大盘最新一行的 T-1 历史指标值
        latest_index = index_df.tail(1)
        if latest_index.is_empty():
            idx_ret_pre = 0.0
        else:
            idx_ret_pre = float(latest_index["idx_ret_5_pre"][0]) if latest_index["idx_ret_5_pre"][0] is not None else 0.0

        # 4. 计算个股截至昨日收盘的超额回报
        res = stock_df.group_by("code").last().with_columns([
            (pl.col("stock_ret_5_pre") - idx_ret_pre).fill_null(0.0).alias("rs_5_pre")
        ])
        
        return res.select(["code", "rs_5_pre"])
    except Exception as e:
        print(f"❌ [RS Context] 相对强弱计算过程异常: {e}", flush=True)
        return pl.DataFrame()
