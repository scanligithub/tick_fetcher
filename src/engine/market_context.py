import polars as pl

def calculate_market_relative_strength(kline_csv_path: str, index_csv_path: str) -> pl.DataFrame:
    """计算个股相对大盘(上证指数)的真实 5 日超额收益率 (RS_5)"""
    try:
        # 强制声明 Dtypes
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
        # 1. 股票分组使用 .last() 是完全合法的
        stock_df = stock_df.sort(["code", "date"]).with_columns([
            (pl.col("close") / pl.col("close").shift(5).over("code") - 1.0).alias("stock_ret_5")
        ]).group_by("code").last().select(["code", "stock_ret_5"])

        # 2. 🚀 核心修复：普通 DataFrame 没有 .last() 方法，修正为标准的 .tail(1)
        index_df = index_df.sort("date").with_columns([
            (pl.col("close") / pl.col("close").shift(5) - 1.0).alias("idx_ret_5")
        ]).tail(1)
        
        idx_ret = float(index_df["idx_ret_5"][0]) if not index_df.is_empty() and index_df["idx_ret_5"][0] is not None else 0.0

        res = stock_df.with_columns([
            (pl.col("stock_ret_5") - idx_ret).fill_null(0.0).alias("rs_5")
        ])
        return res.select(["code", "rs_5"])
    except Exception as e:
        print(f"❌ [RS Context] 相对强弱计算过程异常: {e}", flush=True)
        return pl.DataFrame()
