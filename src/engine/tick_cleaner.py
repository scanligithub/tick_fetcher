import polars as pl

def clean_raw_ticks(file_path: str) -> pl.DataFrame:
    """
    Layer 0 清洗层：
    加载原始 CSV 格式 Tick 数据，执行统一格式转换与基础数据过滤。
    """
    df = pl.read_csv(
        file_path,
        dtypes={
            "code": pl.String,
            "date": pl.String,
            "time": pl.String,
            "price": pl.Float64,
            "volume": pl.Int64,
            "status": pl.Int8,
            "number": pl.Int32
        }
    )
    
    # 物理去噪：过滤异常成交量与价格
    df = df.filter(
        (pl.col("price") > 0) & 
        (pl.col("volume") >= 0) & 
        (pl.col("time").str.len_chars() == 8)
    )
    
    # 统一映射成交方向：0=主买, 1=主卖, 2=中性
    df = df.with_columns(
        pl.when(pl.col("status") == 0).then(pl.lit(0))
        .when(pl.col("status") == 1).then(pl.lit(1))
        .otherwise(pl.lit(2))
        .alias("direction")
    )
    return df
