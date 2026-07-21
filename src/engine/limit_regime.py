import polars as pl

def calculate_limit_regime(
    ticks_df: pl.DataFrame, 
    prev_close: float, 
    limit_up_pct: float = 0.099, 
    limit_down_pct: float = -0.099
) -> dict:
    """
    Layer 3: 限制区探测与降权因子计算。
    """
    if ticks_df.is_empty() or prev_close <= 0:
        return {"limit_state": 0, "price_response_validity": 1.0}

    limit_up_price = round(prev_close * (1.0 + limit_up_pct), 2)
    limit_down_price = round(prev_close * (1.0 + limit_down_pct), 2)

    # 取全天极值价格
    max_price = ticks_df["price"].max()
    min_price = ticks_df["price"].min()

    limit_state = 0 # 0=NORMAL, 1=LIMIT_UP, 2=LIMIT_DOWN
    price_response_validity = 1.0

    if max_price >= limit_up_price:
        limit_state = 1
        # 计算封板成交量占比，作为降权参考
        up_vol = ticks_df.filter(pl.col("price") >= limit_up_price)["volume"].sum()
        total_vol = ticks_df["volume"].sum()
        up_ratio = up_vol / (total_vol + 1e-8)
        price_response_validity = max(0.1, 1.0 - up_ratio) # 封板量越多，价格弹性越差，权数趋向 0.1
        
    elif min_price <= limit_down_price:
        limit_state = 2
        down_vol = ticks_df.filter(pl.col("price") <= limit_down_price)["volume"].sum()
        total_vol = ticks_df["volume"].sum()
        down_ratio = down_vol / (total_vol + 1e-8)
        price_response_validity = max(0.1, 1.0 - down_ratio)

    return {
        "limit_state": limit_state,
        "price_response_validity": price_response_validity
    }
