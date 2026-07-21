import polars as pl

def calculate_limit_regime(ticks_df, prev_close, limit_up_pct=0.099, limit_down_pct=-0.099):
    if ticks_df.is_empty() or prev_close <= 0:
        return {"limit_state": 0, "price_response_validity": 1.0}
        
    limit_up_price = round(prev_close * (1.0 + limit_up_pct), 2)
    limit_down_price = round(prev_close * (1.0 + limit_down_pct), 2)
    max_price = ticks_df["price"].max()
    min_price = ticks_df["price"].min()
    limit_state = 0
    price_response_validity = 1.0
    
    if max_price >= limit_up_price:
        limit_state = 1
        up_ratio = ticks_df.filter(pl.col("price") >= limit_up_price)["volume"].sum() / (ticks_df["volume"].sum() + 1e-8)
        price_response_validity = max(0.1, 1.0 - up_ratio)
    elif min_price <= limit_down_price:
        limit_state = 2
        down_ratio = ticks_df.filter(pl.col("price") <= limit_down_price)["volume"].sum() / (ticks_df["volume"].sum() + 1e-8)
        price_response_validity = max(0.1, 1.0 - down_ratio)
        
    return {"limit_state": limit_state, "price_response_validity": price_response_validity}
