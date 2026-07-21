import polars as pl

def extract_auction_features(ticks_df: pl.DataFrame, prev_close: float) -> dict:
    """
    Layer 2: 09:15-09:25 集合竞价特征聚合器。
    """
    auction_df = ticks_df.filter(
        (pl.col("time") >= "09:15:00") & (pl.col("time") <= "09:25:00")
    )
    
    if auction_df.is_empty():
        return {
            "auction_volume": 0.0,
            "auction_gap": 0.0,
            "auction_strength": 0.0,
            "auction_stability": 1.0
        }

    auction_volume = float(auction_df["volume"].sum())
    open_price = float(auction_df["price"].last()) # 09:25分确定开盘价
    
    auction_gap = (open_price / prev_close - 1.0) if prev_close > 0 else 0.0
    
    # 价格变动标准差作为稳定性表征
    price_std = float(auction_df["price"].std()) if len(auction_df) > 1 else 0.0
    auction_stability = 1.0 / (price_std + 1e-5)

    # 竞价能量系数 (成交总量 * 价格跳空幅度)
    auction_strength = auction_volume * abs(auction_gap)

    return {
        "auction_volume": auction_volume,
        "auction_gap": auction_gap,
        "auction_strength": auction_strength,
        "auction_stability": auction_stability
    }
