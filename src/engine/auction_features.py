import polars as pl

def extract_auction_features(ticks_df: pl.DataFrame, prev_close: float) -> dict:
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
    open_price = float(auction_df["price"].last())
    auction_gap = (open_price / prev_close - 1.0) if prev_close > 0 else 0.0
    price_std = float(auction_df["price"].std()) if len(auction_df) > 1 else 0.0
    auction_stability = 1.0 / (price_std + 1e-5)
    auction_strength = auction_volume * abs(auction_gap)
    return {
        "auction_volume": auction_volume,
        "auction_gap": auction_gap,
        "auction_strength": auction_strength,
        "auction_stability": auction_stability
    }
