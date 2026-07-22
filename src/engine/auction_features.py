import polars as pl
import math

def extract_auction_features(ticks_df: pl.DataFrame, prev_close: float) -> dict:
    """
    重构后的 V2.2 集合竞价特征提取算子：
    引入价格比例尺归一化与观测质量平滑，实现无量纲、有界 [0, 1] 的集合竞价稳定性指标。
    """
    # 提取 09:15:00 至 09:25:00 的集合竞价 ticks
    auction_df = ticks_df.filter((pl.col("time") >= "09:15:00") & (pl.col("time") <= "09:25:00"))
    
    if auction_df.is_empty() or prev_close <= 0.0:
        return {
            "auction_volume": 0.0, 
            "auction_gap": 0.0, 
            "auction_strength": 0.0, 
            "auction_tick_count": 0,
            "auction_range": 0.0,
            "auction_obs_qty": 0.0,
            "auction_stab_obs": 0.0,
            "auction_stability": 0.0
        }
    
    auction_volume = float(auction_df["volume"].sum())
    open_price = float(auction_df["price"].last())
    auction_gap = (open_price / prev_close - 1.0)
    
    # 提取中间物理特征
    auction_tick_count = int(len(auction_df))
    prices = auction_df["price"].to_list()
    auction_high = float(max(prices))
    auction_low = float(min(prices))
    auction_range = auction_high - auction_low
    
    # 1. 测度 1: 竞价观测质量（不满足5笔Tick则衰减权重，防止无观测造成的伪高稳定性）
    auction_obs_qty = min(float(auction_tick_count) / 5.0, 1.0)
    
    # 2. 测度 2: 观测价格偏离稳定性 (以昨收价的 0.5% 作为波动率比例尺基准)
    scale = prev_close * 0.005
    auction_stab_obs = math.exp(-auction_range / (scale + 1e-8))
    
    # 3. 最终组合：有效竞价稳定性
    auction_stability = auction_stab_obs * auction_obs_qty
    auction_strength = auction_volume * abs(auction_gap)
    
    return {
        "auction_volume": auction_volume, 
        "auction_gap": auction_gap,
        "auction_strength": auction_strength, 
        "auction_tick_count": auction_tick_count,
        "auction_range": auction_range,
        "auction_obs_qty": auction_obs_qty,
        "auction_stab_obs": auction_stab_obs,
        "auction_stability": auction_stability
    }
