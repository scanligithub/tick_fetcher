import polars as pl

def aggregate_to_daily_row(
    aligned_min_df: pl.DataFrame, 
    limit_dict: dict, 
    auction_dict: dict,
    dq_weights: dict
) -> pl.DataFrame:
    """
    完成 240 分钟标准化微窗指标向单一交易日宽表的降维合并与数据质量评估。
    """
    if aligned_min_df.is_empty():
        return pl.DataFrame()

    code = aligned_min_df["code"][0]
    date = aligned_min_df["trade_date"][0]

    # 日级基础量额
    daily_vol = aligned_min_df["volume_sum"].sum()
    daily_amt = aligned_min_df["volume_sum"].sum() * aligned_min_df["vwap"].mean() # 近似额
    daily_ticks = aligned_min_df["trade_count"].sum()

    buy_vol_sum = aligned_min_df["buy_vol"].sum()
    sell_vol_sum = aligned_min_df["sell_vol"].sum()

    # 统计偏度与分位数
    imbalance_mean = aligned_min_df["imbalance"].mean()
    imbalance_std = aligned_min_df["imbalance"].std()
    
    # 极值过滤
    p10_val = aligned_min_df["imbalance"].quantile(0.1)
    p90_val = aligned_min_df["imbalance"].quantile(0.9)

    # 持续性分钟计算
    buy_active_mins = aligned_min_df.filter(pl.col("imbalance") > 0.3).len()
    sell_active_mins = aligned_min_df.filter(pl.col("imbalance") < -0.3).len()

    # 数据质量构成计算 (Layer 0)
    observed_ratio = aligned_min_df.filter(pl.col("has_trade")).len() / 240.0
    empty_ratio = aligned_min_df.filter(pl.col("is_empty_bin")).len() / 240.0
    dir_coverage = (buy_vol_sum + sell_vol_sum) / (daily_vol + 1e-8)

    dq_score = (
        dq_weights.get("observed_ratio", 0.4) * observed_ratio +
        dq_weights.get("coverage_ratio", 0.3) * dir_coverage +
        dq_weights.get("non_empty_ratio", 0.3) * (1.0 - empty_ratio)
    )

    # 价格推动响应响应度
    price_response = abs(aligned_min_df["price_return"].sum())
    
    # 连续型吸收微观事实计算
    # 吸收 = 攻击强度 * (1 - 价格推动效率) * 行为持续性比例
    buy_aggression = buy_vol_sum / (daily_vol + 1e-8)
    sell_aggression = sell_vol_sum / (daily_vol + 1e-8)
    
    buy_absorption_raw = buy_aggression * (1.0 - min(price_response, 1.0)) * (buy_active_mins / 240.0)
    sell_absorption_raw = sell_aggression * (1.0 - min(price_response, 1.0)) * (sell_active_mins / 240.0)

    # 整合单行结果
    row = pl.DataFrame({
        "code": [code],
        "date": [date],
        "total_volume": [daily_vol],
        "total_amount": [daily_amt],
        "tick_count": [daily_ticks],
        "buy_volume_sum": [buy_vol_sum],
        "sell_volume_sum": [sell_vol_sum],
        "buy_aggression": [buy_aggression],
        "sell_aggression": [sell_aggression],
        "imbalance_mean": [imbalance_mean],
        "imbalance_std": [imbalance_std],
        "imbalance_p10": [p10_val],
        "imbalance_p90": [p90_val],
        "buy_active_minutes": [buy_active_mins],
        "sell_active_minutes": [sell_active_mins],
        "price_response": [price_response],
        "buy_absorption_raw": [buy_absorption_raw * limit_dict["price_response_validity"]],
        "sell_absorption_raw": [sell_absorption_raw * limit_dict["price_response_validity"]],
        "observed_minute_ratio": [observed_ratio],
        "empty_minute_ratio": [empty_ratio],
        "direction_coverage_ratio": [dir_coverage],
        "data_quality_score": [dq_score],
        **auction_dict,
        **limit_dict
    })
    return row
