import polars as pl
import math

def aggregate_to_daily_row(aligned_min_df, limit_dict, auction_dict, dq_weights, history_ctx: dict):
    """
    日级微观行为物理特征重构算子 (V2.2)：
    结合历史上下文，推导标准化的努力-结果背离因子、双向突破特征，并同时输出 V1/V2 版本的吸收指标。
    """
    if aligned_min_df.is_empty():
        return pl.DataFrame()
        
    code = aligned_min_df["code"][0]
    date = aligned_min_df["trade_date"][0]
    
    # 获取 T-1 日的历史静态上下文特征
    atr_10_pre = float(history_ctx.get("atr_10_pre", 0.1))
    adv_20_pre = float(history_ctx.get("adv_20_pre", 1000000.0))
    low_60_pre = float(history_ctx.get("low_60_pre", 1.0))
    high_60_pre = float(history_ctx.get("high_60_pre", 1.0))
    
    # 1. 基础物理事实统计
    daily_vol = float(aligned_min_df["volume_sum"].sum())
    daily_amt = float(aligned_min_df["volume_sum"].sum() * aligned_min_df["vwap"].mean())
    daily_ticks = int(aligned_min_df["trade_count"].sum())
    buy_vol_sum = float(aligned_min_df["buy_vol"].sum())
    sell_vol_sum = float(aligned_min_df["sell_vol"].sum())
    
    # 2. 持续性特征 (以标准化 240 分钟时间网格为分母)
    buy_active_mins = int(aligned_min_df.filter(pl.col("imbalance") > 0.3).height)
    sell_active_mins = int(aligned_min_df.filter(pl.col("imbalance") < -0.3).height)
    buy_persistence = buy_active_mins / 240.0
    sell_persistence = sell_active_mins / 240.0
    
    # 3. 数据质量评估
    observed_ratio = aligned_min_df.filter(pl.col("has_trade")).height / 240.0
    empty_ratio = aligned_min_df.filter(pl.col("is_empty_bin")).height / 240.0
    dir_coverage = (buy_vol_sum + sell_vol_sum) / (daily_vol + 1e-8)
    dq_score = (
        dq_weights.get("observed_ratio", 0.4) * observed_ratio +
        dq_weights.get("coverage_ratio", 0.3) * dir_coverage +
        dq_weights.get("non_empty_ratio", 0.3) * (1.0 - empty_ratio)
    )
    
    # 4. 单位对齐的价格响应标准化与响应因子计算 (T日 Close 相对于昨收)
    close_t = float(aligned_min_df["close"].last())
    prev_close = float(history_ctx.get("close_pre", close_t))
    if prev_close <= 0:
        prev_close = close_t
        
    price_return_t = (close_t / prev_close - 1.0)
    
    # 将绝对价差 ATR 转化为百分比波动率尺度
    atr_pct_10 = atr_10_pre / (prev_close + 1e-8)
    
    # 日价格响应归一化
    price_response_norm = abs(price_return_t) / (atr_pct_10 + 1e-8)
    # 利用指数衰减公式计算价格响应因子，实现高价格变动对吸收强度的非线性惩罚
    response_factor = math.exp(-price_response_norm)
    
    # 5. 绝对成交努力度因子 (Effort Factor)
    effort_factor = min(daily_vol / (adv_20_pre + 1e-8), 3.0) / 3.0
    
    # 6. 方向偏好 (Aggression)
    buy_aggression = buy_vol_sum / (daily_vol + 1e-8)
    sell_aggression = sell_vol_sum / (daily_vol + 1e-8)
    
    # 7. 计算 V1 与 V2 吸收因子
    # V1: 经典版 (Aggression * Response_Factor * Persistence)
    buy_absorption_v1 = buy_aggression * response_factor * buy_persistence
    sell_absorption_v1 = sell_aggression * response_factor * sell_persistence
    
    # V2: 重构版 (V1 * Effort_Factor * DataQuality_Score)
    buy_absorption_v2 = buy_absorption_v1 * effort_factor * dq_score
    sell_absorption_v2 = sell_absorption_v1 * effort_factor * dq_score
    
    # 8. 价格区间穿透与双向突破特征计算 (均排除T日数据对最高最低区间的污染)
    pp_60_close = (close_t - low_60_pre) / (high_60_pre - low_60_pre + 1e-8)
    
    # 向上突破
    breakout_60 = (close_t / (high_60_pre + 1e-8)) - 1.0
    breakout_60_flag = 1.0 if breakout_60 > 0.0 else 0.0
    
    # 向下跌破
    breakdown_60 = (close_t / (low_60_pre + 1e-8)) - 1.0
    breakdown_60_flag = 1.0 if breakdown_60 < 0.0 else 0.0
    
    # 保留统计分位数
    imbalance_mean = aligned_min_df["imbalance"].mean()
    imbalance_std = aligned_min_df["imbalance"].std()
    p10_val = aligned_min_df["imbalance"].quantile(0.1)
    p90_val = aligned_min_df["imbalance"].quantile(0.9)
    
    return pl.DataFrame({
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
        "buy_persistence": [buy_persistence],
        "sell_persistence": [sell_persistence],
        "price_return_t": [price_return_t],
        "atr_pct_10": [atr_pct_10],
        "price_response_norm": [price_response_norm],
        "response_factor": [response_factor],
        "effort_factor": [effort_factor],
        "buy_absorption_v1": [buy_absorption_v1],
        "sell_absorption_v1": [sell_absorption_v1],
        "buy_absorption_v2": [buy_absorption_v2],
        "sell_absorption_v2": [sell_absorption_v2],
        "pp_60_close": [pp_60_close],
        "breakout_60": [breakout_60],
        "breakout_60_flag": [breakout_60_flag],
        "breakdown_60": [breakdown_60],
        "breakdown_60_flag": [breakdown_60_flag],
        "observed_minute_ratio": [observed_ratio], 
        "empty_minute_ratio": [empty_ratio],
        "direction_coverage_ratio": [dir_coverage], 
        "data_quality_score": [dq_score],
        **auction_dict, 
        **limit_dict
    })
