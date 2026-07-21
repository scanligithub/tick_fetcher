import polars as pl

def evaluate_behavior_scores(daily_fact_df: pl.DataFrame) -> pl.DataFrame:
    """
    Layer 5: 生成连续型的建仓、攻击、防守和派发倾向分数。
    """
    # 结合原始微观指标与位置上下文调制计算
    return daily_fact_df.with_columns([
        # 1. 连续型建仓分数：买方吸收强 + 低位位置强化 + 相对强弱保护
        (
            pl.col("buy_absorption_raw") * 
            (1.0 - pl.col("pp_60")) * 
            (1.0 + pl.col("rs_5").clamp(0.0, 1.0)) * 
            pl.col("data_quality_score")
        ).alias("accumulation_score"),

        # 2. 连续型攻击分数：主买攻击大 + 强推动效率 + 正向相对强弱
        (
            pl.col("buy_aggression") * 
            pl.col("price_response") * 
            (1.0 + pl.col("rs_5").clamp(0.0, 1.0)) * 
            pl.col("data_quality_score")
        ).alias("attack_score"),

        # 3. 连续型防守承接：卖方吸收强 + 极度超跌(MA Bias负向极值) + 大盘暴跌下的独立抗跌
        (
            pl.col("sell_absorption_raw") * 
            (1.0 - pl.col("pp_20")) * 
            (1.0 - pl.col("bias_20").clamp(-1.0, 0.0)) * 
            pl.col("data_quality_score")
        ).alias("defense_score"),

        # 4. 连续型派发出货：买方吸收强 + 极高位置 (PP_60高位) + 上行乏力
        (
            pl.col("buy_absorption_raw") * 
            pl.col("pp_60") * 
            (1.0 - pl.col("price_response")) * 
            pl.col("data_quality_score")
        ).alias("distribution_score")
    ]).with_columns([
        # 6. 生成基础分类和置信度 (基于最高分)
        pl.struct([
            "accumulation_score", "attack_score", "defense_score", "distribution_score"
        ]).map_elements(lambda x: _determine_primary_state(x), return_dtype=pl.Struct([
            pl.Field("primary_state", pl.String),
            pl.Field("state_confidence", pl.Float64)
        ]))
    ]).unnest("primary_state")

def _determine_primary_state(scores: dict) -> dict:
    state_mapping = {
        "ACCUMULATION": scores.get("accumulation_score", 0.0),
        "ATTACK": scores.get("attack_score", 0.0),
        "DEFENSE": scores.get("defense_score", 0.0),
        "DISTRIBUTION": scores.get("distribution_score", 0.0)
    }
    # 寻找最高得分项
    best_state = "NEUTRAL"
    best_score = 0.0
    for state, val in state_mapping.items():
        if val is not None and val > best_score:
            best_score = val
            best_state = state
            
    # 若最大分数过低，则归为中性
    if best_score < 0.15:
        best_state = "NEUTRAL"
        best_score = 0.0
        
    return {"primary_state": best_state, "state_confidence": float(best_score)}
