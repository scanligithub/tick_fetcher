# FILE: src/engine/behavior_scores.py
import polars as pl

def evaluate_behavior_scores(daily_fact_df: pl.DataFrame) -> pl.DataFrame:
    """
    重构后的机构高阶行为倾向得分评价算子 (V2.2-Final)：
    采用排除未来数据干扰的位置指标计算行为分，并执行双重阈值门槛截断与多维相对置信度推演。
    """
    # 1. 调制四维行为得分（采用 T-1 日相对水位 pp_60_pre 避免 T 日价格自我循环污染）
    df_scored = daily_fact_df.with_columns([
        # 机构低位建仓得分：低水位 + 买方吸收 + 市场超额 + 数据质量
        (pl.col("buy_absorption_v2") * (1.0 - pl.col("pp_60_pre")) * 
         (1.0 + pl.col("rs_5_pre").clip(0.0, 1.0)) * pl.col("data_quality_score")).alias("accumulation_score"),
         
        # 机构高位突击得分：向上突破 + 买方吸收 + 价格未完全释放 + 市场超额 + 数据质量
        (pl.col("buy_absorption_v2") * pl.col("breakout_60_flag") * pl.col("response_factor") * 
         (1.0 + pl.col("rs_5_pre").clip(0.0, 1.0)) * pl.col("data_quality_score")).alias("attack_score"),
         
        # 机构防御性承接得分：低水位 + 卖方吸收（承接）+ 乖离回补 + 数据质量
        (pl.col("sell_absorption_v2") * (1.0 - pl.col("pp_20_pre")) * 
         (1.0 - pl.col("bias_20_pre").clip(-1.0, 0.0)) * pl.col("data_quality_score")).alias("defense_score"),
         
        # 机构高位筹码派发得分：高水位 + 买方吸收（散户接盘）+ 价格滞涨 + 数据质量
        (pl.col("buy_absorption_v2") * pl.col("pp_60_pre") * 
         (1.0 - pl.col("response_factor")) * pl.col("data_quality_score")).alias("distribution_score")
    ])

    # 2. 执行判定，判定主导行为与组合竞争置信度
    res_df = df_scored.with_columns([
        pl.struct([
            "accumulation_score", "attack_score", "defense_score", "distribution_score"
        ]).map_elements(lambda x: _determine_state_machine_v2(x), return_dtype=pl.Struct([
            pl.Field("primary_state", pl.String),
            pl.Field("state_confidence", pl.Float64),
            pl.Field("score_margin", pl.Float64),
            pl.Field("score_dominance", pl.Float64)
        ])).alias("state_struct")
    ]).unnest("state_struct")

    return res_df

def _determine_state_machine_v2(scores: dict) -> dict:
    """
    V2.2-Final 状态判定机：双重门槛短路 + 领先边缘度(Margin) + 能量支配度(Dominance)
    """
    state_mapping = {
        "ACCUMULATION": scores.get("accumulation_score", 0.0),
        "ATTACK": scores.get("attack_score", 0.0),
        "DEFENSE": scores.get("defense_score", 0.0),
        "DISTRIBUTION": scores.get("distribution_score", 0.0)
    }

    # 提取第1和第2领先分数
    sorted_scores = sorted(state_mapping.items(), key=lambda x: x[1], reverse=True)
    best_state, best_score = sorted_scores[0]
    second_state, second_score = sorted_scores[1]
    
    best_score = best_score if best_score is not None else 0.0
    second_score = second_score if second_score is not None else 0.0
    sum_all = sum([v for v in state_mapping.values() if v is not None])

    # 1. 前置双重强门槛短路：
    # 🚀 核心重构校准：由于V2.2中V2吸收因子集成了努力度（均值约1/12），此处门槛由 0.15 对应校准为 0.005，完美契合 Top 8% 优势信号极值触发区间
    if best_score < 0.005:
        return {
            "primary_state": "NEUTRAL",
            "state_confidence": 0.0,
            "score_margin": 0.0,
            "score_dominance": 0.0
        }

    # 2. 测度 1: 领先边缘度 (Margin)
    score_margin = (best_score - second_score) / (best_score + 1e-8)
    
    # 3. 测度 2: 能量支配度 (Dominance)
    score_dominance = best_score / (sum_all + 1e-8)
    
    # 4. 最终复合置信度
    state_confidence = 0.5 * score_margin + 0.5 * score_dominance

    return {
        "primary_state": best_state,
        "state_confidence": float(state_confidence),
        "score_margin": float(score_margin),
        "score_dominance": float(score_dominance)
    }
