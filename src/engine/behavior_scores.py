import polars as pl

def evaluate_behavior_scores(daily_fact_df):
    return daily_fact_df.with_columns([
        (pl.col("buy_absorption_raw") * (1.0 - pl.col("pp_60")) * (1.0 + pl.col("rs_5").clip(0.0, 1.0)) * pl.col("data_quality_score")).alias("accumulation_score"),
        (pl.col("buy_aggression") * pl.col("price_response") * (1.0 + pl.col("rs_5").clip(0.0, 1.0)) * pl.col("data_quality_score")).alias("attack_score"),
        (pl.col("sell_absorption_raw") * (1.0 - pl.col("pp_20")) * (1.0 - pl.col("bias_20").clip(-1.0, 0.0)) * pl.col("data_quality_score")).alias("defense_score"),
        (pl.col("buy_absorption_raw") * pl.col("pp_60") * (1.0 - pl.col("price_response")) * pl.col("data_quality_score")).alias("distribution_score")
    ]).with_columns([
        pl.struct([
            "accumulation_score", "attack_score", "defense_score", "distribution_score"
        ]).map_elements(lambda x: _determine_primary_state(x), return_dtype=pl.Struct([
            pl.Field("primary_state", pl.String),
            pl.Field("state_confidence", pl.Float64)
        ])).alias("state_struct")
    ]).unnest("state_struct")

def _determine_primary_state(scores):
    state_mapping = {
        "ACCUMULATION": scores.get("accumulation_score", 0.0),
        "ATTACK": scores.get("attack_score", 0.0),
        "DEFENSE": scores.get("defense_score", 0.0),
        "DISTRIBUTION": scores.get("distribution_score", 0.0)
    }
    best_state = "NEUTRAL"
    best_score = 0.0
    for state, val in state_mapping.items():
        if val is not None and val > best_score:
            best_score = val
            best_state = state
            
    if best_score < 0.15:
        best_state = "NEUTRAL"
        best_score = 0.0
    return {"primary_state": best_state, "state_confidence": float(best_score)}
