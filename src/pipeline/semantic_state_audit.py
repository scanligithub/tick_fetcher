# FILE: src/pipeline/semantic_state_audit.py
import os
import glob
import polars as pl

def find_latest_factor_file(output_dir="data/output") -> str | None:
    """搜寻最新落盘的日级因子 Parquet 文件"""
    pattern = os.path.join(output_dir, "factors_*.parquet")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.basename)

def run_audit1_extreme_response(df: pl.DataFrame):
    """
    Audit 1: Extreme Response Audit (事实拆解)
    输出 price_response_norm 最高的前 50 只标的，带上真实的 ATR Floor 与 Range Floor 触发状态，
    回答：极端响应是来自真实极端行情，还是价格上下文残留异动？
    """
    print("\n" + "="*112)
    print("      ⚡  AUDIT 1: Extreme Response Facts (Top 50 Extreme Price Response)      ")
    print("="*112)
    
    if "price_response_norm" not in df.columns:
        print("❌ 错误: 数据集中缺少 'price_response_norm' 字段。")
        return

    # 动态判定 ATR Floor 与 Range Floor 触发事实
    df_flagged = df.with_columns([
        # ATR Floor (<= 0.002001)
        (pl.col("atr_pct_10_pre") <= 0.002001).alias("atr_floor_triggered"),
        # Range Floor (effective_range_60 <= close_pre * 0.005 + 1e-6)
        ((pl.col("high_60_pre") - pl.col("low_60_pre")) <= (pl.col("close_pre") * 0.005 + 1e-6)).alias("range_floor_triggered"),
        # 算出 T 日重构收盘价
        (pl.col("close_pre") * (1.0 + pl.col("price_return_t"))).alias("close_t_adjusted")
    ])

    top50 = df_flagged.sort("price_response_norm", descending=True).head(50)
    
    header = f"{'Code':<9} | {'NormResp':<9} | {'Return(T)':<9} | {'ATR_PCT':<8} | {'ATR_Flr':<7} | {'Rng_Flr':<7} | {'ClosePre':<8} | {'CloseAdj':<8} | {'High60':<8} | {'Low60':<8} | {'PP60':<6} | {'State':<11} | {'Conf':<6}"
    print(header)
    print("-" * len(header))
    
    for r in top50.iter_rows(named=True):
        ret = r.get("price_return_t", 0.0)
        atr_pct = r.get("atr_pct_10_pre", 0.0)
        norm_resp = r.get("price_response_norm", 0.0)
        atr_flr = "TRUE" if r.get("atr_floor_triggered", False) else "FALSE"
        rng_flr = "TRUE" if r.get("range_floor_triggered", False) else "FALSE"
        c_pre = r.get("close_pre", 0.0)
        c_adj = r.get("close_t_adjusted", 0.0)
        h60 = r.get("high_60_pre", 0.0)
        l60 = r.get("low_60_pre", 0.0)
        pp60 = r.get("pp_60_pre", 0.0)
        st = r.get("primary_state", "UNKNOWN")
        conf = r.get("state_confidence", 0.0)
        
        print(f"{r['code']:<9} | {norm_resp:>9.4f} | {ret*100:>+8.2f}% | {atr_pct*100:>7.3f}% | {atr_flr:<7} | {rng_flr:<7} | {c_pre:>8.2f} | {c_adj:>8.2f} | {h60:>8.2f} | {l60:>8.2f} | {pp60:>6.2f} | {st:<11} | {conf:>6.2f}")
        
    print("="*112)

def run_audit2_position_state(df: pl.DataFrame):
    """
    Audit 2: Position x State (一维位置分桶与状态分布)
    将 PP60_PRE 切分为 4 层，统计各层内部的状态占比，以及买方/卖方微观吸收与努力度事实。
    """
    print("\n" + "="*112)
    print("      🌊  AUDIT 2: Position x State Facts (4-Bucket Position Distribution)      ")
    print("="*112)
    
    pos_df = df.with_columns([
        pl.when(pl.col("pp_60_pre") == 0.0).then(pl.lit("1. FLOOR (PP60==0)"))
          .when((pl.col("pp_60_pre") > 0.0) & (pl.col("pp_60_pre") < 0.05)).then(pl.lit("2. DEEP_LOW (0<PP60<0.05)"))
          .when((pl.col("pp_60_pre") >= 0.05) & (pl.col("pp_60_pre") < 0.15)).then(pl.lit("3. NORMAL_LOW (0.05<=PP60<0.15)"))
          .otherwise(pl.lit("4. HIGH (PP60>=0.15)"))
          .alias("pos_bucket")
    ])
    
    total_mkt = len(pos_df)
    buckets = sorted(pos_df["pos_bucket"].unique().to_list())
    
    header = f"{'Position Bucket':<28} | {'N':<6} | {'NEU%':<7} | {'ACC%':<7} | {'DEF%':<7} | {'DIST%':<7} | {'ATT%':<7} | {'Mean AbsBuy':<11} | {'Med AbsBuy':<10} | {'Mean Effort':<11} | {'Med Effort':<10}"
    print(header)
    print("-" * len(header))
    
    for b in buckets:
        sub = pos_df.filter(pl.col("pos_bucket") == b)
        n = len(sub)
        if n == 0:
            continue
            
        neu_pct = (len(sub.filter(pl.col("primary_state") == "NEUTRAL")) / n) * 100
        acc_pct = (len(sub.filter(pl.col("primary_state") == "ACCUMULATION")) / n) * 100
        def_pct = (len(sub.filter(pl.col("primary_state") == "DEFENSE")) / n) * 100
        dist_pct = (len(sub.filter(pl.col("primary_state") == "DISTRIBUTION")) / n) * 100
        att_pct = (len(sub.filter(pl.col("primary_state") == "ATTACK")) / n) * 100
        
        mean_abs_buy = sub["buy_absorption_v2"].mean()
        med_abs_buy = sub["buy_absorption_v2"].median()
        mean_eff = sub["effort_factor"].mean()
        med_eff = sub["effort_factor"].median()
        
        print(f"{b:<28} | {n:<6} | {neu_pct:>6.1f}% | {acc_pct:>6.1f}% | {def_pct:>6.1f}% | {dist_pct:>6.1f}% | {att_pct:>6.1f}% | {mean_abs_buy:>11.4f} | {med_abs_buy:>10.4f} | {mean_eff:>11.4f} | {med_eff:>10.4f}")
        
    print("="*112)

def run_audit3_position_absorption_2d(df: pl.DataFrame):
    """
    Audit 3: Position x Absorption 2D Grid (二维交叉阵列 - 核心升级)
    直接回答：位置因子是在“放大真实吸收”，还是在“凭低位置空套制造 ACCUMULATION”？
    """
    print("\n" + "="*112)
    print("      🎯  AUDIT 3: Position x Absorption 2D Matrix (PP60 Bucket x BuyAbsorption Quartile)      ")
    print("="*112)
    
    # 计算全市场 BuyAbsorptionV2 的 4 分位数界限
    q25 = df["buy_absorption_v2"].quantile(0.25)
    q50 = df["buy_absorption_v2"].quantile(0.50)
    q75 = df["buy_absorption_v2"].quantile(0.75)
    
    grid_df = df.with_columns([
        pl.when(pl.col("pp_60_pre") == 0.0).then(pl.lit("1. FLOOR (PP60==0)"))
          .when((pl.col("pp_60_pre") > 0.0) & (pl.col("pp_60_pre") < 0.05)).then(pl.lit("2. DEEP_LOW (0<PP60<0.05)"))
          .when((pl.col("pp_60_pre") >= 0.05) & (pl.col("pp_60_pre") < 0.15)).then(pl.lit("3. NORMAL_LOW (0.05<=PP60<0.15)"))
          .otherwise(pl.lit("4. HIGH (PP60>=0.15)"))
          .alias("pos_bucket"),
          
        pl.when(pl.col("buy_absorption_v2") <= q25).then(pl.lit("Q1 [P0-P25]"))
          .when((pl.col("buy_absorption_v2") > q25) & (pl.col("buy_absorption_v2") <= q50)).then(pl.lit("Q2 [P25-P50]"))
          .when((pl.col("buy_absorption_v2") > q50) & (pl.col("buy_absorption_v2") <= q75)).then(pl.lit("Q3 [P50-P75]"))
          .otherwise(pl.lit("Q4 [P75-P100]"))
          .alias("abs_quartile")
    ])
    
    print(f"📊 截面 BuyAbsorptionV2 全局分位基准: Q25={q25:.4f} | Q50={q50:.4f} | Q75={q75:.4f}\n")
    
    pos_buckets = sorted(grid_df["pos_bucket"].unique().to_list())
    abs_quartiles = ["Q1 [P0-P25]", "Q2 [P25-P50]", "Q3 [P50-P75]", "Q4 [P75-P100]"]
    
    header = f"{'Pos Bucket':<28} | {'Abs Quartile':<12} | {'N':<6} | {'ACC%':<7} | {'DEF%':<7} | {'NEU%':<7} | {'Mean Score':<11} | {'Mean Conf':<10}"
    print(header)
    print("-" * len(header))
    
    for pb in pos_buckets:
        for aq in abs_quartiles:
            cell = grid_df.filter((pl.col("pos_bucket") == pb) & (pl.col("abs_quartile") == aq))
            n = len(cell)
            if n == 0:
                print(f"{pb:<28} | {aq:<12} | {'0':<6} | {'0.0%':<7} | {'0.0%':<7} | {'0.0%':<7} | {'0.0000':<11} | {'0.0000':<10}")
                continue
                
            acc_pct = (len(cell.filter(pl.col("primary_state") == "ACCUMULATION")) / n) * 100
            def_pct = (len(cell.filter(pl.col("primary_state") == "DEFENSE")) / n) * 100
            neu_pct = (len(cell.filter(pl.col("primary_state") == "NEUTRAL")) / n) * 100
            
            mean_score = cell["accumulation_score"].mean()
            mean_conf = cell["state_confidence"].mean()
            
            print(f"{pb:<28} | {aq:<12} | {n:<6} | {acc_pct:>6.1f}% | {def_pct:>6.1f}% | {neu_pct:>6.1f}% | {mean_score:>11.4f} | {mean_conf:>10.4f}")
        print("-" * len(header))

    print("="*112)

def run_audit4_defense_bias(df: pl.DataFrame):
    """
    Audit 4: DEFENSE x Bias20_PRE Facts (防御语义纯度拆解)
    回答：DEFENSE 是否大量发生在 Bias20 ≈ 0 的位置？到底是“低位防御”还是“当前位置的卖压吸收”？
    """
    print("\n" + "="*112)
    print("      🛡️  AUDIT 4: DEFENSE x Bias20_PRE Breakdown (Defense Semantic Purity)      ")
    print("="*112)
    
    bias_df = df.with_columns([
        pl.when(pl.col("bias_20_pre") <= -0.05).then(pl.lit("1. Deep Oversold (Bias <= -5%)"))
          .when((pl.col("bias_20_pre") > -0.05) & (pl.col("bias_20_pre") <= -0.02)).then(pl.lit("2. Mod Oversold (-5% < Bias <= -2%)"))
          .when((pl.col("bias_20_pre") > -0.02) & (pl.col("bias_20_pre") <= 0.0)).then(pl.lit("3. Near Zero (-2% < Bias <= 0%)"))
          .otherwise(pl.lit("4. Positive Bias (Bias > 0%)"))
          .alias("bias_bucket")
    ])
    
    buckets = sorted(bias_df["bias_bucket"].unique().to_list())
    
    header = f"{'Bias20_PRE Bucket':<34} | {'N':<6} | {'DEFENSE%':<9} | {'Mean SellAbsV2':<15} | {'Med SellAbsV2':<14} | {'Mean DefScore':<13}"
    print(header)
    print("-" * len(header))
    
    for b in buckets:
        sub = bias_df.filter(pl.col("bias_bucket") == b)
        n = len(sub)
        if n == 0:
            continue
            
        def_cnt = len(sub.filter(pl.col("primary_state") == "DEFENSE"))
        def_pct = (def_cnt / n) * 100
        
        mean_sell_abs = sub["sell_absorption_v2"].mean()
        med_sell_abs = sub["sell_absorption_v2"].median()
        mean_def_score = sub["defense_score"].mean()
        
        print(f"{b:<34} | {n:<6} | {def_pct:>8.2f}% | {mean_sell_abs:>15.4f} | {med_sell_abs:>14.4f} | {mean_def_score:>13.4f}")
        
    print("="*112)

def run_audit5_state_confidence_matrix(df: pl.DataFrame):
    """
    Audit 5: State x Confidence Matrix (描述性概率交叉分布)
    仅做客观事实描述，不做任何自动过滤与切片修剪。
    """
    print("\n" + "="*112)
    print("      🧠  AUDIT 5: State x Confidence Descriptive Matrix (Pure Descriptive Facts)      ")
    print("="*112)
    
    conf_df = df.with_columns([
        pl.when(pl.col("state_confidence") >= 0.90).then(pl.lit("1. >=0.90"))
          .when((pl.col("state_confidence") >= 0.70) & (pl.col("state_confidence") < 0.90)).then(pl.lit("2. 0.70-0.90"))
          .when((pl.col("state_confidence") >= 0.50) & (pl.col("state_confidence") < 0.70)).then(pl.lit("3. 0.50-0.70"))
          .when((pl.col("state_confidence") >= 0.30) & (pl.col("state_confidence") < 0.50)).then(pl.lit("4. 0.30-0.50"))
          .otherwise(pl.lit("5. <0.30"))
          .alias("conf_bucket")
    ])
    
    conf_buckets = ["1. >=0.90", "2. 0.70-0.90", "3. 0.50-0.70", "4. 0.30-0.50", "5. <0.30"]
    states = sorted(conf_df["primary_state"].unique().to_list())
    
    header = f"{'Primary State':<16} | {'Total N':<8} | " + " | ".join([f"{b:<10}" for b in conf_buckets])
    print(header)
    print("-" * len(header))
    
    for st in states:
        st_sub = conf_df.filter(pl.col("primary_state") == st)
        total_n = len(st_sub)
        bucket_cols = []
        
        for cb in conf_buckets:
            cb_cnt = len(st_sub.filter(pl.col("conf_bucket") == cb))
            pct = (cb_cnt / total_n * 100) if total_n > 0 else 0.0
            bucket_cols.append(f"{cb_cnt:>4} ({pct:>4.1f}%)")
            
        print(f"{st:<16} | {total_n:<8} | " + " | ".join(bucket_cols))

    print("="*112 + "\n")

def main():
    target_file = find_latest_factor_file()
    if not target_file:
        print("❌ 未在 'data/output' 目录中搜寻到任何因子 Parquet 文件！请确认管道已成功运行。")
        return
        
    print(f"🔍 [ReadOnly Semantic Audit] 正在加载因子文件进行完全只读事实审计: {target_file}", flush=True)
    df = pl.read_parquet(target_file)
    
    # 严格执行只读事实审计：Audit 1 ~ Audit 5
    run_audit1_extreme_response(df)
    run_audit2_position_state(df)
    run_audit3_position_absorption_2d(df)
    run_audit4_defense_bias(df)
    run_audit5_state_confidence_matrix(df)

if __name__ == "__main__":
    main()
