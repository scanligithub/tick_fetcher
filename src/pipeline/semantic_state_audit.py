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

def run_part0_sample_integrity_audit(df: pl.DataFrame):
    """
    PART 0: 样本完整性审计 (Sample Integrity Audit)
    统计截面总样本量、状态占比，并严格核查核心因子的缺失/Null值情况，确保截面分母 100% 闭环。
    """
    print("\n" + "="*96)
    print("      📊  PART 0: 样本完整性与缺失值审计 (Sample Integrity Audit)      ")
    print("="*96)
    
    total_rows = len(df)
    print(f"📌 截面总样本标的数量: {total_rows} 个")
    
    # 核心字段列表
    key_fields = [
        "primary_state", "state_confidence", "score_margin", "score_dominance",
        "accumulation_score", "attack_score", "defense_score", "distribution_score",
        "buy_absorption_v2", "sell_absorption_v2", "data_quality_score",
        "pp_60_pre", "pp_20_pre", "bias_20_pre", "rs_5_pre",
        "price_return_t", "atr_pct_10_pre", "price_response_norm", "response_factor", "effort_factor"
    ]
    
    header = f"{'Key Field Name':<28} | {'Present Count':<14} | {'Missing / Null Count':<20} | {'Missing Ratio':<12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for field in key_fields:
        if field not in df.columns:
            print(f"{field:<28} | {'0':<14} | {total_rows:<20} | {'100.00%':<12}")
            continue
        
        present_cnt = df[field].drop_nulls().drop_nans().len()
        missing_cnt = total_rows - present_cnt
        missing_pct = (missing_cnt / total_rows) * 100 if total_rows > 0 else 0.0
        
        print(f"{field:<28} | {present_cnt:<14} | {missing_cnt:<20} | {missing_pct:>11.2f}%")
        
    print("="*96)

def run_part1_extreme_response_audit(df: pl.DataFrame):
    """
    PART 1: 极端价格响应来源拆解 (Extreme Response Audit)
    提取 price_response_norm 最高的前 50 只个股，明细拆解其 ATR Floor 触发状态与价格响应分，定位长尾异动源头。
    """
    print("\n" + "="*96)
    print("      ⚡  PART 1: 极端价格响应来源拆解 Audit (Top 50 Extreme Price Response)      ")
    print("="*96)
    
    if "price_response_norm" not in df.columns:
        print("❌ 错误: 数据集中缺少 'price_response_norm' 字段。")
        return

    top50 = df.sort("price_response_norm", descending=True).head(50)
    
    header = f"{'Code':<9} | {'Return(T)':<10} | {'ATR_PCT(T-1)':<13} | {'Floor?':<8} | {'NormResp':<10} | {'RespFactor':<11} | {'ClosePre':<9} | {'CloseAdj':<9}"
    print(header)
    print("-" * len(header))
    
    for row in top50.iter_rows(named=True):
        ret = row.get("price_return_t", 0.0)
        atr_pct = row.get("atr_pct_10_pre", 0.002)
        # 判断是否正好触及或极度接近 0.002 Floor 保护界限
        is_floor = "YES" if abs(atr_pct - 0.002) < 1e-6 else "NO"
        norm_resp = row.get("price_response_norm", 0.0)
        resp_factor = row.get("response_factor", 0.0)
        close_pre = row.get("close_pre", 0.0)
        close_adj = close_pre * (1.0 + ret)
        
        print(f"{row['code']:<9} | {ret*100:>+9.2f}% | {atr_pct*100:>12.4f}% | {is_floor:<8} | {norm_resp:>10.4f} | {resp_factor:>11.4f} | {close_pre:>9.2f} | {close_adj:>9.2f}")
        
    print("="*96)

def run_part2_low_position_dominance_audit(df: pl.DataFrame):
    """
    PART 2: 低水位状态支配与二次筛选实证 (Low Position Dominance Audit)
    按昨收中线水位 PP_60_PRE 划分为 4 个水位桶，统计各桶内的主导行为分布与微观吸收/努力度事实。
    """
    print("\n" + "="*96)
    print("      🌊  PART 2: 低水位状态支配与微观吸收二次筛选实证 (Low Position Audit)      ")
    print("="*96)
    
    # 划分 4 个水位桶
    bucketed_df = df.with_columns([
        pl.when(pl.col("pp_60_pre") == 0.0).then(pl.lit("1. FLOOR (==0)"))
          .when((pl.col("pp_60_pre") > 0.0) & (pl.col("pp_60_pre") < 0.05)).then(pl.lit("2. DEEP_LOW (0~0.05)"))
          .when((pl.col("pp_60_pre") >= 0.05) & (pl.col("pp_60_pre") < 0.15)).then(pl.lit("3. NORMAL_LOW (0.05~0.15)"))
          .otherwise(pl.lit("4. HIGH (>=0.15)"))
          .alias("pos_bucket")
    ])
    
    total_market = len(bucketed_df)
    buckets = sorted(bucketed_df["pos_bucket"].unique().to_list())
    
    for b in buckets:
        sub_df = bucketed_df.filter(pl.col("pos_bucket") == b)
        cnt = len(sub_df)
        ratio = (cnt / total_market) * 100 if total_market > 0 else 0.0
        
        print(f"\n📦 水位分桶: [{b}] | 股票数量: {cnt} ({ratio:.2f}%)")
        print("-" * 96)
        
        # 统计桶内 state 分布与吸收/努力度交叉事实
        state_stats = sub_df.group_by("primary_state").agg([
            pl.len().alias("state_count"),
            pl.col("buy_absorption_v2").mean().alias("mean_buy_abs"),
            pl.col("buy_absorption_v2").median().alias("med_buy_abs"),
            pl.col("sell_absorption_v2").mean().alias("mean_sell_abs"),
            pl.col("sell_absorption_v2").median().alias("med_sell_abs"),
            pl.col("effort_factor").mean().alias("mean_effort")
        ]).sort("state_count", descending=True)
        
        header = f"  {'Primary State':<16} | {'Count':<6} | {'Bucket%':<8} | {'Mean BuyAbsV2':<14} | {'Med BuyAbsV2':<13} | {'Mean SellAbsV2':<14} | {'Mean Effort':<11}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        
        for r in state_stats.iter_rows(named=True):
            st_ratio = (r["state_count"] / cnt) * 100 if cnt > 0 else 0.0
            print(f"  {r['primary_state']:<16} | {r['state_count']:<6} | {st_ratio:>7.2f}% | {r['mean_buy_abs']:>14.4f} | {r['med_buy_abs']:>13.4f} | {r['mean_sell_abs']:>14.4f} | {r['mean_effort']:>11.4f}")

    print("="*96)

def run_part3_confidence_matrix_audit(df: pl.DataFrame):
    """
    PART 3: 状态内部与置信度双向矩阵 (State x Confidence Matrix)
    从“状态内部置信度分布”和“置信度桶内状态构成”两个视角进行双向交叉矩阵审计。
    """
    print("\n" + "="*96)
    print("      🧠  PART 3: 状态与置信度双向交叉矩阵审计 (State x Confidence Matrix)      ")
    print("="*96)
    
    # 划分 5 个置信度桶
    conf_df = df.with_columns([
        pl.when(pl.col("state_confidence") < 0.3).then(pl.lit("1. [<0.3] Very Low"))
          .when((pl.col("state_confidence") >= 0.3) & (pl.col("state_confidence") < 0.5)).then(pl.lit("2. [0.3-0.5] Low"))
          .when((pl.col("state_confidence") >= 0.5) & (pl.col("state_confidence") < 0.7)).then(pl.lit("3. [0.5-0.7] Medium"))
          .when((pl.col("state_confidence") >= 0.7) & (pl.col("state_confidence") < 0.9)).then(pl.lit("4. [0.7-0.9] High"))
          .otherwise(pl.lit("5. [>=0.9] Extreme"))
          .alias("conf_bucket")
    ])
    
    conf_buckets = ["1. [<0.3] Very Low", "2. [0.3-0.5] Low", "3. [0.5-0.7] Medium", "4. [0.7-0.9] High", "5. [>=0.9] Extreme"]
    states = sorted(conf_df["primary_state"].unique().to_list())
    
    # --- 视角 1: 各状态内部分散在各个 Confidence Bucket 的分布 ---
    print("\n🔍 【视角 1】各主导状态内部 -> 置信度分布明细 (Row-wise Confidence Distribution):")
    print("-" * 96)
    v1_header = f"{'Primary State':<16} | {'Total':<6} | " + " | ".join([f"{b[3:12]:<11}" for b in conf_buckets])
    print(v1_header)
    print("-" * len(v1_header))
    
    for st in states:
        st_sub = conf_df.filter(pl.col("primary_state") == st)
        st_total = len(st_sub)
        bucket_pcts = []
        for cb in conf_buckets:
            cb_cnt = len(st_sub.filter(pl.col("conf_bucket") == cb))
            pct = (cb_cnt / st_total * 100) if st_total > 0 else 0.0
            bucket_pcts.append(f"{pct:>10.1f}%")
            
        print(f"{st:<16} | {st_total:<6} | " + " | ".join(bucket_pcts))

    # --- 视角 2: 各 Confidence Bucket 内部 -> 各状态的构成比例 ---
    print("\n🔍 【视角 2】各置信度区间内部 -> 主导状态构成明细 (Column-wise State Composition):")
    print("-" * 96)
    v2_header = f"{'Confidence Bucket':<22} | {'Total':<6} | " + " | ".join([f"{st:<12}" for st in states])
    print(v2_header)
    print("-" * len(v2_header))
    
    for cb in conf_buckets:
        cb_sub = conf_df.filter(pl.col("conf_bucket") == cb)
        cb_total = len(cb_sub)
        state_pcts = []
        for st in states:
            st_cnt = len(cb_sub.filter(pl.col("primary_state") == st))
            pct = (st_cnt / cb_total * 100) if cb_total > 0 else 0.0
            state_pcts.append(f"{pct:>11.1f}%")
            
        print(f"{cb:<22} | {cb_total:<6} | " + " | ".join(state_pcts))

    print("="*96)

def run_part4_score_decomposition_audit(df: pl.DataFrame):
    """
    PART 4: 高分个股得分分量分解 (Score Component Decomposition Audit)
    提取激活门槛 MaxScore >= 0.005 的高分个股 Top 20，精准拆解 RawScore 的各乘数子项来源。
    """
    print("\n" + "="*96)
    print("      🔬  PART 4: 越线标的 (MaxScore >= 0.005) 得分分量精细拆解 (Top 20)      ")
    print("="*96)
    
    # 动态构建各主导状态得分的分量结构
    df_decomp = df.with_columns([
        # 1. 组合计算最高得分与主导得分
        pl.max_horizontal(["accumulation_score", "attack_score", "defense_score", "distribution_score"]).alias("max_raw_score"),
        # 2. 建仓分分量
        (1.0 - pl.col("pp_60_pre")).alias("acc_pos_factor"),
        (1.0 + pl.col("rs_5_pre").clip(0.0, 1.0)).alias("acc_rs_factor"),
        # 3. 进攻分分量
        (pl.col("response_factor") * (1.0 + pl.col("rs_5_pre").clip(0.0, 1.0))).alias("att_resp_rs_factor"),
        # 4. 防御分分量
        (1.0 - pl.col("pp_20_pre")).alias("def_pos_factor"),
        (1.0 - pl.col("bias_20_pre").clip(-1.0, 0.0)).alias("def_bias_factor"),
        # 5. 派发分分量
        (1.0 - pl.col("response_factor")).alias("dist_non_resp_factor")
    ])
    
    high_score_df = df_decomp.filter(pl.col("max_raw_score") >= 0.005).sort("state_confidence", descending=True).head(20)
    
    if high_score_df.is_empty():
        print("   ⚠️ (本日无满足 RawScore >= 0.005 起评分的越线标的)")
        print("="*96)
        return
        
    header = f"{'Code':<9} | {'State':<12} | {'RawScore':<9} | {'Absorption':<10} | {'PosFactor':<10} | {'Resp/Bias/RS':<12} | {'Quality':<8} | {'Confidence':<10}"
    print(header)
    print("-" * len(header))
    
    for r in high_score_df.iter_rows(named=True):
        st = r["primary_state"]
        raw_score = r["max_raw_score"]
        
        if st == "ACCUMULATION":
            abs_val = r["buy_absorption_v2"]
            pos_val = r["acc_pos_factor"]
            resp_bias_val = r["acc_rs_factor"]
        elif st == "ATTACK":
            abs_val = r["buy_absorption_v2"]
            pos_val = r["breakout_60_flag"]
            resp_bias_val = r["att_resp_rs_factor"]
        elif st == "DEFENSE":
            abs_val = r["sell_absorption_v2"]
            pos_val = r["def_pos_factor"]
            resp_bias_val = r["def_bias_factor"]
        elif st == "DISTRIBUTION":
            abs_val = r["buy_absorption_v2"]
            pos_val = r["pp_60_pre"]
            resp_bias_val = r["dist_non_resp_factor"]
        else:
            abs_val = 0.0
            pos_val = 0.0
            resp_bias_val = 0.0
            
        print(f"{r['code']:<9} | {st:<12} | {raw_score:>9.5f} | {abs_val:>10.4f} | {pos_val:>10.4f} | {resp_bias_val:>12.4f} | {r['data_quality_score']:>8.4f} | {r['state_confidence']:>10.4f}")

    print("="*96 + "\n")

def main():
    target_file = find_latest_factor_file()
    if not target_file:
        print("❌ 未在 'data/output' 目录中搜寻到任何因子 Parquet 文件！请确保管道已成功运行。")
        return
        
    print(f"🔍 [Semantic Audit] 正在加载因子文件进行深度语义诊断: {target_file}", flush=True)
    df = pl.read_parquet(target_file)
    
    # 依次执行审计模块 PART 0 ~ PART 4
    run_part0_sample_integrity_audit(df)
    run_part1_extreme_response_audit(df)
    run_part2_low_position_dominance_audit(df)
    run_part3_confidence_matrix_audit(df)
    run_part4_score_decomposition_audit(df)

if __name__ == "__main__":
    main()
