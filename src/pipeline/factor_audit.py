# FILE: src/pipeline/factor_audit.py
import os
import glob
import polars as pl

def find_latest_factor_file(output_dir="data/output"):
    """
    搜寻最新的日级因子 Parquet 文件
    """
    pattern = os.path.join(output_dir, "factors_*.parquet")
    files = glob.glob(pattern)
    if not files:
        return None
    # 按修改时间或文件名排序取最新一个
    return max(files, key=os.path.basename)

def run_distribution_audit(df: pl.DataFrame):
    """
    对核心因子的截面数值分布进行分位数与极端值审计
    """
    print("\n" + "="*80)
    print("      📊  PART 1: 核心因子截面数值分布审计 (Factor Distribution Audit)      ")
    print("="*80)
    
    # 待审计的重点因子清单
    target_cols = [
        "auction_tick_count", "auction_obs_qty", "auction_stability",
        "price_return_t", "price_response_norm", "response_factor",
        "effort_factor", "data_quality_score",
        "buy_absorption_v1", "buy_absorption_v2",
        "pp_60_pre", "pp_60_close", "breakout_60", "breakdown_60",
        "accumulation_score", "attack_score", "defense_score", "distribution_score",
        "score_margin", "score_dominance", "state_confidence"
    ]
    
    # 过滤出存在于数据表中的列
    audit_cols = [c for c in target_cols if c in df.columns]
    
    # 构建 ASCII 分布宽表
    header = f"{'Factor Name':<28} | {'Null%':<6} | {'Min':<8} | {'P1%':<8} | {'P25%':<8} | {'P50%':<8} | {'P75%':<8} | {'P95%':<8} | {'P99%':<8} | {'Max':<8}"
    print(header)
    print("-" * len(header))
    
    n_rows = len(df)
    for col in audit_cols:
        col_series = df[col].drop_nans().drop_nulls()
        if len(col_series) == 0:
            print(f"{col:<28} | {'100%':<6} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<8}")
            continue
            
        null_ratio = (n_rows - len(col_series)) / n_rows * 100
        val_min = col_series.min()
        val_max = col_series.max()
        
        # 计算指定分位数
        p1 = col_series.quantile(0.01)
        p25 = col_series.quantile(0.25)
        p50 = col_series.quantile(0.50)
        p75 = col_series.quantile(0.75)
        p95 = col_series.quantile(0.95)
        p99 = col_series.quantile(0.99)
        
        print(f"{col:<28} | {null_ratio:>5.1f}% | {val_min:>8.4f} | {p1:>8.4f} | {p25:>8.4f} | {p50:>8.4f} | {p75:>8.4f} | {p95:>8.4f} | {p99:>8.4f} | {val_max:>8.4f}")
    print("="*80)

def run_state_machine_audit(df: pl.DataFrame):
    """
    审计双重门槛状态判定机下的决策空间分布
    """
    print("\n" + "="*80)
    print("      🧠  PART 2: 状态机判定结果及置信度分布审计 (State Machine Audit)      ")
    print("="*80)
    
    if "primary_state" not in df.columns:
        print("❌ 错误：数据表中未找到 'primary_state' 判定列。")
        return
        
    state_counts = df.group_by("primary_state").agg([
        pl.len().alias("count"),
        pl.col("state_confidence").mean().alias("avg_confidence"),
        pl.col("state_confidence").quantile(0.5).alias("median_confidence"),
        pl.col("state_confidence").max().alias("max_confidence")
    ]).sort("count", descending=True)
    
    total_stocks = len(df)
    print(f"{'Primary State':<18} | {'Count':<6} | {'Ratio':<8} | {'Avg Confidence':<14} | {'Med Confidence':<14} | {'Max Confidence':<14}")
    print("-" * 88)
    for row in state_counts.iter_rows(named=True):
        ratio = row["count"] / total_stocks * 100
        print(f"{row['primary_state']:<18} | {row['count']:<6} | {ratio:>6.2f}% | {row['avg_confidence']:>14.4f} | {row['median_confidence']:>14.4f} | {row['max_confidence']:>14.4f}")
    print("="*80)

def run_case_studies(df: pl.DataFrame):
    """
    物理语义回归测试：提取各个典型判定下的个股样本，供人工检查与真实盘面核对
    """
    print("\n" + "="*80)
    print("      🔍  PART 3: 典型高确定性机构行为案例审计 (Quant Case Studies)      ")
    print("="*80)
    
    # 1. 强吸收建仓案例 (High ACC Score & Confidence)
    if "accumulation_score" in df.columns:
        acc_samples = df.filter(pl.col("primary_state") == "ACCUMULATION").sort("state_confidence", descending=True).head(3)
        print("\n📈 典型[机构建仓 (ACCUMULATION)]案例:")
        if acc_samples.is_empty():
            print("   (本日无满足强门槛的建仓标的)")
        for r in acc_samples.iter_rows(named=True):
            print(f"   标的: {r['code']} | 日收盘: {r['close_pre']*(1+r['price_return_t']):.2f} ({r['price_return_t']*100:+.2f}%) "
                  f"| 主买占比: {r['buy_aggression']*100:.1f}% | 努力度(Effort): {r['effort_factor']:.2f} | 价格响应分: {r['response_factor']:.2f} "
                  f"| 吸收分V2: {r['buy_absorption_v2']:.3f} | 昨中线水位 PP60_PRE: {r['pp_60_pre']:.2f} | 置信度: {r['state_confidence']:.2f}")

    # 2. 向上突破进攻案例 (High ATTACK Score & Breakout)
    if "attack_score" in df.columns:
        att_samples = df.filter((pl.col("primary_state") == "ATTACK") & (pl.col("breakout_60_flag") == 1.0)).sort("state_confidence", descending=True).head(3)
        print("\n⚡ 典型[向上进攻 (ATTACK)]突破案例:")
        if att_samples.is_empty():
            print("   (本日无满足强门槛的突破进攻标的)")
        for r in att_samples.iter_rows(named=True):
            print(f"   标的: {r['code']} | 日收盘: {r['close_pre']*(1+r['price_return_t']):.2f} ({r['price_return_t']*100:+.2f}%) "
                  f"| 突破幅度: {r['breakout_60']*100:+.2f}% | 吸收分V2: {r['buy_absorption_v2']:.3f} "
                  f"| 领先边缘(Margin): {r['score_margin']:.2f} | 能量支配(Dominance): {r['score_dominance']:.2f} | 置信度: {r['state_confidence']:.2f}")

    # 3. 强承接防御案例 (High DEFENSE Score)
    if "defense_score" in df.columns:
        def_samples = df.filter(pl.col("primary_state") == "DEFENSE").sort("state_confidence", descending=True).head(3)
        print("\n🛡️ 典型[低位承接 (DEFENSE)]防御案例:")
        if def_samples.is_empty():
            print("   (本日无满足强门槛的防御承接标的)")
        for r in def_samples.iter_rows(named=True):
            print(f"   标的: {r['code']} | 日收盘: {r['close_pre']*(1+r['price_return_t']):.2f} ({r['price_return_t']*100:+.2f}%) "
                  f"| 主卖占比: {r['sell_aggression']*100:.1f}% | 卖方承接V2: {r['sell_absorption_v2']:.3f} "
                  f"| 昨20日乖离 Bias20_PRE: {r['bias_20_pre']*100:+.1f}% | 置信度: {r['state_confidence']:.2f}")

    # 4. 有效集合竞价稳定性对比 (检查 Quality & Stability 的复合效用)
    if "auction_stability" in df.columns:
        print("\n🔔 集合竞价观测校准审计:")
        # 挑选仅有1笔观测的
        low_obs = df.filter(pl.col("auction_tick_count") == 1).head(1)
        # 挑选高频稳定观测的
        high_obs = df.filter((pl.col("auction_tick_count") >= 5) & (pl.col("auction_range") == 0.0)).head(1)
        # 挑选高频剧烈变动观测的
        turbulent_obs = df.filter((pl.col("auction_tick_count") >= 5) & (pl.col("auction_stability") < 0.2)).sort("auction_range", descending=True).head(1)
        
        for name, item_df in [("仅单次撮合(1 Tick)", low_obs), ("稳定观测(>=5 Ticks且无波动)", high_obs), ("剧烈变动竞价(高频宽幅振荡)", turbulent_obs)]:
            if item_df.is_empty():
                continue
            r = item_df.to_dicts()[0]
            print(f"   {name:<28}: 标的: {r['code']} | Ticks: {r['auction_tick_count']} | Range: {r['auction_range']:.3f} "
                  f"| 观测质量分: {r['auction_obs_qty']:.2f} | 物理稳定性: {r['auction_stab_obs']:.2f} | 组合稳定性: {r['auction_stability']:.2f}")

    print("="*80 + "\n")

def run_score_decomposition_audit(df: pl.DataFrame):
    """
    🚀 V2.2-Final 核心设计：得分分量分解审计
    在不改动底层数据Schema的情况下，动态拆解每个行为得分的各组成物理乘数，揪出将股票推过起评分的“幕后元凶”。
    """
    print("\n" + "="*80)
    print("      📊  PART 4: 得分分量分解审计 (Score Component Decomposition Audit)      ")
    print("="*80)
    
    # 在内存中计算出各个高阶乘数的物理分量
    df_decomp = df.with_columns([
        # 建仓得分的分量
        (1.0 - pl.col("pp_60_pre")).alias("acc_pos_factor"),
        (1.0 + pl.col("rs_5_pre").clip(0.0, 1.0)).alias("acc_rs_factor"),
        # 防御得分的分量
        (1.0 - pl.col("pp_20_pre")).alias("def_pos_factor"),
        (1.0 - pl.col("bias_20_pre").clip(-1.0, 0.0)).alias("def_bias_factor")
    ])
    
    decomp_cols = {
        "ACC_Absorption (Buy V2)": "buy_absorption_v2",
        "ACC_Position Factor (1-PP60)": "acc_pos_factor",
        "ACC_Excess Return (1+RS5)": "acc_rs_factor",
        "DEF_Absorption (Sell V2)": "sell_absorption_v2",
        "DEF_Position Factor (1-PP20)": "def_pos_factor",
        "DEF_Bias Factor (1-Bias20)": "def_bias_factor",
        "Data Quality Score": "data_quality_score"
    }
    
    # 1. 打印各子分量的截面分布分位数，定位整体膨胀分量
    header = f"{'Constituent Factor Name':<28} | {'Min':<8} | {'P25%':<8} | {'P50%':<8} | {'P75%':<8} | {'P95%':<8} | {'Max':<8}"
    print(header)
    print("-" * len(header))
    for name, col in decomp_cols.items():
        if col not in df_decomp.columns:
            continue
        series = df_decomp[col].drop_nulls()
        print(f"{name:<28} | {series.min():>8.4f} | {series.quantile(0.25):>8.4f} | {series.quantile(0.50):>8.4f} | {series.quantile(0.75):>8.4f} | {series.quantile(0.95):>8.4f} | {series.max():>8.4f}")
    
    # 2. 打印触发起评分门槛 (score >= 0.005) 的代表性个股分量拆解 (Top 15)
    print("\n🔥 触发高起评分 (score >= 0.005) 的代表性个股分量拆解 (Top 15):")
    high_scores_df = df_decomp.filter(
        (pl.col("accumulation_score") >= 0.005) | (pl.col("defense_score") >= 0.005)
    ).sort("state_confidence", descending=True).head(15)
    
    if high_scores_df.is_empty():
        print("   (本日无满足 >= 0.005 起评分的激活个股)")
    else:
        detail_header = f"{'Code':<9} | {'State':<12} | {'Raw Score':<9} | {'Absorption':<10} | {'Pos Factor':<10} | {'Resp/Bias':<10} | {'Quality':<8} | {'Confidence':<10}"
        print(detail_header)
        print("-" * len(detail_header))
        for r in high_scores_df.iter_rows(named=True):
            state = r["primary_state"]
            if state == "ACCUMULATION":
                raw_score = r["accumulation_score"]
                absorption = r["buy_absorption_v2"]
                pos_factor = r["acc_pos_factor"]
                resp_bias = r["acc_rs_factor"]
            elif state == "DEFENSE":
                raw_score = r["defense_score"]
                absorption = r["sell_absorption_v2"]
                pos_factor = r["def_pos_factor"]
                resp_bias = r["def_bias_factor"]
            elif state == "DISTRIBUTION":
                raw_score = r["distribution_score"]
                absorption = r["buy_absorption_v2"]
                pos_factor = r["pp_60_pre"]
                resp_bias = 1.0 - r["response_factor"]
            else:
                raw_score = 0.0
                absorption = 0.0
                pos_factor = 0.0
                resp_bias = 0.0
            
            print(f"{r['code']:<9} | {state:<12} | {raw_score:>9.4f} | {absorption:>10.4f} | {pos_factor:>10.4f} | {resp_bias:>10.4f} | {r['data_quality_score']:>8.4f} | {r['state_confidence']:>10.4f}")
    print("="*80)

def main():
    target_file = find_latest_factor_file()
    if not target_file:
        print("❌ 未在 'data/output' 中找到任何日级因子 Parquet 文件。请确认管道脚本已成功运行。")
        return
        
    print(f"🔍 正在加载因子库进行多维语义分布审计: {target_file}", flush=True)
    df = pl.read_parquet(target_file)
    
    # 依次执行审计模块
    run_distribution_audit(df)
    run_state_machine_audit(df)
    run_case_studies(df)
    run_score_decomposition_audit(df) # 启动高阶分量分解审计

if __name__ == "__main__":
    main()
