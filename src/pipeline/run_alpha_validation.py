# FILE: src/pipeline/run_alpha_validation.py
import os
import sys
import glob
import subprocess
import polars as pl

def find_latest_factor_file(output_dir="data/output") -> str | None:
    """搜寻最新落盘的日级因子 Parquet 文件"""
    pattern = os.path.join(output_dir, "factors_*.parquet")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.basename)

def ensure_fetcher_core():
    """确保 Go 内核可执行文件存在，若不存在则动态编译"""
    if os.path.exists("./fetcher_core"):
        return
    print("🛠️  [GHA Environment] 未检测到 fetcher_core，正在动态编译 Go 内核...", flush=True)
    root_dir = os.getcwd()
    go_dir = os.path.join(root_dir, "src", "go_fetcher")
    out_path = os.path.join(root_dir, "fetcher_core")
    
    cmd = ["go", "build", "-o", out_path, "fetcher.go"]
    res = subprocess.run(cmd, cwd=go_dir, capture_output=True)
    if res.returncode != 0:
        print(f"❌ Go 内核编译失败: {res.stderr.decode('utf-8')}", flush=True)
        sys.exit(1)

def fetch_forward_klines(codes: list[str], kline_out_csv: str):
    """通过 Go 内核批量拉取最新 K 线数据以推导 Forward Returns"""
    print(f"📈 [GHA Environment] 正在为 {len(codes)} 只标的拉取历史与未来 K 线数据...", flush=True)
    codes_str = ",".join(codes)
    cmd = ["./fetcher_core", "-mode=kline", f"-codes={codes_str}", f"-out={kline_out_csv}"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"⚠️ K线拉取告警: {res.stderr}", flush=True)

def append_to_github_summary(markdown_text: str):
    """输出 Markdown 内容至 GitHub Actions Step Summary 网页面板"""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(markdown_text + "\n\n")
        except Exception as e:
            print(f"⚠️ 写入 GITHUB_STEP_SUMMARY 失败: {e}")

def calculate_forward_returns(factor_df: pl.DataFrame, kline_csv: str) -> pl.DataFrame:
    """
    向量化推导 T+1, T+2, T+3, T+5, T+10 的未来真实收益率 (Forward Returns)
    利用 Polars .shift(-k).over("code") 彻底切断未来函数污染
    """
    if not os.path.exists(kline_csv) or os.path.getsize(kline_csv) < 100:
        print("⚠️ 警告: K线数据文件为空，无法计算 Forward Returns。")
        return factor_df

    kline_df = pl.read_csv(
        kline_csv,
        schema_overrides={
            "code": pl.String, "date": pl.String,
            "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
            "volume": pl.Float64, "amount": pl.Float64
        }
    )

    if kline_df.is_empty():
        return factor_df

    # 按代码和日期升序排列，向未来 shift 计算收益
    kline_df = kline_df.sort(["code", "date"]).with_columns([
        ((pl.col("close").shift(-1).over("code") / pl.col("close")) - 1.0).alias("fwd_ret_t1"),
        ((pl.col("close").shift(-2).over("code") / pl.col("close")) - 1.0).alias("fwd_ret_t2"),
        ((pl.col("close").shift(-3).over("code") / pl.col("close")) - 1.0).alias("fwd_ret_t3"),
        ((pl.col("close").shift(-5).over("code") / pl.col("close")) - 1.0).alias("fwd_ret_t5"),
        ((pl.col("close").shift(-10).over("code") / pl.col("close")) - 1.0).alias("fwd_ret_t10"),
    ])

    # 与因子表在 [code, date] 维度精确定向连接
    joined_df = factor_df.join(
        kline_df.select(["code", "date", "fwd_ret_t1", "fwd_ret_t2", "fwd_ret_t3", "fwd_ret_t5", "fwd_ret_t10"]),
        on=["code", "date"],
        how="left"
    )
    return joined_df

def compute_rank_ic(df: pl.DataFrame, score_col: str, ret_col: str) -> float | None:
    """使用纯 Polars 向量化计算 Spearman Rank IC (即 Percentile Ranks 的 Pearson Correlation)"""
    valid_sub = df.select([score_col, ret_col]).drop_nulls().drop_nans()
    if len(valid_sub) < 10:
        return None
    
    corr_df = valid_sub.select(
        pl.corr(pl.col(score_col).rank(), pl.col(ret_col).rank()).alias("rank_ic")
    )
    res = corr_df["rank_ic"][0]
    return float(res) if res is not None else None

def run_part0_alignment_audit(df: pl.DataFrame, factor_file: str) -> str:
    """PART 0: 样本对齐与未来收益覆盖率审计"""
    total_n = len(df)
    t1_valid = len(df.filter(pl.col("fwd_ret_t1").is_not_null()))
    t3_valid = len(df.filter(pl.col("fwd_ret_t3").is_not_null()))
    t5_valid = len(df.filter(pl.col("fwd_ret_t5").is_not_null()))
    t10_valid = len(df.filter(pl.col("fwd_ret_t10").is_not_null()))

    text = f"""
================================================================================================
      📊  PART 0: Sample Alignment & Forward Returns Coverage Audit
================================================================================================
Target Factor File : {os.path.basename(factor_file)}
Total Stocks Fact  : {total_n}
Forward Returns Coverage:
  ├── T+1 Forward Return Valid : {t1_valid:<6} ({t1_valid/total_n*100:>5.1f}%)
  ├── T+3 Forward Return Valid : {t3_valid:<6} ({t3_valid/total_n*100:>5.1f}%)
  ├── T+5 Forward Return Valid : {t5_valid:<6} ({t5_valid/total_n*100:>5.1f}%)
  └── T+10 Forward Return Valid: {t10_valid:<6} ({t10_valid/total_n*100:>5.1f}%)
================================================================================================
"""
    print(text)
    
    md = f"""### 📊 PART 0: 样本对齐与未来收益覆盖率
* **因子文件**: `{os.path.basename(factor_file)}`
* **截面样本数**: `{total_n}`
* **T+1 覆盖率**: `{t1_valid}` ({t1_valid/total_n*100:.1f}%)
* **T+3 覆盖率**: `{t3_valid}` ({t3_valid/total_n*100:.1f}%)
* **T+5 覆盖率**: `{t5_valid}` ({t5_valid/total_n*100:.1f}%)
"""
    return md

def run_part1_rank_ic_audit(df: pl.DataFrame) -> str:
    """PART 1: 连续型行为得分 Rank IC 与 IC 衰减分析"""
    target_scores = [
        ("accumulation_score", "Low Pos Acc Score"),
        ("attack_score", "High Pos Attack Score"),
        ("defense_score", "Low Pos Def Score"),
        ("distribution_score", "High Pos Dist Score"),
        ("buy_absorption_v2", "Buy Absorption V2"),
        ("sell_absorption_v2", "Sell Absorption V2"),
        ("effort_factor", "Effort Factor"),
        ("response_factor", "Price Response Factor")
    ]
    
    ret_cols = ["fwd_ret_t1", "fwd_ret_t3", "fwd_ret_t5", "fwd_ret_t10"]
    
    print("\n" + "="*96)
    print("      ⚡  PART 1: Continuous Behavior Scores Spearman Rank IC & Decay Audit      ")
    print("="*96)
    header = f"{'Factor Score Name':<28} | {'Rank IC (T+1)':<14} | {'Rank IC (T+3)':<14} | {'Rank IC (T+5)':<14} | {'Rank IC (T+10)':<14}"
    print(header)
    print("-" * len(header))

    md_table = "| Factor Score Name | Rank IC (T+1) | Rank IC (T+3) | Rank IC (T+5) | Rank IC (T+10) |\n| :--- | :--- | :--- | :--- | :--- |\n"

    for score_col, label in target_scores:
        if score_col not in df.columns:
            continue
        ic_t1 = compute_rank_ic(df, score_col, "fwd_ret_t1")
        ic_t3 = compute_rank_ic(df, score_col, "fwd_ret_t3")
        ic_t5 = compute_rank_ic(df, score_col, "fwd_ret_t5")
        ic_t10 = compute_rank_ic(df, score_col, "fwd_ret_t10")

        fmt_t1 = f"{ic_t1:>+14.4f}" if ic_t1 is not None else f"{'N/A':>14}"
        fmt_t3 = f"{ic_t3:>+14.4f}" if ic_t3 is not None else f"{'N/A':>14}"
        fmt_t5 = f"{ic_t5:>+14.4f}" if ic_t5 is not None else f"{'N/A':>14}"
        fmt_t10 = f"{ic_t10:>+14.4f}" if ic_t10 is not None else f"{'N/A':>14}"

        print(f"{label:<28} | {fmt_t1} | {fmt_t3} | {fmt_t5} | {fmt_t10}")

        md_t1 = f"{ic_t1:+.4f}" if ic_t1 is not None else "N/A"
        md_t3 = f"{ic_t3:+.4f}" if ic_t3 is not None else "N/A"
        md_t5 = f"{ic_t5:+.4f}" if ic_t5 is not None else "N/A"
        md_t10 = f"{ic_t10:+.4f}" if ic_t10 is not None else "N/A"
        md_table += f"| **{label}** | {md_t1} | {md_t3} | {md_t5} | {md_t10} |\n"

    print("="*96)
    return "### ⚡ PART 1: 行为得分 Rank IC 与 IC 衰减分析\n" + md_table

def run_part2_state_returns_audit(df: pl.DataFrame) -> str:
    """PART 2: 状态机判定结果的 Forward Return 表现与胜率剖析"""
    print("\n" + "="*112)
    print("      🧠  PART 2: Primary State Forward Return & Win Rate Profile      ")
    print("="*112)
    header = f"{'Primary State':<16} | {'N':<6} | {'Mean T+1':<10} | {'WinRate T+1':<12} | {'Mean T+3':<10} | {'WinRate T+3':<12} | {'Mean T+5':<10} | {'WinRate T+5':<12}"
    print(header)
    print("-" * len(header))

    md_table = "| Primary State | Count | Mean Return (T+1) | Win Rate (T+1) | Mean Return (T+3) | Win Rate (T+3) |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"

    states = sorted(df["primary_state"].unique().to_list())
    for st in states:
        sub = df.filter(pl.col("primary_state") == st)
        n = len(sub)
        if n == 0:
            continue

        # T+1
        sub_t1 = sub.filter(pl.col("fwd_ret_t1").is_not_null())
        mean_t1 = sub_t1["fwd_ret_t1"].mean() if len(sub_t1) > 0 else 0.0
        win_t1 = (len(sub_t1.filter(pl.col("fwd_ret_t1") > 0)) / len(sub_t1) * 100) if len(sub_t1) > 0 else 0.0

        # T+3
        sub_t3 = sub.filter(pl.col("fwd_ret_t3").is_not_null())
        mean_t3 = sub_t3["fwd_ret_t3"].mean() if len(sub_t3) > 0 else 0.0
        win_t3 = (len(sub_t3.filter(pl.col("fwd_ret_t3") > 0)) / len(sub_t3) * 100) if len(sub_t3) > 0 else 0.0

        # T+5
        sub_t5 = sub.filter(pl.col("fwd_ret_t5").is_not_null())
        mean_t5 = sub_t5["fwd_ret_t5"].mean() if len(sub_t5) > 0 else 0.0
        win_t5 = (len(sub_t5.filter(pl.col("fwd_ret_t5") > 0)) / len(sub_t5) * 100) if len(sub_t5) > 0 else 0.0

        print(f"{st:<16} | {n:<6} | {mean_t1*100:>+9.2f}% | {win_t1:>11.1f}% | {mean_t3*100:>+9.2f}% | {win_t3:>11.1f}% | {mean_t5*100:>+9.2f}% | {win_t5:>11.1f}%")
        md_table += f"| **{st}** | {n} | {mean_t1*100:+.2f}% | {win_t1:.1f}% | {mean_t3*100:+.2f}% | {win_t3:.1f}% |\n"

    print("="*112)
    return "### 🧠 PART 2: 各大状态未来收益率与胜率表\n" + md_table

def run_part3_confidence_tier_audit(df: pl.DataFrame) -> str:
    """PART 3: 置信度梯队 (Confidence Tier) 收益单调性验证"""
    print("\n" + "="*112)
    print("      🎯  PART 3: Confidence Tier Monotonicity Audit (State x Confidence Profile)      ")
    print("="*112)

    df_tier = df.with_columns([
        pl.when(pl.col("state_confidence") >= 0.70).then(pl.lit("1. High (>=0.70)"))
          .when((pl.col("state_confidence") >= 0.50) & (pl.col("state_confidence") < 0.70)).then(pl.lit("2. Normal (0.50-0.70)"))
          .when((pl.col("state_confidence") >= 0.30) & (pl.col("state_confidence") < 0.50)).then(pl.lit("3. Weak (0.30-0.50)"))
          .otherwise(pl.lit("4. Low (<0.30)"))
          .alias("conf_tier")
    ])

    header = f"{'State':<14} | {'Confidence Tier':<22} | {'N':<6} | {'Mean Ret T+1':<13} | {'WinRate T+1':<12} | {'Mean Ret T+3':<13} | {'WinRate T+3':<12}"
    print(header)
    print("-" * len(header))

    md_table = "| State | Confidence Tier | Count | Mean Ret (T+1) | Win Rate (T+1) | Mean Ret (T+3) | Win Rate (T+3) |\n| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"

    target_states = ["ACCUMULATION", "ATTACK", "DEFENSE", "DISTRIBUTION"]
    tiers = ["1. High (>=0.70)", "2. Normal (0.50-0.70)", "3. Weak (0.30-0.50)", "4. Low (<0.30)"]

    for st in target_states:
        for tr in tiers:
            cell = df_tier.filter((pl.col("primary_state") == st) & (pl.col("conf_tier") == tr))
            n = len(cell)
            if n == 0:
                continue

            sub_t1 = cell.filter(pl.col("fwd_ret_t1").is_not_null())
            m_t1 = sub_t1["fwd_ret_t1"].mean() if len(sub_t1) > 0 else 0.0
            w_t1 = (len(sub_t1.filter(pl.col("fwd_ret_t1") > 0)) / len(sub_t1) * 100) if len(sub_t1) > 0 else 0.0

            sub_t3 = cell.filter(pl.col("fwd_ret_t3").is_not_null())
            m_t3 = sub_t3["fwd_ret_t3"].mean() if len(sub_t3) > 0 else 0.0
            w_t3 = (len(sub_t3.filter(pl.col("fwd_ret_t3") > 0)) / len(sub_t3) * 100) if len(sub_t3) > 0 else 0.0

            print(f"{st:<14} | {tr:<22} | {n:<6} | {m_t1*100:>+12.2f}% | {w_t1:>11.1f}% | {m_t3*100:>+12.2f}% | {w_t3:>11.1f}%")
            md_table += f"| **{st}** | {tr} | {n} | {m_t1*100:+.2f}% | {w_t1:.1f}% | {m_t3*100:+.2f}% | {w_t3:.1f}% |\n"
        print("-" * len(header))

    print("="*112)
    return "### 🎯 PART 3: 置信度梯队收益单调性验证表\n" + md_table

def main():
    target_file = find_latest_factor_file()
    if not target_file:
        print("❌ 未在 'data/output' 目录找到任何因子 Parquet 文件！Alpha 验证程序终止。")
        return

    print(f"🔍 [Alpha Validation Engine] 正在加载最新因子文件进行 Alpha 验证: {target_file}", flush=True)
    factor_df = pl.read_parquet(target_file)

    # 1. 确保 Go 提取内核存在
    ensure_fetcher_core()

    # 2. 动态获取全市场最新 K 线
    codes = factor_df["code"].unique().to_list()
    kline_csv = "data/temp_chunks/alpha_validation_kline.csv"
    fetch_forward_klines(codes, kline_csv)

    # 3. 拼接推导 T+1 ~ T+10 Forward Returns
    validated_df = calculate_forward_returns(factor_df, kline_csv)

    # 4. 执行 4 大审计面板
    md0 = run_part0_alignment_audit(validated_df, target_file)
    md1 = run_part1_rank_ic_audit(validated_df)
    md2 = run_part2_state_returns_audit(validated_df)
    md3 = run_part3_confidence_tier_audit(validated_df)

    # 5. 输出 Markdown 至 GitHub Step Summary
    full_markdown = f"# 📈 TDX High-Freq Behavior Factors: Forward Alpha & Rank IC Report\n\n" + md0 + "\n" + md1 + "\n" + md2 + "\n" + md3
    append_to_github_summary(full_markdown)

    # 清理临时文件
    if os.path.exists(kline_csv):
        os.remove(kline_csv)

    print("\n🏁 [Alpha Validation Engine] 跨截面 Alpha 收益率与 Rank IC 验证完成。\n", flush=True)

if __name__ == "__main__":
    main()
