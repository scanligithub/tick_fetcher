# FILE: src/pipeline/run_alpha_validation.py
import os
import sys
import glob
import subprocess
import polars as pl

def find_factor_files(output_dir="data/output") -> list[str]:
    """搜寻 data/output 目录下所有的日级因子 Parquet 文件"""
    pattern = os.path.join(output_dir, "factors_*.parquet")
    files = glob.glob(pattern)
    return sorted(files)

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
    print(f"📈 [GHA Environment] 正在为 {len(codes)} 只标的拉取历史与最新日 K 线...", flush=True)
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

def calculate_forward_returns_with_guard(factor_df: pl.DataFrame, kline_csv: str) -> tuple[pl.DataFrame, str]:
    """
    带严格时间轴防线的 Forward Returns 计算算子：
    1. 检查最新 K 线收盘日期 latest_kline_date。
    2. 针对每一个 factor_date_T，仅当未来第 k 个交易日实际存在且 date_Tk <= latest_kline_date 时才计算收益。
    3. 未到期的未来收益（如昨日因子的 T+3/T+5）强行置为 None (Null)，杜绝虚假未来收益！
    """
    if not os.path.exists(kline_csv) or os.path.getsize(kline_csv) < 100:
        print("⚠️ 警告: K线数据文件为空，无法计算 Forward Returns。")
        return factor_df, "N/A"

    kline_df = pl.read_csv(
        kline_csv,
        schema_overrides={
            "code": pl.String, "date": pl.String,
            "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
            "volume": pl.Float64, "amount": pl.Float64
        }
    )

    if kline_df.is_empty():
        return factor_df, "N/A"

    latest_kline_date = kline_df["date"].max()

    # 按代码与日期升序排序，向未来 shift 交易日
    kline_df = kline_df.sort(["code", "date"]).with_columns([
        pl.col("close").shift(-1).over("code").alias("close_t1"),
        pl.col("date").shift(-1).over("code").alias("date_t1"),
        pl.col("close").shift(-3).over("code").alias("close_t3"),
        pl.col("date").shift(-3).over("code").alias("date_t3"),
        pl.col("close").shift(-5).over("code").alias("close_t5"),
        pl.col("date").shift(-5).over("code").alias("date_t5"),
        pl.col("close").shift(-10).over("code").alias("close_t10"),
        pl.col("date").shift(-10).over("code").alias("date_t10"),
    ])

    # 🚀 物理时间轴防护关键逻辑：仅当未来交易日实际存在且不晚于数据库最新收盘日时有效
    kline_df = kline_df.with_columns([
        pl.when(pl.col("date_t1").is_not_null() & (pl.col("date_t1") <= latest_kline_date))
          .then((pl.col("close_t1") / pl.col("close")) - 1.0).otherwise(None).alias("fwd_ret_t1"),

        pl.when(pl.col("date_t3").is_not_null() & (pl.col("date_t3") <= latest_kline_date))
          .then((pl.col("close_t3") / pl.col("close")) - 1.0).otherwise(None).alias("fwd_ret_t3"),

        pl.when(pl.col("date_t5").is_not_null() & (pl.col("date_t5") <= latest_kline_date))
          .then((pl.col("close_t5") / pl.col("close")) - 1.0).otherwise(None).alias("fwd_ret_t5"),

        pl.when(pl.col("date_t10").is_not_null() & (pl.col("date_t10") <= latest_kline_date))
          .then((pl.col("close_t10") / pl.col("close")) - 1.0).otherwise(None).alias("fwd_ret_t10"),
    ])

    # 与因子表精确定向连接
    joined_df = factor_df.join(
        kline_df.select(["code", "date", "fwd_ret_t1", "fwd_ret_t3", "fwd_ret_t5", "fwd_ret_t10"]),
        on=["code", "date"],
        how="left"
    )
    return joined_df, latest_kline_date

def compute_rank_ic(df: pl.DataFrame, score_col: str, ret_col: str) -> float | None:
    """纯 Polars 向量化 Spearman Rank IC 计算算子"""
    valid_sub = df.select([score_col, ret_col]).drop_nulls().drop_nans()
    if len(valid_sub) < 10:
        return None
    
    corr_df = valid_sub.select(
        pl.corr(pl.col(score_col).rank(), pl.col(ret_col).rank()).alias("rank_ic")
    )
    res = corr_df["rank_ic"][0]
    return float(res) if res is not None else None

def run_part0_alignment_audit(df: pl.DataFrame, latest_kline_date: str) -> str:
    """PART 0: 严格时间轴样本对齐与交割状态审计"""
    total_n = len(df)
    t1_valid = len(df.filter(pl.col("fwd_ret_t1").is_not_null()))
    t3_valid = len(df.filter(pl.col("fwd_ret_t3").is_not_null()))
    t5_valid = len(df.filter(pl.col("fwd_ret_t5").is_not_null()))
    t10_valid = len(df.filter(pl.col("fwd_ret_t10").is_not_null()))

    # 判定交割状态
    st_t1 = "VALID (已到期交割)" if t1_valid > 0 else "PENDING (未到期)"
    st_t3 = "VALID (已到期交割)" if t3_valid > 0 else "PENDING (未到期)"
    st_t5 = "VALID (已到期交割)" if t5_valid > 0 else "PENDING (未到期)"
    st_t10 = "VALID (已到期交割)" if t10_valid > 0 else "PENDING (未到期)"

    text = f"""
================================================================================================
      📊  PART 0: Timestamp-Protected Sample Alignment & Forward Returns Audit
================================================================================================
Latest Market Kline Date : {latest_kline_date}
Total Evaluated Samples  : {total_n}
Forward Horizon Delivery Status:
  ├── T+1 Forward Return : {t1_valid:<6} ({t1_valid/total_n*100:>5.1f}%) | Status: {st_t1}
  ├── T+3 Forward Return : {t3_valid:<6} ({t3_valid/total_n*100:>5.1f}%) | Status: {st_t3}
  ├── T+5 Forward Return : {t5_valid:<6} ({t5_valid/total_n*100:>5.1f}%) | Status: {st_t5}
  └── T+10 Forward Return: {t10_valid:<6} ({t10_valid/total_n*100:>5.1f}%) | Status: {st_t10}
================================================================================================
"""
    print(text)
    
    md = f"""### 📊 PART 0: 严格时间轴样本对齐与未来收益交割状态
* **最新市场 K 线日期**: `{latest_kline_date}`
* **评估样本总行数**: `{total_n}`
* **T+1 交割状态**: `{t1_valid}` 标的有效 (`{st_t1}`)
* **T+3 交割状态**: `{t3_valid}` 标的有效 (`{st_t3}`)
* **T+5 交割状态**: `{t5_valid}` 标的有效 (`{st_t5}`)
* **T+10 交割状态**: `{t10_valid}` 标的有效 (`{st_t10}`)
"""
    return md

def run_part1_rank_ic_audit(df: pl.DataFrame) -> str:
    """PART 1: 连续型行为得分 Spearman Rank IC 审计"""
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
    
    print("\n" + "="*96)
    print("      ⚡  PART 1: Continuous Behavior Scores Spearman Rank IC Audit      ")
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

        fmt_t1 = f"{ic_t1:>+14.4f}" if ic_t1 is not None else f"{'PENDING':>14}"
        fmt_t3 = f"{ic_t3:>+14.4f}" if ic_t3 is not None else f"{'PENDING':>14}"
        fmt_t5 = f"{ic_t5:>+14.4f}" if ic_t5 is not None else f"{'PENDING':>14}"
        fmt_t10 = f"{ic_t10:>+14.4f}" if ic_t10 is not None else f"{'PENDING':>14}"

        print(f"{label:<28} | {fmt_t1} | {fmt_t3} | {fmt_t5} | {fmt_t10}")

        md_t1 = f"{ic_t1:+.4f}" if ic_t1 is not None else "PENDING"
        md_t3 = f"{ic_t3:+.4f}" if ic_t3 is not None else "PENDING"
        md_t5 = f"{ic_t5:+.4f}" if ic_t5 is not None else "PENDING"
        md_t10 = f"{ic_t10:+.4f}" if ic_t10 is not None else "PENDING"
        md_table += f"| **{label}** | {md_t1} | {md_t3} | {md_t5} | {md_t10} |\n"

    print("="*96)
    return "### ⚡ PART 1: 行为得分 Rank IC 审计\n" + md_table

def run_part2_state_returns_audit(df: pl.DataFrame) -> str:
    """PART 2: 状态机判定结果的 Forward Return 与胜率剖析"""
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
        mean_t1 = sub_t1["fwd_ret_t1"].mean() if len(sub_t1) > 0 else None
        win_t1 = (len(sub_t1.filter(pl.col("fwd_ret_t1") > 0)) / len(sub_t1) * 100) if len(sub_t1) > 0 else None

        # T+3
        sub_t3 = sub.filter(pl.col("fwd_ret_t3").is_not_null())
        mean_t3 = sub_t3["fwd_ret_t3"].mean() if len(sub_t3) > 0 else None
        win_t3 = (len(sub_t3.filter(pl.col("fwd_ret_t3") > 0)) / len(sub_t3) * 100) if len(sub_t3) > 0 else None

        # T+5
        sub_t5 = sub.filter(pl.col("fwd_ret_t5").is_not_null())
        mean_t5 = sub_t5["fwd_ret_t5"].mean() if len(sub_t5) > 0 else None
        win_t5 = (len(sub_t5.filter(pl.col("fwd_ret_t5") > 0)) / len(sub_t5) * 100) if len(sub_t5) > 0 else None

        fmt_m_t1 = f"{mean_t1*100:>+9.2f}%" if mean_t1 is not None else f"{'PENDING':>10}"
        fmt_w_t1 = f"{win_t1:>11.1f}%" if win_t1 is not None else f"{'PENDING':>12}"
        fmt_m_t3 = f"{mean_t3*100:>+9.2f}%" if mean_t3 is not None else f"{'PENDING':>10}"
        fmt_w_t3 = f"{win_t3:>11.1f}%" if win_t3 is not None else f"{'PENDING':>12}"
        fmt_m_t5 = f"{mean_t5*100:>+9.2f}%" if mean_t5 is not None else f"{'PENDING':>10}"
        fmt_w_t5 = f"{win_t5:>11.1f}%" if win_t5 is not None else f"{'PENDING':>12}"

        print(f"{st:<16} | {n:<6} | {fmt_m_t1} | {fmt_w_t1} | {fmt_m_t3} | {fmt_w_t3} | {fmt_m_t5} | {fmt_w_t5}")

        md_m_t1 = f"{mean_t1*100:+.2f}%" if mean_t1 is not None else "PENDING"
        md_w_t1 = f"{win_t1:.1f}%" if win_t1 is not None else "PENDING"
        md_m_t3 = f"{mean_t3*100:+.2f}%" if mean_t3 is not None else "PENDING"
        md_w_t3 = f"{win_t3:.1f}%" if win_t3 is not None else "PENDING"
        md_table += f"| **{st}** | {n} | {md_m_t1} | {md_w_t1} | {md_m_t3} | {md_w_t3} |\n"

    print("="*112)
    return "### 🧠 PART 2: 各大状态未来收益率与胜率表\n" + md_table

def run_part3_confidence_tier_audit(df: pl.DataFrame) -> str:
    """PART 3: 置信度梯队收益单调性验证"""
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
            m_t1 = sub_t1["fwd_ret_t1"].mean() if len(sub_t1) > 0 else None
            w_t1 = (len(sub_t1.filter(pl.col("fwd_ret_t1") > 0)) / len(sub_t1) * 100) if len(sub_t1) > 0 else None

            sub_t3 = cell.filter(pl.col("fwd_ret_t3").is_not_null())
            m_t3 = sub_t3["fwd_ret_t3"].mean() if len(sub_t3) > 0 else None
            w_t3 = (len(sub_t3.filter(pl.col("fwd_ret_t3") > 0)) / len(sub_t3) * 100) if len(sub_t3) > 0 else None

            fmt_m_t1 = f"{m_t1*100:>+12.2f}%" if m_t1 is not None else f"{'PENDING':>13}"
            fmt_w_t1 = f"{w_t1:>11.1f}%" if w_t1 is not None else f"{'PENDING':>12}"
            fmt_m_t3 = f"{m_t3*100:>+12.2f}%" if m_t3 is not None else f"{'PENDING':>13}"
            fmt_w_t3 = f"{w_t3:>11.1f}%" if w_t3 is not None else f"{'PENDING':>12}"

            print(f"{st:<14} | {tr:<22} | {n:<6} | {fmt_m_t1} | {fmt_w_t1} | {fmt_m_t3} | {fmt_w_t3}")

            md_m_t1 = f"{m_t1*100:+.2f}%" if m_t1 is not None else "PENDING"
            md_w_t1 = f"{w_t1:.1f}%" if w_t1 is not None else "PENDING"
            md_m_t3 = f"{m_t3*100:+.2f}%" if m_t3 is not None else "PENDING"
            md_w_t3 = f"{w_t3:.1f}%" if w_t3 is not None else "PENDING"
            md_table += f"| **{st}** | {tr} | {n} | {md_m_t1} | {md_w_t1} | {md_m_t3} | {md_w_t3} |\n"
        print("-" * len(header))

    print("="*112)
    return "### 🎯 PART 3: 置信度梯队收益单调性验证表\n" + md_table

def main():
    files = find_factor_files()
    if not files:
        print("❌ 未在 'data/output' 目录找到任何因子 Parquet 文件！Alpha 验证程序终止。")
        return

    print(f"🔍 [Alpha Validation Engine] 搜寻到 {len(files)} 个因子文件，加载进行联合 Alpha 验证...", flush=True)
    
    # 汇总 data/output 下所有的因子 Parquet 文件
    dfs = [pl.read_parquet(f) for f in files]
    factor_df = pl.concat(dfs)

    # 1. 确保 Go 提取内核存在
    ensure_fetcher_core()

    # 2. 动态获取全市场最新 K 线
    codes = factor_df["code"].unique().to_list()
    kline_csv = "data/temp_chunks/alpha_validation_kline.csv"
    fetch_forward_klines(codes, kline_csv)

    # 3. 拼接推导 T+1 ~ T+10 Forward Returns (带严格时间轴防线)
    validated_df, latest_kline_date = calculate_forward_returns_with_guard(factor_df, kline_csv)

    # 4. 执行 4 大审计面板
    md0 = run_part0_alignment_audit(validated_df, latest_kline_date)
    md1 = run_part1_rank_ic_audit(validated_df)
    md2 = run_part2_state_returns_audit(validated_df)
    md3 = run_part3_confidence_tier_audit(validated_df)

    # 5. 输出 Markdown 至 GitHub Step Summary
    full_markdown = f"# 📈 TDX High-Freq Behavior Factors: Timestamp-Protected Alpha Report\n\n" + md0 + "\n" + md1 + "\n" + md2 + "\n" + md3
    append_to_github_summary(full_markdown)

    # 清理临时文件
    if os.path.exists(kline_csv):
        os.remove(kline_csv)

    print("\n🏁 [Alpha Validation Engine] 严格时间轴 Alpha 收益率与 Rank IC 验证完成。\n", flush=True)

if __name__ == "__main__":
    main()
