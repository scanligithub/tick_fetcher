import os
import sys
import json
import subprocess
import polars as pl
import time
import requests

def get_beijing_time_date() -> str:
    """
    动态获取真实的北京时间，避开 Actions 环境的 UTC 偏差
    """
    try:
        resp = requests.head("https://www.baidu.com", timeout=3)
        if resp.status_code == 200 and "Date" in resp.headers:
            t = time.strptime(resp.headers["Date"], "%a, %d %b %Y %H:%M:%S GMT")
            bj_timestamp = time.mktime(t) + 8 * 3600
            bj_time = time.localtime(bj_timestamp)
    except Exception:
        bj_time = time.localtime()

    # 如果在下午 16:00 前运行，回退到前一天以确保有完整的收盘结算数据
    bj_timestamp = time.mktime(bj_time)
    if bj_time.tm_hour < 16:
        bj_timestamp -= 24 * 3600
        bj_time = time.localtime(bj_timestamp)

    # 避开周末
    while bj_time.tm_wday >= 5:
        bj_timestamp -= 24 * 3600
        bj_time = time.localtime(bj_timestamp)

    return time.strftime("%Y%m%d", bj_time)

def compile_core():
    print("🛠️  正在编译高性能 Go 核心提取网关...", flush=True)
    root_dir = os.getcwd()
    go_dir = os.path.join(root_dir, "src", "go_fetcher")
    out_path = os.path.join(root_dir, "fetcher_core")
    
    cmd = ["go", "build", "-o", out_path, "fetcher.go"]
    res = subprocess.run(cmd, cwd=go_dir, capture_output=True)
    if res.returncode != 0:
        print(f"❌ Go 网关编译失败: {res.stderr.decode('utf-8')}", flush=True)
        sys.exit(1)
    print("✅ Go 核心网关编译成功。", flush=True)

def run_physical_sanity_check(code: str, date_str: str):
    print(f"\n📢 开始对标的 [{code}] 在交易日 [{date_str}] 进行双端数据验尸...", flush=True)
    
    tick_csv = f"data/temp_chunks/sanity_ticks_{code}.csv"
    kline_csv = f"data/temp_chunks/sanity_kline_{code}.csv"
    
    # 1. 抓取分时原始 Tick
    subprocess.run([
        "./fetcher_core", 
        "-mode=fetch", 
        f"-codes={code}", 
        f"-date={date_str}", 
        f"-out={tick_csv}"
    ], capture_output=True)

    # 2. 抓取日K线
    subprocess.run([
        "./fetcher_core", 
        "-mode=kline", 
        f"-codes={code}", 
        f"-out={kline_csv}"
    ], capture_output=True)

    if not os.path.exists(tick_csv) or os.path.getsize(tick_csv) < 100:
        print(f"❌ 诊断失败：无法获取 [{code}] 的原始 Tick 文件或文件为空。", flush=True)
        return
    if not os.path.exists(kline_csv) or os.path.getsize(kline_csv) < 100:
        print(f"❌ 诊断失败：无法获取 [{code}] 的原始 K线 文件或文件为空。", flush=True)
        return

    try:
        # 3. 读取并分析 Ticks 数据
        # 强制声明以防 Polars 解析错位
        ticks_df = pl.read_csv(
            tick_csv,
            dtypes={
                "code": pl.String, "date": pl.String, "time": pl.String,
                "price": pl.Float64, "volume": pl.Int64, "status": pl.Int8, "number": pl.Int32
            }
        ).sort("time")

        # 4. 读取并分析 K线 数据
        kline_df = pl.read_csv(
            kline_csv,
            dtypes={
                "code": pl.String, "date": pl.String,
                "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
                "volume": pl.Float64, "amount": pl.Float64
            }
        )

        # 过滤出 $T$ 日的 K 线记录
        kline_row = kline_df.filter(pl.col("date") == date_str)
        if kline_row.is_empty():
            print(f"⚠️ 警告: K线数据中未发现当天 [{date_str}] 的记录，默认回退使用最后一行。", flush=True)
            kline_row = kline_df.tail(1)

        # 5. 提取原始价格事实
        k_open = kline_row["open"][0]
        k_high = kline_row["high"][0]
        k_low = kline_row["low"][0]
        k_close = kline_row["close"][0]

        t_first = ticks_df["price"].first()
        t_last = ticks_df["price"].last()
        t_min = ticks_df["price"].min()
        t_max = ticks_df["price"].max()
        
        # 提取 10:00:00 附近的 Tick 价格作为分时采样点
        t_1000_row = ticks_df.filter(pl.col("time") >= "10:00:00").head(1)
        t_1000 = t_1000_row["price"][0] if not t_1000_row.is_empty() else None

        # 计算同收盘事实价格比例
        price_ratio = k_close / t_last if t_last > 0 else 0.0

        # 6. 提取量能事实（三重证据）
        k_volume = kline_row["volume"][0]  # K 线成交量
        t_rows = ticks_df.height           # Tick 总行数
        
        t_volume_first = ticks_df["volume"].first()
        t_volume_last = ticks_df["volume"].last()
        t_volume_min = ticks_df["volume"].min()
        t_volume_max = ticks_df["volume"].max()

        sum_tick_vol = float(ticks_df["volume"].sum())  # 证据 A: 日内 Tick 总和 (S)
        last_tick_vol = float(t_volume_last)            # 证据 B: 最后一笔 Tick (L)

        # 7. 增长形态学差分计算
        # 计算相邻 Tick 的差分
        vol_array = ticks_df["volume"].to_numpy()
        diff_array = vol_array[1:] - vol_array[:-1]
        
        total_diffs = len(diff_array)
        pos_diff = int((diff_array > 0).sum())
        neg_diff = int((diff_array < 0).sum())
        zero_diff = int((diff_array == 0).sum())

        pos_pct = (pos_diff / total_diffs * 100) if total_diffs > 0 else 0.0
        neg_pct = (neg_diff / total_diffs * 100) if total_diffs > 0 else 0.0
        zero_pct = (zero_diff / total_diffs * 100) if total_diffs > 0 else 0.0

        # 增量差分绝对累计值
        diff_sum = float(t_volume_first + sum([d for d in diff_array if d > 0]))

        # 计算比值
        ratio_s_k = sum_tick_vol / k_volume if k_volume > 0 else 0.0
        ratio_l_k = last_tick_vol / k_volume if k_volume > 0 else 0.0
        ratio_diff_k = diff_sum / k_volume if k_volume > 0 else 0.0

        # 8. 交叉验证物理判定（Verdict）
        # 价格判定
        if 0.95 <= price_ratio <= 1.05:
            price_verdict = "ALIGNED (量纲完全一致)"
        elif 95.0 <= price_ratio <= 105.0:
            price_verdict = "X100 MISMATCH (K线放大100倍)"
        elif 950.0 <= price_ratio <= 1050.0:
            price_verdict = "X1000 MISMATCH (K线放大1000倍)"
        else:
            price_verdict = f"UNKNOWN SCALE (比例为 {price_ratio:.4f})"

        # 成交量语义判定 (三重证据交叉法)
        # 如果 L/K 接近 1 且 S/K 远大于 1 且几乎不出现负增长，则判定为 CUMULATIVE
        if 0.90 <= ratio_l_k <= 1.10 and ratio_s_k > 5.0 and neg_pct < 1.0:
            vol_semantics = "CUMULATIVE (日内累计量)"
        else:
            vol_semantics = "INCREMENTAL (单笔增量)"

        # 成交量单位判定
        if vol_semantics == "CUMULATIVE":
            ref_vol = last_tick_vol
        else:
            ref_vol = sum_tick_vol

        unit_ratio = ref_vol / k_volume if k_volume > 0 else 0.0
        if 0.90 <= unit_ratio <= 1.10:
            unit_verdict = "SAME (单位完全对齐)"
        elif 90.0 <= unit_ratio <= 110.0:
            unit_verdict = "TICK_X100 (Tick为股, K线为手 - 需K线×100)"
        elif 0.009 <= unit_ratio <= 0.011:
            unit_verdict = "KLINE_X100 (Tick为手, K线为股 - 需Tick×100)"
        else:
            unit_verdict = f"UNKNOWN UNIT RATIO (比例为 {unit_ratio:.4f})"

        # 9. 输出诊断 ASCII 报告表
        print("="*64)
        print("           TICK / KLINE PHYSICAL SANITY CHECK           ")
        print("="*64)
        print(f"Code: {code:<15} Date: {date_str:<15}")
        print("-"*64)
        print("[PRICE FACTS]")
        print(f"  Kline Open        : {k_open:<12.2f} | Kline Close   : {k_close:.2f}")
        print(f"  Kline High        : {k_high:<12.2f} | Kline Low     : {k_low:.2f}")
        print(f"  Tick First Price  : {t_first:<12.3f} | Tick Last Price: {t_last:.3f}")
        print(f"  Tick Min Price    : {t_min:<12.3f} | Tick Max Price : {t_max:.3f}")
        print(f"  Tick 10:00 Price  : {t_1000}")
        print(f"  Price Ratio (K/T) : {price_ratio:.6f}")
        print("-"*64)
        print("[VOLUME FACTS (TRIPLE-EVIDENCE)]")
        print(f"  Tick Rows         : {t_rows:<12}")
        print(f"  Tick Vol First    : {t_volume_first:<12} | Tick Vol Last : {t_volume_last}")
        print(f"  Tick Vol Min      : {t_volume_min:<12} | Tick Vol Max  : {t_volume_max}")
        print(f"  Sum(Tick Vol) [S] : {sum_tick_vol:<12.1f}")
        print(f"  Last(Tick Vol) [L]: {last_tick_vol:<12.1f}")
        print(f"  Reconstructed [D] : {diff_sum:<12.1f} (Sum of positive diffs)")
        print(f"  Kline Volume [K]  : {k_volume:<12.1f}")
        print(f"  S / K Ratio       : {ratio_s_k:<12.6f}")
        print(f"  L / K Ratio       : {ratio_l_k:<12.6f}")
        print(f"  D / K Ratio       : {ratio_diff_k:<12.6f}")
        print("-"*64)
        print("[GROWTH SHAPE ANALYSIS]")
        print(f"  Positive Diff     : {pos_diff:<6} ({pos_pct:>5.2f}%)")
        print(f"  Negative Diff     : {neg_diff:<6} ({neg_pct:>5.2f}%)")
        print(f"  Zero Diff         : {zero_diff:<6} ({zero_pct:>5.2f}%)")
        print("-"*64)
        print("[FINAL PHYSICAL VERDICT]")
        print(f"  Price Scale       : {price_verdict}")
        print(f"  Volume Semantics  : {vol_semantics}")
        print(f"  Volume Unit       : {unit_verdict}")
        print("="*64 + "\n")

    except Exception as e:
        print(f"❌ 解析过程中发生异常: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        # 清理临时文件，保持物理目录整洁
        if os.path.exists(tick_csv): os.remove(tick_csv)
        if os.path.exists(kline_csv): os.remove(kline_csv)

def main():
    os.makedirs("data/temp_chunks", exist_ok=True)
    compile_core()
    
    # 自动解析最可靠的测试交易日 (排除了未收盘时间与周末)
    target_date = get_beijing_time_date()
    
    # 选取极具代表性的两只南北标的进行物理交叉验证
    # SZ000001 (平安银行 - 深圳主板代表)
    # SH600000 (浦发银行 - 上海主板代表)
    test_stocks = ["SZ000001", "SH600000"]
    
    for code in test_stocks:
        run_physical_sanity_check(code, target_date)

if __name__ == "__main__":
    main()
