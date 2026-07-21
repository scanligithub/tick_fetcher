import os
import sys
import json
import subprocess
import polars as pl
import yaml
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.engine.tick_cleaner import clean_raw_ticks
from src.engine.micro_binner import perform_micro_binning
from src.engine.auction_features import extract_auction_features
from src.engine.limit_regime import calculate_limit_regime
from src.engine.daily_aggregator import aggregate_to_daily_row
from src.engine.behavior_scores import evaluate_behavior_scores

def load_configs():
    with open("config/settings.yaml", "r") as f:
        settings = yaml.safe_load(f)
    with open("config/factor_config.yaml", "r") as f:
        factors = yaml.safe_load(f)
    return settings, factors

def compile_go_core():
    print("🛠  正在编译 Go 核心网关...", flush=True)
    cmd = ["go", "build", "-o", "fetcher_core", "src/go_fetcher/fetcher.go"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        print(f"❌ 编译失败: {res.stderr.decode('utf-8')}", flush=True)
        sys.exit(1)
    print("✅ 核心网关编译成功。", flush=True)

def prepare_stock_list():
    if os.path.exists("stock_list.json"):
        with open("stock_list.json", "r") as f:
            return json.load(f)

    print("📋 正在抓取全市场标的...", flush=True)
    res = subprocess.run(["./fetcher_core", "-mode=list"], capture_output=True, text=True)
    if res.returncode == 0:
        print("✅ 成功完成全市场标的初始化。", flush=True)
        with open("stock_list.json", "r") as f:
            return json.load(f)
    else:
        print(f"❌ 错误：Go 网关初始化失败: {res.stderr.strip()}", flush=True)
        sys.exit(1)

def run_chunk_pipeline(chunk_idx: int, codes: list, date_str: str, settings: dict, factors: dict) -> pl.DataFrame:
    csv_path = f"data/temp_chunks/chunk_{chunk_idx}.csv"
    codes_str = ",".join(codes)
    
    print(f"▶️  [分片 {chunk_idx}] 启动，调度股票数量: {len(codes)}...", flush=True)
    
    try:
        res = subprocess.run([
            "./fetcher_core", "-mode=fetch", f"-codes={codes_str}", f"-date={date_str}", f"-out={csv_path}"
        ], capture_output=True, text=True, timeout=180)
        
        if res.returncode != 0:
            print(f"[分片 {chunk_idx} 异常] Go 进程失败: {res.stderr}", flush=True)
            return pl.DataFrame()
    except subprocess.TimeoutExpired:
        print(f"[分片 {chunk_idx} 超时] 该批次任务响应超时，强行截断。", flush=True)
        return pl.DataFrame()
    
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 100:
        return pl.DataFrame()

    print(f"📥 [分片 {chunk_idx}] 数据拉取完毕。启动 Polars 聚合清洗...", flush=True)
    raw_df = clean_raw_ticks(csv_path)
    os.remove(csv_path)

    results = []
    for code in codes:
        single_ticks = raw_df.filter(pl.col("code") == code)
        if single_ticks.is_empty(): continue
        
        prev_close = float(single_ticks["price"][0])
        auction_features = extract_auction_features(single_ticks, prev_close)
        limit_features = calculate_limit_regime(
            single_ticks, prev_close, factors["limit_thresholds"]["limit_up_pct"], factors["limit_thresholds"]["limit_down_pct"]
        )
        aligned_bins = perform_micro_binning(
            single_ticks, adv_1m=1000.0, atr_1m=0.01, eps_vol=float(factors["epsilon"]["vol"]), eps_res=float(factors["epsilon"]["res"])
        )
        if aligned_bins.is_empty(): continue

        daily_fact = aggregate_to_daily_row(aligned_bins, limit_features, auction_features, factors["data_quality_weights"])
        results.append(daily_fact)

    print(f"⏹️  [分片 {chunk_idx}] 处理完毕。解析标的数量: {len(results)} / {len(codes)}", flush=True)
    
    if len(results) == 0:
        return pl.DataFrame()
    return pl.concat(results)

def main():
    os.makedirs("data/temp_chunks", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    
    settings, factors = load_configs()
    compile_go_core()
    
    all_stocks = prepare_stock_list()
    codes = [s["code"] for s in all_stocks]
    
    if settings.get("test_mode", False):
        codes = settings.get("test_stocks", ["SZ000001"])
        
    import requests, time
    resp = requests.head("https://www.baidu.com")
    bj_time_struct = time.strptime(resp.headers["Date"], "%a, %d %b %Y %H:%M:%S GMT")
    bj_timestamp = time.mktime(bj_time_struct) + 8 * 3600
    bj_time = time.localtime(bj_timestamp)

    if bj_time.tm_hour < 16:
        bj_timestamp -= 24 * 3600
        bj_time = time.localtime(bj_timestamp)

    while bj_time.tm_wday >= 5:
        bj_timestamp -= 24 * 3600
        bj_time = time.localtime(bj_timestamp)

    date_str = time.strftime("%Y%m%d", bj_time)
    print(f"📅 本日数据结算目标日期: {date_str}", flush=True)
    
    chunk_size = settings.get("chunk_size", 500)
    chunks = [codes[i:i + chunk_size] for i in range(0, len(codes), chunk_size)]
    
    daily_results = []
    concurrency = settings.get("concurrency", 4)
    
    print(f"🔥 启动并发任务线: 并发度={concurrency}，总批次={len(chunks)}", flush=True)
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for idx, subset in enumerate(chunks):
            futures.append(executor.submit(run_chunk_pipeline, idx, subset, date_str, settings, factors))
        for f in futures:
            res_df = f.result()
            if res_df is not None and not res_df.is_empty():
                daily_results.append(res_df)

    if len(daily_results) == 0:
        print("❌ 本日全市场高频清洗未能产出数据。", flush=True)
        sys.exit(0)

    final_fact_df = pl.concat(daily_results)
    
    context_df = pl.DataFrame({
        "code": final_fact_df["code"],
        "pp_20": [0.3] * len(final_fact_df),
        "pp_60": [0.4] * len(final_fact_df),
        "bias_20": [0.01] * len(final_fact_df),
        "rs_5": [0.02] * len(final_fact_df)
    })
    final_fact_df = final_fact_df.join(context_df, on="code", how="left")
    
    output_df = evaluate_behavior_scores(final_fact_df)
    
    out_file = f"data/output/factors_{date_str}.parquet"
    output_df.write_parquet(out_file, compression="zstd")
    print(f"🏁 日级高维特征落盘成功: {out_file}", flush=True)

if __name__ == "__main__":
    main()
