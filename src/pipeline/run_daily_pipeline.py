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
from src.engine.price_context import calculate_price_context
from src.engine.market_context import calculate_market_relative_strength

def load_configs():
    with open("config/settings.yaml", "r") as f:
        settings = yaml.safe_load(f)
    with open("config/factor_config.yaml", "r") as f:
        factors = yaml.safe_load(f)
    return settings, factors

def compile_go_core():
    print("🛠  正在编译 Go 核心网关...", flush=True)
    root_dir = os.getcwd()
    go_dir = os.path.join(root_dir, "src", "go_fetcher")
    out_path = os.path.join(root_dir, "fetcher_core")
    
    cmd = ["go", "build", "-o", out_path, "fetcher.go"]
    res = subprocess.run(cmd, cwd=go_dir, capture_output=True)
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
        print(f"❌ 错误：Go 网关初始化失败", flush=True)
        sys.exit(1)

def run_chunk_pipeline(chunk_idx: int, codes: list, date_str: str, settings: dict, factors: dict) -> pl.DataFrame:
    tick_csv = f"data/temp_chunks/chunk_{chunk_idx}_ticks.csv"
    kline_csv = f"data/temp_chunks/chunk_{chunk_idx}_kline.csv"
    index_csv = "data/temp_chunks/index_kline.csv" # 全局复用的大盘 K 线
    codes_str = ",".join(codes)
    
    print(f"▶️  [分片 {chunk_idx}] 启动，调度股票数量: {len(codes)}...", flush=True)
    
    # 1. 抓取分时 Tick
    subprocess.run(["./fetcher_core", "-mode=fetch", f"-codes={codes_str}", f"-date={date_str}", f"-out={tick_csv}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # 2. 抓取日线 K 线
    subprocess.run(["./fetcher_core", "-mode=kline", f"-codes={codes_str}", f"-out={kline_csv}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if not os.path.exists(tick_csv) or os.path.getsize(tick_csv) < 100:
        return pl.DataFrame()

    print(f"📥 [分片 {chunk_idx}] 数据拉取完毕。启动 Polars 聚合清洗...", flush=True)
    raw_df = clean_raw_ticks(tick_csv)
    
    # 3. 计算真实的个股位置(Price Context)和相对强弱(Market Context)
    price_ctx_df = calculate_price_context(kline_csv)
    market_ctx_df = calculate_market_relative_strength(kline_csv, index_csv)
    
    # 合并真实的 Context
    real_context_df = pl.DataFrame({"code": codes})
    if not price_ctx_df.is_empty():
        real_context_df = real_context_df.join(price_ctx_df, on="code", how="left")
    if not market_ctx_df.is_empty():
        real_context_df = real_context_df.join(market_ctx_df, on="code", how="left")
        
    # 补全可能缺失的指标默认值
    real_context_df = real_context_df.with_columns([
        pl.col("pp_20").fill_null(0.5), pl.col("pp_60").fill_null(0.5),
        pl.col("bias_20").fill_null(0.0), pl.col("rs_5").fill_null(0.0),
        pl.col("atr_10").fill_null(0.1) # 默认 ATR 防止除 0
    ])

    if os.path.exists(tick_csv): os.remove(tick_csv)
    if os.path.exists(kline_csv): os.remove(kline_csv)

    results = []
    for code in codes:
        single_ticks = raw_df.filter(pl.col("code") == code)
        if single_ticks.is_empty(): continue
        
        prev_close = float(single_ticks["price"][0])
        auction_features = extract_auction_features(single_ticks, prev_close)
        limit_features = calculate_limit_regime(
            single_ticks, prev_close, factors["limit_thresholds"]["limit_up_pct"], factors["limit_thresholds"]["limit_down_pct"]
        )
        
        # 从真实的 Context 中提取该股票的 ATR 波动率传给微窗计算
        stock_ctx = real_context_df.filter(pl.col("code") == code)
        atr_1m_val = float(stock_ctx["atr_10"][0] / 240.0) if not stock_ctx.is_empty() and stock_ctx["atr_10"][0] is not None else 0.01
        
        aligned_bins = perform_micro_binning(
            single_ticks, adv_1m=1000.0, atr_1m=max(atr_1m_val, 0.001), 
            eps_vol=float(factors["epsilon"]["vol"]), eps_res=float(factors["epsilon"]["res"])
        )
        if aligned_bins.is_empty(): continue

        daily_fact = aggregate_to_daily_row(aligned_bins, limit_features, auction_features, factors["data_quality_weights"])
        results.append(daily_fact)

    print(f"⏹️  [分片 {chunk_idx}] 处理完毕。解析标的数量: {len(results)} / {len(codes)}", flush=True)
    
    if len(results) == 0:
        return pl.DataFrame()
        
    # 将该分片的 Tick 事实数据与真实的 Context 数据进行横向拼合
    chunk_fact_df = pl.concat(results)
    chunk_final_df = chunk_fact_df.join(real_context_df, on="code", how="left")
    
    return chunk_final_df

def main():
    os.makedirs("data/temp_chunks", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    
    settings, factors = load_configs()
    compile_go_core()
    
    # 提前抓取大盘基准日线 (上证指数 SH999999) 供所有分片复用算相对强弱
    print("📈 正在获取大盘基准 K 线 (上证指数)...", flush=True)
    subprocess.run(["./fetcher_core", "-mode=kline", "-codes=SH999999", "-out=data/temp_chunks/index_kline.csv"], stdout=subprocess.DEVNULL)
    
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
    
    # 🧠 现在，送入 evaluate_behavior_scores 的是拥有【真实灵魂 (PP/RS)】的 DataFrame
    print("🧠 正在调制连续型机构高阶行为映射模型...", flush=True)
    output_df = evaluate_behavior_scores(final_fact_df)
    
    out_file = f"data/output/factors_{date_str}.parquet"
    output_df.write_parquet(out_file, compression="zstd")
    print(f"🏁 日级高维特征落盘成功: {out_file}", flush=True)

if __name__ == "__main__":
    main()
