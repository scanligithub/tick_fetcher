import os
import sys
import json
import subprocess
import polars as pl
import yaml
import time
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
    print("🛠️  正在编译高性能 Go 核心网关...", flush=True)
    root_dir = os.getcwd()
    go_dir = os.path.join(root_dir, "src", "go_fetcher")
    out_path = os.path.join(root_dir, "fetcher_core")
    
    cmd = ["go", "build", "-o", out_path, "fetcher.go"]
    res = subprocess.run(cmd, cwd=go_dir, capture_output=True)
    if res.returncode != 0:
        print(f"❌ Go 内核编译失败: {res.stderr.decode('utf-8')}", flush=True)
        sys.exit(1)
    print("✅ Go 核心网关编译成功。", flush=True)

def prepare_stock_list():
    if os.path.exists("stock_list.json"):
        with open("stock_list.json", "r") as f:
            return json.load(f)

    print("📋 正在初始化全市场标的种子...", flush=True)
    res = subprocess.run(["./fetcher_core", "-mode=list"], capture_output=True, text=True)
    if res.returncode == 0:
        print("✅ 种子清单初始化落盘成功。", flush=True)
        with open("stock_list.json", "r") as f:
            return json.load(f)
    else:
        print("❌ 错误：Go 网关初始化种子失败", flush=True)
        sys.exit(1)

def run_chunk_pipeline(chunk_idx: int, codes: list, date_str: str, settings: dict, factors: dict) -> pl.DataFrame:
    tick_csv = f"data/temp_chunks/chunk_{chunk_idx}_ticks.csv"
    kline_csv = f"data/temp_chunks/chunk_{chunk_idx}_kline.csv"
    index_csv = "data/temp_chunks/index_kline.csv"
    codes_str = ",".join(codes)
    
    print(f"▶️  [分片 {chunk_idx}] 启动，调度股票数量: {len(codes)}...", flush=True)
    
    # 1. 抓取分时原始 Tick
    res_ticks = subprocess.run(["./fetcher_core", f"-mode=fetch", f"-codes={codes_str}", f"-date={date_str}", f"-out={tick_csv}"], capture_output=True, text=True)
    if res_ticks.returncode != 0:
         print(f"⚠️ [分片 {chunk_idx}] 原始分时拉取警告", flush=True)

    # 2. 抓取个股日 K 线（包含100日历史区间）
    res_kline = subprocess.run(["./fetcher_core", f"-mode=kline", f"-codes={codes_str}", f"-out={kline_csv}"], capture_output=True, text=True)
    if res_kline.returncode != 0:
         print(f"⚠️ [分片 {chunk_idx}] 历史K线拉取警告", flush=True)
         
    if not os.path.exists(tick_csv) or os.path.getsize(tick_csv) < 100:
        return pl.DataFrame()

    # 3. 运行清洗与历史上下文对齐
    raw_df = clean_raw_ticks(tick_csv)
    
    # 计算 T-1 静态历史状态指标 (物理隔離)
    price_ctx_df = calculate_price_context(kline_csv)
    market_ctx_df = calculate_market_relative_strength(kline_csv, index_csv)
    
    # 装配 T-1 相对强弱与位置状态特征
    real_context_df = pl.DataFrame({"code": codes})
    if not price_ctx_df.is_empty():
        real_context_df = real_context_df.join(price_ctx_df, on="code", how="left")
    if not market_ctx_df.is_empty():
        real_context_df = real_context_df.join(market_ctx_df, on="code", how="left")
        
    # 定义缺失特征的前置回退默认值（防止新股计算为空导致的整行丢失）
    fallback_map = {
        "adv_20_pre": 1000000.0,
        "atr_10_pre": 0.1,
        "low_20_pre": 1.0,
        "high_20_pre": 1.0,
        "low_60_pre": 1.0,
        "high_60_pre": 1.0,
        "close_pre": 1.0,
        "pp_20_pre": 0.5,
        "pp_60_pre": 0.5,
        "bias_20_pre": 0.0,
        "rs_5_pre": 0.0
    }
    for col_name, default_val in fallback_map.items():
        if col_name not in real_context_df.columns:
            real_context_df = real_context_df.with_columns(pl.lit(default_val).alias(col_name))
        else:
            real_context_df = real_context_df.with_columns(pl.col(col_name).fill_null(default_val))

    # 物理物理清理临时分片，防止 Actions 沙盒存储溢出
    if os.path.exists(tick_csv): os.remove(tick_csv)
    if os.path.exists(kline_csv): os.remove(kline_csv)

    # 4. 构建代码对齐计算字典并开始重采样及单行宽表推导
    context_records = real_context_df.to_dicts()
    context_map = {item["code"]: item for item in context_records}

    results = []
    for code in codes:
        single_ticks = raw_df.filter(pl.col("code") == code)
        if single_ticks.is_empty(): 
            continue
        
        stock_ctx = context_map.get(code, fallback_map)
        prev_close_val = float(stock_ctx.get("close_pre", 1.0))
        
        # 提取集合竞价特征
        auction_features = extract_auction_features(single_ticks, prev_close_val)
        
        # 提取涨跌停界限
        limit_features = calculate_limit_regime(
            single_ticks, prev_close_val, factors["limit_thresholds"]["limit_up_pct"], factors["limit_thresholds"]["limit_down_pct"]
        )
        
        # 240分钟微窗分箱计算
        atr_1m_val = float(stock_ctx["atr_10_pre"] / 240.0) if stock_ctx.get("atr_10_pre") else 0.001
        aligned_bins = perform_micro_binning(
            single_ticks, adv_1m=1000.0, atr_1m=max(atr_1m_val, 0.001), 
            eps_vol=float(factors["epsilon"]["vol"]), eps_res=float(factors["epsilon"]["res"])
        )
        if aligned_bins.is_empty(): 
            continue

        # 降维合并历史上下文生成因子行
        daily_fact = aggregate_to_daily_row(aligned_bins, limit_features, auction_features, factors["data_quality_weights"], stock_ctx)
        results.append(daily_fact)

    print(f"⏹️  [分片 {chunk_idx}] 结算完毕。解析有效标的: {len(results)} / {len(codes)}", flush=True)
    
    if len(results) == 0:
        return pl.DataFrame()
        
    chunk_fact_df = pl.concat(results)
    chunk_final_df = chunk_fact_df.join(real_context_df, on="code", how="left")
    return chunk_final_df

def main():
    os.makedirs("data/temp_chunks", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    
    settings, factors = load_configs()
    compile_go_core()
    
    # 1. 抓取基准大盘 (上证指数) K线
    print("📈 正在下载大盘基准日线 (SH000001)...", flush=True)
    res_index = subprocess.run(["./fetcher_core", "-mode=kline", "-codes=SH000001", "-out=data/temp_chunks/index_kline.csv"], capture_output=True, text=True)
    if res_index.returncode != 0:
        print(f"❌ 大盘 K 线拉取失败: {res_index.stderr}", flush=True)
        sys.exit(1)
    
    all_stocks = prepare_stock_list()
    codes = [s["code"] for s in all_stocks]
    
    if settings.get("test_mode", False):
        print("⚠️ 【警告】运行处于测试模式，将仅测试少量个股。", flush=True)
        codes = settings.get("test_stocks", ["SZ000001", "SH600519"])
        
    # 2. 定位结算日期（避开非交易日及交易时段）
    import requests
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
    print(f"📅 本日行为特征结算目标交易日: {date_str}", flush=True)
    
    chunk_size = settings.get("chunk_size", 500)
    chunks = [codes[i:i + chunk_size] for i in range(0, len(codes), chunk_size)]
    
    # 3. 启动多线程调度
    daily_results = []
    concurrency = settings.get("concurrency", 4)
    print(f"🔥 正在启动并发特征调制管道 [并发数={concurrency} | 总任务片={len(chunks)}]...", flush=True)
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for idx, subset in enumerate(chunks):
            futures.append(executor.submit(run_chunk_pipeline, idx, subset, date_str, settings, factors))
        for f in futures:
            res_df = f.result()
            if res_df is not None and not res_df.is_empty():
                daily_results.append(res_df)

    if len(daily_results) == 0:
        print("❌ 警告：未调制出任何个股有效日级行为事实。")
        sys.exit(0)

    # 4. 合并执行多维相对竞争置信度调制
    final_fact_df = pl.concat(daily_results)
    print("🧠 正在进行多维状态机概率调制与置信度校准...", flush=True)
    output_df = evaluate_behavior_scores(final_fact_df)
    
    # 5. ZSTD物理压缩落盘
    out_file = f"data/output/factors_{date_str}.parquet"
    output_df.write_parquet(out_file, compression="zstd")
    print(f"🏁 日级行为因子的特征宽表已安全落盘: {out_file}", flush=True)

if __name__ == "__main__":
    main()
