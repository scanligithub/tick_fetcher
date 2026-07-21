import os
import sys
import glob
import yaml
import json
import subprocess
import polars as pl
from concurrent.futures import ThreadPoolExecutor

# 加载路径
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
    print("🛠️ 编译高性能 Go 提取内核...")
    cmd = ["go", "build", "-o", "fetcher_core", "src/go_fetcher/fetcher.go"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        print(f"❌ 编译失败: {res.stderr.decode('utf-8')}")
        sys.exit(1)
    print("✅ 内核编译成功。")

def prepare_stock_list(server: str):
    if not os.path.exists("stock_list.json"):
        print("📋 拉取全市场 A 股名单...")
        subprocess.run(["./fetcher_core", "-mode=list", f"-server={server}"], check=True)
    with open("stock_list.json", "r") as f:
        return json.load(f)

def run_chunk_pipeline(chunk_idx: int, codes: list, date_str: str, server: str, settings: dict, factors: dict) -> pl.DataFrame:
    """单个分片数据拉取 -> Polars 清洗 -> 微窗聚合。全部在内存中流式闭环，不留硬盘脏数据。"""
    csv_path = f"data/temp_chunks/chunk_{chunk_idx}.csv"
    codes_str = ",".join(codes)
    
    # 1. 抓取原始数据
    subprocess.run([
        "./fetcher_core",
        "-mode=fetch",
        f"-codes={codes_str}",
        f"-date={date_str}",
        f"-out={csv_path}",
        f"-server={server}"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 100:
        return pl.DataFrame()

    # 2. 载入 Layer 0 物理清洗
    raw_df = clean_raw_ticks(csv_path)
    os.remove(csv_path) # 内存读取后立刻物理删除临时原始 CSV 碎片

    results = []
    # 针对每只股票执行闭环计算
    for code in codes:
        single_ticks = raw_df.filter(pl.col("code") == code)
        if single_ticks.is_empty():
            continue
        
        # 极轻量化自愈：计算日线常数
        prices = single_ticks["price"]
        prev_close = float(prices[0]) # 暂以开盘第一笔作为前收基准自愈
        
        # 3. 计算 Layer 2 Auction
        auction_features = extract_auction_features(single_ticks, prev_close)
        
        # 4. 计算 Layer 3 Limit Regime
        limit_features = calculate_limit_regime(
            single_ticks, 
            prev_close, 
            factors["limit_thresholds"]["limit_up_pct"], 
            factors["limit_thresholds"]["limit_down_pct"]
        )
        
        # 5. 执行 240 分钟聚合微窗
        # 参数基准默认使用模拟常数
        aligned_bins = perform_micro_binning(
            single_ticks, 
            adv_1m=1000.0, 
            atr_1m=0.01,
            eps_vol=float(factors["epsilon"]["vol"]),
            eps_res=float(factors["epsilon"]["res"])
        )
        
        if aligned_bins.is_empty():
            continue

        # 6. 生成日级宽表原子指标
        daily_fact = aggregate_to_daily_row(
            aligned_bins, 
            limit_features, 
            auction_features, 
            factors["data_quality_weights"]
        )
        results.append(daily_fact)

    if len(results) == 0:
        return pl.DataFrame()
        
    return pl.concat(results)

def main():
    settings, factors = load_configs()
    compile_go_core()
    
    # 获取通达信优质服务器IP并旋转分配
    servers = settings.get("tdx_servers", ["119.147.212.81:7709"])
    server_main = servers[0]
    
    all_stocks = prepare_stock_list(server_main)
    codes = [s["code"] for s in all_stocks]
    
    if settings.get("test_mode", False):
        print("⚠️  运行状态：【测试运行模式】")
        codes = settings.get("test_stocks", ["SZ000001"])
        
    # Actions 容错：从百度获取真实北京时间，避免 Github Runner 的虚拟机时区异常
    import requests, time
    resp = requests.head("https://www.baidu.com")
    bj_time = time.strptime(resp.headers["Date"], "%a, %d %b %Y %H:%M:%S GMT")
    bj_timestamp = time.mktime(bj_time) + 8 * 3600
    date_str = time.strftime("%Y%m%d", time.localtime(bj_timestamp))
    
    # 排除周末不提取
    tm_wday = time.localtime(bj_timestamp).tm_wday
    if tm_wday >= 5:
        print("☕ 周末闭市，因子计算流水线挂起。")
        sys.exit(0)

    print(f"📅 本次处理的交易日期: {date_str}")
    
    os.makedirs("data/temp_chunks", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    
    chunk_size = settings.get("chunk_size", 500)
    chunks = [codes[i:i + chunk_size] for i in range(0, len(codes), chunk_size)]
    
    daily_results = []
    concurrency = settings.get("concurrency", 4)
    
    print(f"🚀 开始流水线并发作业，分片大小：{chunk_size}，并发度：{concurrency}...")
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for idx, subset in enumerate(chunks):
            # 将不同的服务器 IP 旋转分配给并发进程，突破单网关流量限制
            srv = servers[idx % len(servers)]
            futures.append(executor.submit(run_chunk_pipeline, idx, subset, date_str, srv, settings, factors))
            
        for f in futures:
            res_df = f.result()
            if not res_df.is_empty():
                daily_results.append(res_df)

    if len(daily_results) == 0:
        print("⚠️ 今天未拉取到任何有效高频数据。")
        sys.exit(0)

    # 7. 合并所有分片并附加 Context (本期采用模拟 Mock context, 生产中直接从 data/daily_kline 读取)
    final_fact_df = pl.concat(daily_results)
    
    # 模拟外部读入的 K 线位置
    context_df = pl.DataFrame({
        "code": final_fact_df["code"],
        "pp_20": [0.3] * len(final_fact_df),
        "pp_60": [0.4] * len(final_fact_df),
        "bias_20": [0.01] * len(final_fact_df),
        "rs_5": [0.02] * len(final_fact_df)
    })
    
    final_fact_df = final_fact_df.join(context_df, on="code", how="left")
    
    # 8. 映射 Layer 5 高阶行为画像得分
    output_df = evaluate_behavior_scores(final_fact_df)
    
    # 9. 高度压缩落盘 (ZSTD Parquet)
    out_file = f"data/output/factors_{date_str}.parquet"
    output_df.write_parquet(out_file, compression="zstd")
    
    file_size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"🏁 任务圆满结束。日级因子文件已落盘: {out_file} (最终大小: {file_size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
