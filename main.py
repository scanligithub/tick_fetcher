# FILE: main.py
import os
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from src.utils import get_recent_trading_days, convert_csv_to_parquet

def compile_go_engine():
    """编译高性能提取内核"""
    print("🛠️  正在编译高性能 Go 提取内核...")
    cmd = ["go", "build", "-o", "fetcher_core", "src/fetcher.go"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        print(f"❌ 编译失败: {res.stderr.decode('utf-8')}")
        sys.exit(1)
    print("✅ 内核编译成功。")

def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

def get_stock_list():
    """拉取股票清单"""
    if not os.path.exists("stock_list.json"):
        print("📋 正在获取最新的 A 股标的清单...")
        subprocess.run(["./fetcher_core", "-mode=list"], check=True)
    
    with open("stock_list.json", "r") as f:
        return json.load(f)

def fetch_chunk_ticks(chunk_idx, codes_subset, date_str):
    """提取单个标的分片"""
    csv_path = f"temp_chunk_{chunk_idx}.csv"
    codes_str = ",".join(codes_subset)
    cmd = [
        "./fetcher_core",
        "-mode=fetch",
        f"-codes={codes_str}",
        f"-date={date_str}",
        f"-out={csv_path}"
    ]
    subprocess.run(cmd, check=True)
    return csv_path

def main():
    compile_go_engine()
    config = load_config()
    
    # 自动加载/获取股票
    all_stocks = get_stock_list()
    codes = [s["code"] for s in all_stocks]
    
    if config.get("test_mode", False):
        print("⚠️ 运行模式：【测试模式】（仅测试少量个股）")
        codes = config.get("test_stocks", ["SZ000001"])
    
    # 计算待爬取的天数线
    days = get_recent_trading_days(config.get("days_to_fetch", 3))
    print(f"📅 本次计划提取的历史交易日跨度: {days}")
    
    os.makedirs("output", exist_ok=True)
    
    # 将股票分配给各个 Go 进程并发执行 (分片大小)
    chunk_size = 50
    chunks = [codes[i:i + chunk_size] for i in range(0, len(codes), chunk_size)]
    
    for date_str in days:
        print(f"\n🚀 开始采集交易日 [{date_str}] 的全天完整物理 Tick...")
        
        # 线程池调度
        with ThreadPoolExecutor(max_workers=config.get("concurrency", 4)) as executor:
            futures = []
            for idx, subset in enumerate(chunks):
                futures.append(executor.submit(fetch_chunk_ticks, idx, subset, date_str))
            
            chunk_files = []
            for f in futures:
                chunk_files.append(f.result())
                
        # 将各进程导出的临时分片 CSV 合并
        full_csv = f"temp_all_{date_str}.csv"
        print(f"🧱 正在合并历史分片...")
        with open(full_csv, "w") as outfile:
            for idx, f_path in enumerate(chunk_files):
                if not os.path.exists(f_path):
                    continue
                with open(f_path, "r") as infile:
                    header = infile.readline()
                    if idx == 0:
                        outfile.write(header)
                    for line in infile:
                        outfile.write(line)
                os.remove(f_path) # 物理清理临时 CSV 碎片
                
        # 转换为高规格 Parquet 
        parquet_path = f"output/ticks_{date_str}.parquet"
        print(f"💾 正在对合并数据进行 Arrow 强类型转储并实施 ZSTD 物理压缩...")
        success = convert_csv_to_parquet(full_csv, parquet_path)
        
        if success:
            file_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
            print(f"✅ 日期 [{date_str}] 的 Tick 高频数据库落盘成功: {parquet_path} (物理大小: {file_size_mb:.2f} MB)")
            os.remove(full_csv)
        else:
            print(f"⚠️ 日期 [{date_str}] 未抓取到有效交易信息。")

    print("\n🏁 全量物理分笔高频同步工程任务已圆满结束。")

if __name__ == "__main__":
    main()
