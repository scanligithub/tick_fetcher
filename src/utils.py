# FILE: src/utils.py
import requests
import time
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def get_real_beijing_time():
    """动态获取真实世界的北京时间，杜绝沙盒虚拟时钟干扰"""
    try:
        resp = requests.head("https://www.baidu.com", timeout=3)
        if resp.status_code == 200 and "Date" in resp.headers:
            t = time.strptime(resp.headers["Date"], "%a, %d %b %Y %H:%M:%S GMT")
            # 转换为东八区北京时间
            bj_timestamp = time.mktime(t) + 8 * 3600
            return time.localtime(bj_timestamp)
    except Exception:
        pass
    return time.localtime()

def get_recent_trading_days(n=3):
    """计算最近 N 个有效的历史交易日"""
    bj_time = get_real_beijing_time()
    curr_time = time.mktime(bj_time)
    
    # 若当前尚未收盘 (15:00点前)，则今日不参与计算，从昨日回溯
    if bj_time.tm_hour < 15:
        curr_time -= 24 * 3600
        
    days = []
    offset = 0
    while len(days) < n:
        t = time.localtime(curr_time - offset * 24 * 3600)
        # 排除周末 (周六=5, 周日=6)
        if t.tm_wday < 5:
            days.append(time.strftime("%Y%m%d", t))
        offset += 1
    return days

def convert_csv_to_parquet(csv_path, parquet_path):
    """将采集的原始 CSV 数据执行强类型对齐，并输出为极致压缩的 ZSTD Parquet"""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 100:
        return False
        
    # 定义高规格物理 Schema 
    arrow_schema = pa.schema([
        ('code', pa.string()),
        ('date', pa.string()),
        ('time', pa.string()),
        ('price', pa.float32()),       # 浮点单精度足够表达高频价格
        ('volume', pa.int32()),        # 单笔成交股数
        ('direction', pa.int8()),      # 方向：0=红/买，1=绿/卖，2=白/中
        ('order_num', pa.int32())      # 报单序号/单号
    ])
    
    # 物理映射列名
    df = pd.read_csv(csv_path)
    df.columns = ["code", "date", "time", "price", "volume", "direction", "order_num"]
    
    # 🚀 核心修复：显式强制转换 string 相关列为 str，防止 Pandas 自动推断为整型导致 PyArrow 类型不匹配
    df["code"] = df["code"].astype(str)
    df["date"] = df["date"].astype(str)
    df["time"] = df["time"].astype(str)
    
    # 数据安全类型转换
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float32")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int32")
    df["direction"] = pd.to_numeric(df["direction"], errors="coerce").fillna(2).astype("int8")
    df["order_num"] = pd.to_numeric(df["order_num"], errors="coerce").fillna(0).astype("int32")
    
    # 利用 Arrow 转换并写入列式数据，启动 ZSTD 高速算法
    table = pa.Table.from_pandas(df, schema=arrow_schema)
    pq.write_table(table, parquet_path, compression="zstd")
    return True
