import polars as pl
import datetime

def generate_standard_grid() -> pl.DataFrame:
    time_slots = []
    start = datetime.datetime.strptime("09:30:00", "%H:%M:%S")
    for i in range(120):
        t = start + datetime.timedelta(minutes=i)
        time_slots.append(t.strftime("%H:%M:%S"))
    start = datetime.datetime.strptime("13:00:00", "%H:%M:%S")
    for i in range(120):
        t = start + datetime.timedelta(minutes=i)
        time_slots.append(t.strftime("%H:%M:%S"))
    return pl.DataFrame({"minute": time_slots})

def perform_micro_binning(ticks_df, adv_1m, atr_1m, eps_vol=1e-5, eps_res=1e-5):
    continuous_df = ticks_df.filter(
        ((pl.col("time") >= "09:30:00") & (pl.col("time") < "11:30:00")) |
        ((pl.col("time") >= "13:00:00") & (pl.col("time") < "15:00:00"))
    ).with_columns(pl.col("time").str.slice(0, 5).add(":00").alias("minute"))
    
    if continuous_df.is_empty():
        return pl.DataFrame()

    stock_code = continuous_df["code"][0]
    trade_date = continuous_df["date"][0]

    agg_df = continuous_df.group_by("minute").agg([
        pl.col("price").first().alias("open"), pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"), pl.col("price").last().alias("close"),
        (pl.col("price") * pl.col("volume")).sum().alias("amount_sum"),
        pl.col("volume").sum().alias("volume_sum"),
        pl.col("volume").filter(pl.col("direction") == 0).sum().alias("buy_vol"),
        pl.col("volume").filter(pl.col("direction") == 1).sum().alias("sell_vol"),
        pl.col("volume").filter(pl.col("direction") == 2).sum().alias("neutral_vol"),
        pl.len().alias("trade_count")
    ]).with_columns((pl.col("amount_sum") / (pl.col("volume_sum") + 1e-8)).alias("vwap"))

    aligned_df = generate_standard_grid().join(agg_df, on="minute", how="left")
    
    aligned_df = aligned_df.with_columns([
        pl.col("volume_sum").fill_null(0).alias("volume_sum"), pl.col("buy_vol").fill_null(0).alias("buy_vol"),
        pl.col("sell_vol").fill_null(0).alias("sell_vol"), pl.col("neutral_vol").fill_null(0).alias("neutral_vol"),
        pl.col("trade_count").fill_null(0).alias("trade_count"), pl.col("close").forward_fill().alias("close")
    ]).with_columns([
        pl.col("open").fill_null(pl.col("close")).alias("open"), pl.col("high").fill_null(pl.col("close")).alias("high"),
        pl.col("low").fill_null(pl.col("close")).alias("low"), pl.col("vwap").fill_null(pl.col("close")).alias("vwap"),
        (pl.col("volume_sum") > 0).alias("has_trade")
    ]).with_columns([
        (pl.col("has_trade") == False).alias("is_empty_bin"),
        pl.col("has_trade").alias("price_return_valid"),
        (pl.col("buy_vol") + pl.col("sell_vol") > 0).alias("direction_valid")
    ])

    aligned_df = aligned_df.with_columns([
        pl.when(pl.col("has_trade")).then((pl.col("close") / pl.col("open")) - 1.0).otherwise(0.0).alias("price_return"),
        pl.when(pl.col("has_trade")).then((pl.col("high") - pl.col("low")) / (pl.col("vwap") + 1e-8)).otherwise(0.0).alias("range")
    ]).with_columns([
        ((pl.col("buy_vol") - pl.col("sell_vol")) / (pl.col("volume_sum") + eps_vol)).alias("imbalance"),
        (pl.col("buy_vol") / (pl.col("volume_sum") + eps_vol)).alias("buy_ratio"),
        (pl.col("sell_vol") / (pl.col("volume_sum") + eps_vol)).alias("sell_ratio"),
        ((pl.col("volume_sum") - adv_1m) / (adv_1m * 0.5 + eps_vol)).alias("effort_z"),
        (pl.col("price_return") / (atr_1m + eps_res)).alias("result_z")
    ]).with_columns([
        (pl.col("effort_z") / (pl.col("result_z").abs() + eps_res)).alias("effort_result_ratio"),
        pl.lit(stock_code).alias("code"), pl.lit(trade_date).alias("trade_date")
    ])
    return aligned_df
