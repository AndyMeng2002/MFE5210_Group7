import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from jqdatasdk import auth, get_price, get_current_tick, get_trade_days
from tqdm import tqdm
import time
import warnings
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

warnings.filterwarnings("ignore")


# ============================================================
# 0. 全局常量 & 工具函数
# ------------------------------------------------------------

MORNING_START = "09:30:00"
MORNING_END = "11:30:00"
AFTERNOON_START = "13:00:00"
AFTERNOON_END = "15:00:00"

today_trade_record = pd.DataFrame()


# 三个小工具函数为了防止日期相关的问题报错
def get_trading_minutes(date_str: str) -> pd.DatetimeIndex:
    """返回一个交易日的所有有效分钟索引。"""
    m1 = pd.date_range(f"{date_str} {MORNING_START}", f"{date_str} {MORNING_END}", freq="1min")
    m2 = pd.date_range(f"{date_str} {AFTERNOON_START}", f"{date_str} {AFTERNOON_END}", freq="1min")
    return m1.append(m2)


def to_date(x):
    if isinstance(x, str):
        return pd.to_datetime(x).date()
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, datetime):
        return x.date()
    return x


def to_datetime(x):
    if isinstance(x, str):
        return pd.to_datetime(x)
    if isinstance(x, datetime):
        return pd.Timestamp(x)
    if isinstance(x, pd.Timestamp):
        return x
    return pd.Timestamp(x)

# ============================================================
# 1. 策略、持仓、环境类
# ------------------------------------------------------------
class StrategySetting:
    def __init__(self):
        self.start_date = None
        self.end_date = None
        self.mode = None                    # 采用"实盘"还是"回测"
        self.fee_rate = None 
        self.order_method = None            # 具体使用什么下单方式(正常按照volume下单、TWAP下单、VWAP下单)
        self.execution_params = None        # 采用算法执行模式所需的参数(TWAP/VWAP会用到)


class PortfolioInfo:
    def __init__(self, initial_cash: float = 1e8):
        self.positions = {}                 # 记录当前持仓,用字典{code:volume}的形式存储当前仓位
        self.available_cash = initial_cash  # 记录当前可用资金
        self.position_value = 0             # 记录当前持仓市值
        self.total_value = initial_cash     # 记录当前总资产
        self.return_until_today = 0.0       # 记录当前收益,(total_value-initial_cash)/initial_cash
        self.initial_cash = initial_cash    # 记录初始资金


class BacktestEnv:
    def __init__(self, strategy_setting: StrategySetting, portfolio_info: PortfolioInfo):
        self.strategy_setting = strategy_setting
        self.portfolio_info = portfolio_info
        self.today = strategy_setting.start_date
        self.yesterday = strategy_setting.start_date
        self.stock_pool = None
        self.benchmark = None
        self.trade_book = {}
        self.target_book = pd.DataFrame()
        self.position_book = pd.DataFrame()
        self.total_value_book = pd.DataFrame()
        self.tca_book = {}


# 实例化
setting = StrategySetting()
portfolio = PortfolioInfo(initial_cash=100_000_000)
env = BacktestEnv(setting, portfolio)

# ============================================================
# 2. 数据读取模块(本地 & 在线)
# ------------------------------------------------------------

# 本地分钟数据根目录
local_path = r"C:\Users\Dell\Desktop\算法交易\HS300_data"  # TODO 若您想在本地运行请改成本地分钟频数据地址

# 聚宽认证
# TODO 这个是我的账号,如果有需要可以更改成其他账号的电话号码和密码
# 本账号已开通线上获取数据权限，使用周期为一个月，后续过期需要更换有权限的账号否则无法线上获取数据
auth('13478135995', 'Andy20020826') 


def _code_to_jq(code: str) -> str:
    """
    函数功能：
        将标准的六位 A 股证券代码（如 '600000', '000001' 等）转换为聚宽系统所需的
        带有交易所后缀的格式（如 '600000.XSHG', '000001.XSHE'），以便于调用聚宽 API。
    参数说明：
        - code (str): 原始股票代码，必须为字符串形式的六位数字如 000001
    返回值：
        - str: 聚宽标准格式的股票代码，如 000001.XSHE
    """
    if code.startswith(("0", "3")):
        return code + ".XSHE"
    if code.startswith("6"):
        return code + ".XSHG"


def load_realtime_data(code: str) -> pd.DataFrame:
    """
    函数功能：
        实时获取某一支股票的最新行情数据，包括最新成交价、买一价和卖一价
        返回结构为带有时间戳索引的 Pandas DataFrame，便于后续展示或策略处理
    参数说明：
        - code (str): 六位股票代码，函数内部将自动转换为聚宽格式。
    返回值：
        - DataFrame: 包含三列 ['PRICE', 'BID', 'ASK']，索引为当前时间戳。
    """
    jq_code = _code_to_jq(code)
    # 获取实时数据对象
    current_data = get_current_tick(jq_code)
    # 构造 DataFrame 格式返回
    now = pd.to_datetime('now')
    df = pd.DataFrame({
        'PRICE': [current_data.last_price],
        'BID': [current_data.bid1],
        'ASK': [current_data.ask1],
    }, index=[now])    
    return df


def load_online_data(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    函数功能：
        调用聚宽的历史数据接口，获取指定股票在给定起止日期范围内的分钟级别历史行情
        同时补全缺失交易分钟并进行前向填充
    参数说明：
        - code (str): 六位股票代码（自动识别市场并转换格式）
        - start_date (str): 开始日期，格式为 'YYYY-MM-DD'
        - end_date (str): 结束日期，格式为 'YYYY-MM-DD'
    返回值：
        - DataFrame: 分钟级别数据,已处理空值与对齐交易分钟,index 为时间戳。
    """
    jq_code = _code_to_jq(code)
    s = pd.to_datetime(start_date)
    e = pd.to_datetime(end_date) + timedelta(days=1)
    df = get_price(jq_code, start_date=s, end_date=e, frequency="1m").drop(columns=["money"])
    fixed_days = []
    for d, g in df.groupby(df.index.date):
        date_str = pd.to_datetime(d).strftime("%Y-%m-%d")
        g = g.reindex(get_trading_minutes(date_str), method="ffill")
        if pd.isna(g.iloc[0]["open"]):
            for i in range(1, 5):
                if not pd.isna(g.iloc[i]["open"]):
                    g.iloc[0] = g.iloc[i]
                    break
        fixed_days.append(g)
    return pd.concat(fixed_days)


def load_local_data(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    本函数用于从本地读取沪深300中某只股票的分钟级别数据,自动处理中文/英文格式和编码问题。
    输入参数为股票代码,开始日期,结束日期
    返回: pd.DataFrame,标准格式,index 为 datetime,
    列包含 ['open', 'high', 'low', 'close', 'volume']
    """
    fpath = os.path.join(local_path, f"{code}.csv")

    # 经过修改这一步应该不会再raise error
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"找不到文件: {fpath}")

    for enc in ("utf-8-sig", "gbk"):
        try:
            df = pd.read_csv(fpath, encoding=enc, engine="python")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("无法识别文件编码")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    mask = (df.index.date >= to_date(start_date)) & (df.index.date <= to_date(end_date))
    return df.loc[mask]

# ============================================================
# 3. 高频因子(示例 3 个)
# ------------------------------------------------------------
"""
选自《选股因子系列研究(六十九)——高频因子的现实与幻想》,
研报中提到 14:30 之后的成交量占比因子和股票未来收益负相关,
10:00-11:00 的成交量占比因子和股票下月收益显著正相关,
故本算法交易系统拟复刻"尾盘成交量占比因子",
同时根据该因子的构建逻辑进行优化,创建"尾盘成交量占比乘以尾盘涨跌幅"因子
参数:
    data: pd.DataFrame,包含一个股票指定交易日内的分钟频 'close' 和 'volume' 列,索引为分钟级别时间戳
    window: int,滚动平均窗口长度
    log_flag: 计算收益率的时候是否使用价格的对数
返回:
    pd.Series, index 为交易日, 值为因子值(已 shift(1))
"""
# 工具函数_volume_ratio用于计算给定时间内的volume/当日整体的volume，同时支持窗口操作，可以计算多日均值
def _volume_ratio(data: pd.DataFrame, st: str, ed: str, window: int) -> pd.Series:
    seg = data.between_time(st, ed)
    tot = data["volume"].groupby(data.index.date).sum()
    part = seg["volume"].groupby(seg.index.date).sum()
    factor = (part / tot).rolling(window).mean() if window > 1 else part / tot
    factor.index = pd.to_datetime(factor.index)
    return factor.shift(1)


def tail_volume_ratio(data: pd.DataFrame, window: int = 1, log_flag: bool = True) -> pd.Series:
    """
    尾盘成交量占比因子,用于计算某只股票在收盘前最后二十分钟的成交量占比
    参数:
    1、data: pd.DataFrame,包含一个股票指定交易日内的分钟频 'close' 和 'volume' 列,索引为分钟级别时间戳
    2、window: int,滚动平均窗口长度
    3、log_flag: 计算收益率的时候是否使用价格的对数
    返回:
    pd.Series, index 为交易日, 值为因子值(已 shift(1))
    """
    tail_start = "14:40:00"
    tail_end = "15:00:00"
    return _volume_ratio(data, tail_start, tail_end, window)


def head_volume_ratio(data: pd.DataFrame, window: int = 1, log_flag: bool = True) -> pd.Series:
    """
    开盘成交量占比因子,用于计算某只股票在10:00-11:00的成交量占比
    参数:
    1、data: pd.DataFrame,包含一个股票指定交易日内的分钟频 'close' 和 'volume' 列,索引为分钟级别时间戳
    2、window: int,滚动平均窗口长度
    3、log_flag: 计算收益率的时候是否使用价格的对数
    返回:
    pd.Series, index 为交易日, 值为因子值(已 shift(1))
    """
    head_start = "10:00:00"
    head_end = "11:00:00"
    return _volume_ratio(data, head_start, head_end, window)


def tail_volume_return(data: pd.DataFrame, window: int = 1, log_flag: bool = True) -> pd.Series:
    """
    尾盘成交量占比乘以尾盘涨跌幅 复合因子

    参数:
    1、data: pd.DataFrame,包含一个股票指定交易日内的分钟频 'close' 和 'volume' 列,索引为分钟级别时间戳
    2、window: int,滚动平均窗口长度
    3、log_flag: 计算收益率的时候是否使用价格的对数
    返回:
    pd.Series, index 为交易日, 值为因子值(已 shift(1))
    """
    tail_start = "14:40:00"
    tail_end = "15:00:00"
    seg = data.between_time(tail_start, tail_end)
    tot = data["volume"].groupby(data.index.date).sum()
    part = seg["volume"].groupby(seg.index.date).sum()
    vr = (part / tot)

    def _ret(s: pd.Series):
        if len(s) < 2:
            return 0.0
        return np.log(s.iloc[-1] / s.iloc[0]) if log_flag else s.iloc[-1] / s.iloc[0] - 1

    seg_ret = seg["close"].groupby(seg.index.date).apply(_ret)
    factor = (vr * seg_ret).rolling(window).mean() if window > 1 else vr * seg_ret
    factor.index = pd.to_datetime(factor.index)
    return factor.shift(1)

# ============================================================
# 4. 下单模块
# ------------------------------------------------------------

def record_tca(code: str, date: datetime, price: float, target_volume: int, executed_volume: int):
    # 注：不管是哪一种挂单方式都会通过base_order调到本函数并记录当日的TCA情况
    """
    把一次订单执行信息写入 env.tca_book[code]
    - price            : 成交均价(这里直接取 base_order 的成交价)
    - target_volume    : 计划成交量(正买负卖)
    - executed_volume  : 实际成交量(正买负卖)
    """
    # 若无 target_volume 传入，则默认等于 executed_volume
    if target_volume is None:
        target_volume = executed_volume

    # 计算开/收盘、VWAP、佣金等信息
    open_price  = minute_cache[code].at[pd.Timestamp(f"{date.strftime('%Y-%m-%d')} {MORNING_START}"), "open"]
    close_price = minute_cache[code].at[pd.Timestamp(f"{date.strftime('%Y-%m-%d')} 14:59:00"), "close"]
    vwap_price  = price  # 简化：用成交价充当当日 VWAP
    commission  = abs(executed_volume) * price * env.strategy_setting.fee_rate
    slippage    = (price - open_price) * executed_volume  # 计算滑点

    row = {
        "成交均价": price,
        "成交量加权平均价": vwap_price,
        "目标成交量": target_volume,
        "实际成交量": executed_volume,
        "交易关联成本": slippage,
        "今日开盘价": open_price,
        "今日收盘价": close_price,
        "机会成本": (close_price - open_price) * target_volume,
        "佣金费用": commission
    }

    # 写入 env.tca_book
    if code not in env.tca_book:
        env.tca_book[code] = pd.DataFrame(columns=row.keys())
    env.tca_book[code].loc[date] = row


def base_order(code: str, price: float, volume: int) -> int:
    '''
    本函数是最底层的order下单函数,其他复杂的算法交易order需要调用本函数
    用于处理一个特定股票在给定价格和计划增/减仓量的挂单情况
    price:股票此时的价格
    volume:打算增仓/减仓的volume
    '''
    # 注: 在遇到特殊情况时，首先将volume调整成可以现实操作的数值
    # 如果volume不是100的倍数，将其调整为100的倍数
    if volume % 100 != 0:
        volume = int(volume / 100) * 100
    # 如果现金不足，则能买多少买多少，同时将volume调整为100的倍数
    if env.portfolio_info.available_cash - volume * price * (1 + env.strategy_setting.fee_rate) < 0:
        volume = int(env.portfolio_info.available_cash / (price * (1 + env.strategy_setting.fee_rate)) / 100) * 100
    if env.portfolio_info.positions.get(code, 0) < -volume:
        volume = -env.portfolio_info.positions.get(code, 0)

    env.portfolio_info.positions[code] = env.portfolio_info.positions.get(code, 0) + volume
    if volume > 0:
        env.portfolio_info.available_cash -= volume * price * (1 + env.strategy_setting.fee_rate)
    elif volume < 0:
        env.portfolio_info.available_cash -= volume * price * (1 - env.strategy_setting.fee_rate)

    env.trade_book[code] = pd.concat([
        env.trade_book.get(code, pd.DataFrame()),
        pd.DataFrame({"price": price, "volume": volume}, index=[env.today])
    ])
    return volume

#==============================================================================================================
# 下方共有三种下单方法：普通下单、Twap下单、Vwap下单; 不管是采用哪一种下单方法均需要调用 order 的底层函数
# 其中价格会在函数内部根据传入的参数是“回测”还是“实盘”来进行获取
# 实盘的价格会通过聚宽API接口实时获取最新tick级别数据; 回测的价格可以读取本地数据或者选择线上通过聚宽API接口读取历史数据
#==============================================================================================================
def normal_order(code: str, total_volume: int):
    """
    普通市价单执行逻辑：
    - 实盘：用当前价格挂单
    - 回测：用当天开盘价挂单
    """
    if env.strategy_setting.mode == "paper_trading":
        price = load_realtime_data(code).iloc[-1]["PRICE"]
        executed = base_order(code, price, total_volume)
    else:
        today_str = env.today.strftime("%Y-%m-%d")
        price = minute_cache[code].at[pd.Timestamp(f"{today_str} {MORNING_START}"), "open"]
        executed = base_order(code, price, total_volume)
    record_tca(code, env.today, price, total_volume, executed)


def twap_order(code: str, total_volume: int, slices: int = 16):
    """
    TWAP 时间加权平均价格算法执行
    参数:
        code: 股票代码(6 位)
        total_volume: 下单总量
        slices: 切片数量
    """
    per_vol = int(total_volume / slices)
    executed_sum = 0
    start_ts = pd.Timestamp(f"{env.today} {MORNING_START}")
    end_ts   = pd.Timestamp(f"{env.today} {AFTERNOON_END}")

    if env.strategy_setting.mode == "paper_trading":
        delta = (end_ts - start_ts) / slices
        for i in range(slices):
            target_time = start_ts + i * delta
            while pd.Timestamp.utcnow().tz_convert("Asia/Shanghai").time() < target_time.time():
                time.sleep(1)
            price = load_realtime_data(code).iloc[-1]["PRICE"]
            executed_sum += base_order(code, price, per_vol)
    else:
        df = minute_cache[code].loc[get_trading_minutes(env.today.strftime("%Y-%m-%d"))]
        idxs = df.index[::max(1, len(df) // slices)]
        for ts in idxs:
            price = df.at[ts, "close"]
            executed_sum += base_order(code, price, per_vol)

    # 记录 TCA
    record_tca(code, env.today, executed_sum * 0.0 + price if executed_sum else 0, total_volume, executed_sum)


def cal_history_vwap(code: str, window: int = 20) -> pd.Series:
    """
    计算过去 window 个交易日内，每分钟的平均成交量权重。
    返回一个 Series, index 为时间戳，值为权重，占比之和为 1。
    """
    trade_days = get_trade_days(end_date=env.today - timedelta(days=1), count=window)
    vols = [minute_cache[code].loc[get_trading_minutes(d.strftime("%Y-%m-%d")), "volume"] for d in trade_days]
    vol_df = pd.concat(vols, axis=1)
    mean_vol = vol_df.mean(axis=1)
    return (mean_vol / mean_vol.sum())


def vwap_order(code: str, total_volume: int):
    """
    VWAP 算法执行
    - 实盘：按照过去 hist_window 天的分钟成交量分布下单
    - 回测：按照当日分钟成交量占比下单
    """
    executed_sum = 0
    if env.strategy_setting.mode == "paper_trading":
        weights = cal_history_vwap(code)
        order_vols = (weights * total_volume).astype(int)
        remainder = total_volume - order_vols.sum()
        if remainder > 0:
            order_vols.iloc[-1] += remainder
        for ts, vol in order_vols.items():
            if vol <= 0:
                continue
            while pd.Timestamp.utcnow().tz_convert("Asia/Shanghai").time() < ts.time():
                time.sleep(1)
            price = load_realtime_data(code).iloc[-1]["PRICE"]
            executed_sum += base_order(code, price, int(vol))
    else:
        today_str = env.today.strftime("%Y-%m-%d")
        df = minute_cache[code].loc[get_trading_minutes(today_str)]
        weights = df["volume"] / df["volume"].sum() if df["volume"].sum() else pd.Series(1 / len(df), index=df.index)
        order_vols = (weights * total_volume).astype(int)
        remainder = total_volume - order_vols.sum()
        for ts, vol in order_vols.items():
            if vol > 0:
                executed_sum += base_order(code, df.at[ts, "close"], int(vol))
        if remainder > 0:
            last_ts = df.index[-1]
            executed_sum += base_order(code, df.at[last_ts, "close"], remainder)

    # 记录 TCA
    record_tca(code, env.today, executed_sum * 0.0 + price if executed_sum else 0, total_volume, executed_sum)


# ============================================================
# 5. 预读取分钟数据(缓存 + 可选线程池)
# ------------------------------------------------------------
'''
本步骤的目的在于将数据缓存以便于在大规模处理回测计算的时候节省时间
'''
def preload_minute_data(stock_list, start_date, end_date, use_thread: bool = False):
    def read_one(c):
        return c, load_local_data(c, start_date, end_date)

    cache = {}
    if use_thread:
        max_workers = min(os.cpu_count() * 2, len(stock_list))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(read_one, c) for c in stock_list]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                code, df = fut.result()
                cache[code] = df
    else:
        for c in tqdm(stock_list):
            cache[c] = load_local_data(c, start_date, end_date)
    return cache

# ============================================================
# 6. 回测主逻辑
# ------------------------------------------------------------

# 回测主函数
def run_backtest():
    '''
    函数作用：
        本函数为回测系统的核心执行入口，负责完成从初始化数据读取、因子计算、
        信号生成、调仓执行、到资产估值的完整流程。该函数设计遵循逐日循环的逻辑，
        遍历指定时间段内的每一个交易日，动态构建持仓结构，并模拟真实交易执行。

    函数结构：
        Step1. 读取股票池及初始化仓位结构
        Step2. 预读取分钟数据（缓存所有日内数据）
        Step3. 进行逐票因子计算（如 tail_volume_return） 注：这里可以根据因子的名称以及特性进行更换
        Step4. 基于因子信号构建每日选股矩阵
        Step5. 遍历每个交易日进行模拟交易和资产更新
    '''

    # TODO 如果要跑代码的话这里记得改成您自己的本地数据存储地址
    df_info = pd.read_csv(r"C:\Users\Dell\Desktop\算法交易\000300.SH_info.csv", dtype={"code": str})
    env.stock_pool = df_info["code"].tolist()
    env.position_book = pd.DataFrame(columns=env.stock_pool)

    trade_days = get_trade_days(start_date=env.strategy_setting.start_date, end_date=env.strategy_setting.end_date)

    # ---------- 预读取全部分钟数据 ----------
    print("\n预读取分钟数据 …")
    global minute_cache
    minute_cache = preload_minute_data(env.stock_pool, trade_days[0], trade_days[-1], use_thread=True)

    # ---------- 因子计算 ----------
    df_signal = pd.DataFrame(index=trade_days, columns=env.stock_pool)
    print("\n开始计算因子 …")
    for code in tqdm(env.stock_pool):
        df = minute_cache[code]
        df_signal[code] = tail_volume_return(df, window=1, log_flag=True)
        # df_signal[code] = head_volume_ratio(df, window=1, log_flag=True)
        # df_signal[code] = tail_volume_ratio(df, window=1, log_flag=True)

    # ---------- 构建选股矩阵 ----------
    # df_position = (df_signal.shift(1).rank(axis=1, pct=True) > 0.9).astype(int)
    df_position = (df_signal.shift(1).rank(axis=1, pct=True) > 0.9).astype(int)

    # ---------- 初始化首日状态 ----------
    first_day = trade_days[0]
    env.position_book.loc[first_day] = 0
    env.total_value_book.loc[first_day, "total_value"] = env.portfolio_info.total_value

    print("\n开始回测 …")
    for i in tqdm(range(1, len(trade_days))):
        today = trade_days[i]
        yesterday = trade_days[i - 1]
        env.today = today
        env.yesterday = yesterday

        last_position = env.position_book.loc[yesterday].fillna(0).to_dict()
        target_row = df_position.loc[today]
        target_stocks = target_row[target_row == 1].index.tolist()
        stock_num = len(target_stocks)

        target_value = env.portfolio_info.total_value * 0.8 / max(stock_num, 1)
        today_str = today.strftime("%Y-%m-%d")

        target_volume = {
            code: int(target_value / minute_cache[code].at[pd.Timestamp(f"{today_str} {MORNING_START}"), "open"] / 100) * 100
            for code in target_stocks
        }

        trade_volume = {
            code: target_volume.get(code, 0) - last_position.get(code, 0)
            for code in env.stock_pool
        }

        if stock_num > 0:
            trade_volume = dict(sorted(trade_volume.items(), key=lambda x: x[1]))
            for code, vol in trade_volume.items():
                if vol == 0:
                    continue
                method = env.strategy_setting.order_method
                qty = abs(vol) * (1 if vol > 0 else -1)
                if method == "vwap":
                    vwap_order(code, qty)
                elif method == "twap":
                    twap_order(code, qty, slices=16)
                else:
                    normal_order(code, qty)

        # 更新仓位快照和资产净值
        env.position_book.loc[today] = env.portfolio_info.positions
        env.position_book.loc[today] = env.position_book.loc[today].fillna(0) # TODO

        total_val = env.portfolio_info.available_cash
        for code, vol in env.portfolio_info.positions.items():
            close_price = minute_cache[code].at[pd.Timestamp(f"{today_str} 14:59:00"), "close"]
            total_val += close_price * vol

        env.portfolio_info.total_value = total_val
        env.total_value_book.loc[today, "total_value"] = total_val

def replay_day(day: str, demo_n: int = 300):
    """
    如果想在非交易日测试实盘实例可以调用本函数用于实时模拟某一日的实盘回测
      - demo_n 可以选择看前n只股票防止卡顿
      - 在开盘第一分钟只对一部分股票进行调仓，其他保持空仓
      - 跳过 zero-volume 调用，循环秒级完成
    """
    global current_sim_ts, minute_cache

    # 1) 参数配置
    env.strategy_setting.mode         = "paper_trading"
    env.strategy_setting.fee_rate     = 0.0003
    env.strategy_setting.order_method = "normal"

    # 2) 取前 demo_n 支股票，初始化结果表
    codes = pd.read_csv(
        r"C:\Users\Dell\Desktop\算法交易\000300.SH_info.csv",
        dtype=str
    )["code"].tolist()[:demo_n]
    env.stock_pool       = codes
    env.position_book    = pd.DataFrame(columns=codes)
    env.total_value_book = pd.DataFrame()

    # 3) 从本地载入当日分钟数据
    minute_cache = preload_minute_data(codes, day, day, use_thread=False)

    # 4) 选出当日要买入的股票，计算首分钟的调仓量
    demo_buy = codes[: math.ceil(len(codes)/2) ]
    first_ts    = sorted(minute_cache[codes[0]].index)[0]
    target_value = env.portfolio_info.initial_cash * 0.8
    per_val      = target_value / len(demo_buy)
    target_volume = {
        code: int(per_val / minute_cache[code].at[first_ts, "open"] / 100) * 100
        for code in demo_buy
    }

    # 5) 重放每一分钟：首分钟按 target_volume 下单，其余分钟跳过
    from tqdm import tqdm
    for ts in tqdm(sorted(minute_cache[codes[0]].index), desc="Replaying"):
        current_sim_ts = ts
        env.today = pd.to_datetime(day)

        if ts == first_ts:
            for code in codes:
                vol = target_volume.get(code, 0)
                if vol != 0:
                    normal_order(code, vol)

    # 6) 补齐持仓和总资产快照
    current_sim_ts = None
    env.position_book.loc[env.today] = env.portfolio_info.positions

    total_val = env.portfolio_info.available_cash
    for code, vol in env.portfolio_info.positions.items():
        last_price = minute_cache[code].iloc[-1]["close"]
        total_val += last_price * vol
    env.portfolio_info.total_value = total_val
    env.total_value_book.loc[env.today, "total_value"] = total_val

    # 7) 弹出 GUI
    create_gui()


# ============================================================
# 7. 绩效评估
# ------------------------------------------------------------
def get_hs300_close(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取沪深300指数的每日收盘价(聚宽代码为 '000300.XSHG')
    """
    df = get_price('000300.XSHG', start_date=start_date, end_date=end_date, frequency='daily', fields=['close'])
    return df[['close']]

def calc_information_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """计算信息比率 IR = 年化超额收益 / 年化标准差"""
    excess_ret = strategy_returns - benchmark_returns
    mean_excess = excess_ret.mean()
    std_excess = excess_ret.std()
    if std_excess == 0:
        return np.nan
    return np.sqrt(252) * mean_excess / std_excess


def evaluate_performance():
    df = env.total_value_book.copy()
    df["return"] = df["total_value"].pct_change()
    df.dropna(inplace=True)
    annual_return = (df["total_value"].iloc[-1] / df["total_value"].iloc[0]) ** (252 / len(df)) - 1
    sharpe = np.sqrt(252) * df["return"].mean() / df["return"].std()
    cummax = df["total_value"].cummax()
    mdd = ((df["total_value"] - cummax) / cummax).min()

    # 获取沪深300基准收益
    start = df.index[0].strftime("%Y-%m-%d")
    end = df.index[-1].strftime("%Y-%m-%d")
    hs300_df = get_hs300_close(start, end)
    hs300_df = hs300_df.reindex(df.index).fillna(method='ffill')
    benchmark_ret = hs300_df["close"].pct_change().fillna(0)

    # 计算 IR(信息比率)
    ir = calc_information_ratio(df["return"], benchmark_ret)

    print("\n 回测绩效评估:")
    print(f"年化收益率: {annual_return:.2%}")
    print(f"夏普比率  : {sharpe:.2f}")
    print(f"最大回撤  : {mdd:.2%}")
    # print(f"信息比率  : {ir:.2f}")


# ============================================================
# 8. 主入口
# ============================================================
# GUI 可视化界面(后续加入了横向和纵向滚动条，这样在大量数据的时候可以上下左右滑动滚动条)
# ------------------------------------------------------------

# 创建表格填充函数：用于将 DataFrame 的内容展示为 Tkinter TreeView 表格
def _populate_table(frame, df: pd.DataFrame):
    """
    函数功能：
        本函数负责在给定的 Tkinter Frame 容器中构建一个完整的表格展示区域，
        该区域使用 ttk.Treeview 控件来展示传入的 Pandas DataFrame 数据，
        同时自动添加垂直与水平滚动条，此函数返回创建好的 Treeview 控件
    参数说明：
        - frame: Tkinter 界面中的容器，用于承载表格控件
        - df: DataFrame 类型的数据，表头及内容将完整展示在表格中
    返回值：
        - tv: 构建完成并已经填充数据的 Treeview 对象，便于外部进一步操作。
    使用场景：
        适用于所有需要将结构化数据(如回测结果、交易记录等)以表格形式
        显示在 GUI 中的情境。该函数是 GUI 表格渲染的核心构建单元。
    """
    # 创建承载 Treeview 和滚动条的容器 Frame
    container = ttk.Frame(frame)
    container.pack(fill="both", expand=True)

    # 创建 Treeview 表格控件，列头来自 DataFrame columns
    tv = ttk.Treeview(container, columns=list(df.columns), show="headings")
    for col in df.columns:
        tv.heading(col, text=col)
        tv.column(col, width=100, anchor="center")

    # 将 DataFrame 的每一行插入到 Treeview 中
    for _, row in df.iterrows():
        tv.insert("", "end", values=list(row))

    # 加入可以用来滑动的滚动条(水平、垂直均有)
    vsb = ttk.Scrollbar(container, orient="vertical", command=tv.yview)
    hsb = ttk.Scrollbar(container, orient="horizontal", command=tv.xview)
    tv.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")
    tv.pack(side="left", fill="both", expand=True)

    return tv

# 在 create_gui() 之前新增搜索功能支持函数
def filter_dataframe(df: pd.DataFrame, code_keyword: str, date_keyword: str) -> pd.DataFrame:
    """
    函数功能：
        对给定的 Pandas DataFrame 进行基于关键字的模糊搜索过滤操作。
        支持按照“股票代码”和“日期”两个字段的部分匹配进行筛选，返回满足
        所有条件的结果子集。若两个关键字均为空字符串，则直接返回原始数据。
    参数说明：
        - df: 原始 Pandas DataFrame,包含完整的记录集。
        - code_keyword: 使用者输入的股票代码(支持部分匹配，如 "600" 匹配所有代码中含有“600”的股票)
        - date_keyword: 使用者输入的日期关键字，可匹配 index 或日期列内容。
    返回值：
        - DataFrame: 筛选后的结果集，仅保留满足条件的记录。
    """
    code_keyword = code_keyword.strip()
    date_keyword = date_keyword.strip()
    if not code_keyword and not date_keyword:
        return df
    # 匹配函数：行中包含对应关键字即可以match
    def row_match(row):
        return (code_keyword in str(row.get("stock_code", ""))) and (date_keyword in str(row.get("index", "")))
    return df[df.apply(row_match, axis=1)]

# 更新 GUI 创建函数，增加两个搜索框：股票代码和日期
def create_gui():
    """
    函数功能：
        构建主图形用户界面(GUI)，集成多个模块页签(Tabs)，用于展示回测相关的
        多项分析结果。包括但不限于 TCA 分析结果、每日持仓、今日调仓情况以及资产
        回测曲线等。界面元素使用 tkinter + ttk 构建，图表部分通过 matplotlib 嵌入。
    模块说明：
        1. TCA Tab: 显示按股票与日期记录的交易成本分析表格，支持搜索筛选。
        2. 每日仓位 Tab: 展示每日持仓明细(按股票分布)，自动格式化为整型。
        3. 今日调仓 Tab: 若当日存在交易记录则显示成交明细，否则提示“今日无交易”。
        4. 回测曲线 Tab: 以折线图形式展示整个策略历史期间的资产总值变化趋势。
    """
    root = tk.Tk()
    root.title("Backtest Analysis") #Title
    root.geometry("1200x800") #窗口大小

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # ======== TCA Tab ========
    tab_tca = ttk.Frame(notebook)
    notebook.add(tab_tca, text="TCA")

    if env.tca_book:
        tca_records = []
        for code, df in env.tca_book.items():
            df = df.copy()
            df['stock_code'] = code
            df.reset_index(inplace=True)
            tca_records.append(df)
        df_tca = pd.concat(tca_records).reset_index(drop=True).sort_values(by=['index', 'stock_code'])

        # 搜索控件：股票代码和日期两个搜索栏
        search_frame = ttk.Frame(tab_tca)
        search_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(search_frame, text="股票代码：").pack(side="left")
        entry_code = tk.Entry(search_frame, width=10)
        entry_code.pack(side="left", padx=5)

        tk.Label(search_frame, text="日期(部分匹配)：").pack(side="left")
        entry_date = tk.Entry(search_frame, width=12)
        entry_date.pack(side="left", padx=5)

        search_btn = tk.Button(search_frame, text="筛选")
        search_btn.pack(side="left", padx=5)

        # 表格区域
        result_frame = ttk.Frame(tab_tca)
        result_frame.pack(fill="both", expand=True)
        table = _populate_table(result_frame, df_tca)

        def search_callback():
            keyword_code = entry_code.get()
            keyword_date = entry_date.get()
            filtered = filter_dataframe(df_tca, keyword_code, keyword_date)
            for i in table.get_children():
                table.delete(i)
            if filtered.empty:
                table.insert("", "end", values=["无匹配结果"] + [""] * (len(df_tca.columns) - 1))
            else:
                for _, row in filtered.iterrows():
                    table.insert("", "end", values=list(row))

        search_btn.config(command=search_callback)

    else:
        tk.Label(tab_tca, text="暂无 TCA 数据", font=("Arial", 12)).pack(pady=20)

    # ======== 每日仓位 Tab ========
    tab_pos = ttk.Frame(notebook)
    notebook.add(tab_pos, text="每日仓位")
    _populate_table(tab_pos, env.position_book.fillna(0).round(0).astype(int))

    # ======== 今日调仓 Tab ========
    tab_trade = ttk.Frame(notebook)
    notebook.add(tab_trade, text="今日调仓")
    today_trades = []
    for code, df in env.trade_book.items():
        if env.today in df.index:
            row = df.loc[[env.today]].copy()
            row['code'] = code
            today_trades.append(row)
    if today_trades:
        df_today = pd.concat(today_trades).reset_index()
        df_today = df_today[['code', 'price', 'volume']]
        df_today.columns = ['股票代码', '成交价', '成交量']
        _populate_table(tab_trade, df_today)
    else:
        tk.Label(tab_trade, text="今日无交易记录", font=("Arial", 12)).pack(pady=30)

    # ======== 回测曲线 Tab ========
    tab_curve = ttk.Frame(notebook)
    notebook.add(tab_curve, text="回测曲线")
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(env.total_value_book.index, env.total_value_book["total_value"])
    ax.set_title("Total Value Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Portfolio Value")
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=tab_curve)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    root.mainloop()



# 调用 GUI：
'''
本部分暂时保留,可以通过main_run.py调用本代码
'''
# if __name__ == "__main__":
#     env.strategy_setting.start_date = pd.to_datetime("2024-08-01")
#     env.strategy_setting.end_date   = pd.to_datetime("2024-09-30")
#     env.strategy_setting.mode       = "backtest"
#     env.strategy_setting.fee_rate   = 0.0003
#     env.strategy_setting.order_method = "twap"

#     run_backtest()
#     evaluate_performance()
#     create_gui()