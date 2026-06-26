"""
A股数据获取模块
- 实时行情：新浪财经轻量 API
- 历史K线：新浪财经 API（不用 AKShare，更稳定）
- 资金流向：AKShare（非关键功能，失败不影响核心监控）
- 大盘指数：新浪财经
- 财务数据：AKShare
"""
import logging
import re
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# 交易所前缀映射
EXCHANGE_MAP = {
    "6": "sh", "9": "sh",
    "0": "sz", "3": "sz",
}

# 大盘指数代码映射
INDEX_MAP = {
    "000001": "上证指数",
    "399001": "深证成指",
    "399006": "创业板指",
    "000688": "科创50",
    "000300": "沪深300",
    "000016": "上证50",
}


def _code_to_symbol(stock_code: str) -> str:
    prefix = EXCHANGE_MAP.get(stock_code[0], "sh")
    return f"{prefix}{stock_code}"


def _sina_headers() -> dict:
    return {
        "Referer": "https://finance.sina.com.cn",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }


# ──────────────────────────────────────────────
#  实时行情（新浪）
# ──────────────────────────────────────────────

def fetch_realtime(stock_code: str) -> Optional[dict]:
    """获取单只股票的实时行情（新浪财经接口）"""
    symbol = _code_to_symbol(stock_code)
    url = f"https://hq.sinajs.cn/list={symbol}"

    try:
        resp = requests.get(url, headers=_sina_headers(), timeout=10)
        resp.encoding = "gbk"
        match = re.search(r'"(.*?)"', resp.text.strip())
        if not match:
            logger.warning("股票 %s 实时数据解析失败", stock_code)
            return None

        parts = match.group(1).split(",")
        if len(parts) < 32:
            return None

        name = parts[0]
        yesterday_close = float(parts[2]) if parts[2] else 0
        price = float(parts[3]) if parts[3] else 0
        volume_str = parts[8] if len(parts) > 8 else "0"

        change_pct = 0.0
        if yesterday_close > 0:
            change_pct = round((price - yesterday_close) / yesterday_close * 100, 2)

        return {
            "code": stock_code,
            "name": name,
            "price": price,
            "change_pct": change_pct,
            "yesterday_close": yesterday_close,
            "volume": int(float(volume_str)),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        logger.error("获取股票 %s 实时行情失败: %s", stock_code, e)
        return None


# ──────────────────────────────────────────────
#  历史K线（新浪）— 替代 AKShare，更稳定
# ──────────────────────────────────────────────

def fetch_kline(stock_code: str, days: int = 60) -> Optional[pd.DataFrame]:
    """获取股票的历史日K线数据（新浪财经）

    Returns:
        DataFrame: ['date', 'open', 'close', 'high', 'low', 'volume']
    """
    symbol = _code_to_symbol(stock_code)
    url = (
        f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen={days}"
    )

    try:
        resp = requests.get(url, headers=_sina_headers(), timeout=15)
        data = resp.json()

        if not data or len(data) < 5:
            logger.warning("股票 %s K线数据不足", stock_code)
            return None

        rows = []
        for item in data:
            rows.append({
                "date": item.get("day", ""),
                "open": float(item.get("open", 0)),
                "close": float(item.get("close", 0)),
                "high": float(item.get("high", 0)),
                "low": float(item.get("low", 0)),
                "volume": float(item.get("volume", 0)),
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("date").reset_index(drop=True)
        return df

    except Exception as e:
        logger.error("获取股票 %s K线失败: %s", stock_code, e)
        return None


# ──────────────────────────────────────────────
#  资金流向（AKShare）— 非关键
# ──────────────────────────────────────────────

def fetch_fund_flow(stock_code: str) -> Optional[dict]:
    """获取个股资金流向（东方财富）"""
    try:
        import akshare as ak
        market = "sh" if stock_code.startswith(("6", "9")) else "sz"
        df = ak.stock_individual_fund_flow(stock=stock_code, market=market)
        if df.empty:
            return None

        latest = df.iloc[-1]
        main_net = float(latest.get("主力净流入-净额", 0))
        main_ratio = float(latest.get("主力净流入-净占比", 0))

        if main_net > 50_000_000:
            flow_status = "主力大幅流入 💰"
        elif main_net > 10_000_000:
            flow_status = "主力小幅流入 💰"
        elif main_net > -10_000_000:
            flow_status = "主力进出平衡 ⚖️"
        elif main_net > -50_000_000:
            flow_status = "主力小幅流出 💸"
        else:
            flow_status = "主力大幅流出 💸"

        return {
            "main_force_net": main_net,
            "main_force_ratio": main_ratio,
            "super_large_net": float(latest.get("超大单净流入-净额", 0)),
            "large_net": float(latest.get("大单净流入-净额", 0)),
            "medium_net": float(latest.get("中单净流入-净额", 0)),
            "small_net": float(latest.get("小单净流入-净额", 0)),
            "flow_status": flow_status,
        }

    except Exception as e:
        logger.debug("获取股票 %s 资金流向失败: %s", stock_code, e)
        return None


# ──────────────────────────────────────────────
#  大盘指数（新浪财经）
# ──────────────────────────────────────────────

def fetch_market_index(index_code: str = "000001") -> Optional[dict]:
    """获取大盘指数实时行情

    Args:
        index_code: 指数代码，000001=上证, 399001=深证, 399006=创业板

    Returns:
        {name, price, change_pct}
    """
    # 指数用的是 sh/sz 前缀 + 代码
    prefix = "sh" if index_code.startswith("0") else "sz"
    symbol = f"{prefix}{index_code}"
    url = f"https://hq.sinajs.cn/list=s_{symbol}"

    try:
        resp = requests.get(url, headers=_sina_headers(), timeout=10)
        resp.encoding = "gbk"
        match = re.search(r'"(.*?)"', resp.text.strip())
        if not match:
            return None

        parts = match.group(1).split(",")
        name = parts[0] if len(parts) > 0 else INDEX_MAP.get(index_code, "")
        price = float(parts[1]) if len(parts) > 1 and parts[1] else 0
        chg_pct = float(parts[3]) if len(parts) > 3 and parts[3] else 0

        return {"name": name, "code": index_code, "price": price, "change_pct": chg_pct}
    except Exception as e:
        logger.debug("获取指数 %s 失败: %s", index_code, e)
        return None


def fetch_all_indices() -> list[dict]:
    """获取主要大盘指数"""
    results = []
    for code in INDEX_MAP:
        data = fetch_market_index(code)
        if data:
            results.append(data)
    return results


# ──────────────────────────────────────────────
#  个股财务数据（AKShare）
# ──────────────────────────────────────────────

def fetch_financial(stock_code: str) -> Optional[dict]:
    """获取个股基本财务数据

    Returns:
        {pe, pb, revenue_growth, profit_growth, roe, market_cap}
    """
    try:
        import akshare as ak
        df = ak.stock_a_lg_indicator(symbol=stock_code)
        if df.empty:
            return None
        # 取最新一行
        latest = df.iloc[-1]
        return {
            "pe": float(latest.get("市盈率-动态", 0) or 0),
            "pb": float(latest.get("市净率", 0) or 0),
            "revenue_growth": float(latest.get("营业收入同比增长率", 0) or 0),
            "profit_growth": float(latest.get("净利润同比增长率", 0) or 0),
            "roe": float(latest.get("净资产收益率", 0) or 0),
            "market_cap": float(latest.get("总市值", 0) or 0),
        }
    except Exception as e:
        logger.debug("获取股票 %s 财务数据失败: %s", stock_code, e)
        return None


# ──────────────────────────────────────────────
#  板块行情（AKShare）
# ──────────────────────────────────────────────

def fetch_sector_performance(sector_name: str = "") -> list[dict]:
    """获取板块涨跌排行

    Returns:
        [{name, price, change_pct}]
    """
    try:
        import akshare as ak
        df = ak.stock_board_industry_name_em()
        if df.empty:
            return []
        results = []
        for _, row in df.iterrows():
            name = row.get("板块名称", "")
            if sector_name and sector_name not in name:
                continue
            if sector_name:
                results.append({
                    "name": name,
                    "price": float(row.get("最新价", 0)),
                    "change_pct": float(row.get("涨跌幅", 0)),
                })
        return results[:10]  # 最多10个
    except Exception as e:
        logger.debug("获取板块行情失败: %s", e)
        return []
