"""
尾盘低吸扫描 — 14:30-15:00 运行

筛选条件：
  1. 基本面好（预置优质股票池，行业龙头）
  2. 近60天内有涨停记录（有涨停基因）
  3. 当前回调到支撑位（适合低吸）
  4. 整体趋势向上（非下降通道）
"""
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.fetcher import fetch_realtime, fetch_kline, fetch_fund_flow
from src.signals import _sma, _calc_rsi, _fmt_volume
from src.sectors import get_sector_tag

logger = logging.getLogger(__name__)

# ── 优质股票池（基本面好+有涨停基因的行业龙头） ──
QUALITY_POOL = {
    # 科技
    "000063": "中兴通讯",
    "002475": "立讯精密",
    "603501": "韦尔股份",
    "002371": "北方华创",
    "688981": "中芯国际",
    "300782": "卓胜微",
    "603986": "兆易创新",
    "688041": "海光信息",
    "000977": "浪潮信息",
    "603019": "中科曙光",
    # 新能源
    "300750": "宁德时代",
    "002594": "比亚迪",
    "300274": "阳光电源",
    "601012": "隆基绿能",
    # 消费
    "600519": "贵州茅台",
    "000858": "五粮液",
    "603288": "海天味业",
    "000568": "泸州老窖",
    # 医药
    "600276": "恒瑞医药",
    "300760": "迈瑞医疗",
    "603259": "药明康德",
    "300015": "爱尔眼科",
    # 金融/券商
    "600036": "招商银行",
    "601318": "中国平安",
    "300059": "东方财富",
    "600030": "中信证券",
    # 高端制造
    "300124": "汇川技术",
    "600031": "三一重工",
    "002129": "中环股份",
    # 有色/资源
    "601899": "紫金矿业",
    "600585": "海螺水泥",
    # 通信/运营商
    "600941": "中国移动",
    "688008": "澜起科技",
    "002156": "通富微电",
}


TECH_STOCKS = {
    "000063", "600487", "600522", "600105",  # 通信
    "603501", "002371", "688981", "300782", "603986", "688041", "688012", "300661", "688008", "002156", "603629",  # 半导体
    "002475",                     # 消费电子
    "688111", "002230",           # 软件/AI
    "603019", "000977",           # 算力
    "300454",                     # 网络安全
    "300124",                     # 工业自动化
}


def has_limit_up_history(kline_df: pd.DataFrame, lookback: int = 60) -> bool:
    """检查近 lookback 天内是否有涨停"""
    if kline_df is None or len(kline_df) < 5:
        return False
    df = kline_df.tail(lookback)
    for i in range(1, len(df)):
        prev_close = df.iloc[i - 1]["close"]
        curr_close = df.iloc[i]["close"]
        if prev_close > 0 and (curr_close / prev_close - 1) >= 0.095:
            return True
    return False


def scan_dip_buy_candidates(max_price: float = 0, tech_only: bool = False) -> list[dict]:
    """扫描符合条件的低吸候选股

    Args:
        max_price: 最高股价限制，0 表示不限制
        tech_only: 是否只扫描科技板块

    Returns:
        按评分排序的候选列表
    """
    results = []
    pool = {k: v for k, v in QUALITY_POOL.items() if not tech_only or k in TECH_STOCKS}

    for code, name in pool.items():
        # 跳过创业板
        if code.startswith("30"):
            continue
        logger.info("扫描 %s (%s)...", name, code)

        # 实时行情
        realtime = fetch_realtime(code)
        if not realtime:
            continue
        price = realtime["price"]
        disp_name = realtime.get("name") or name

        # K线（取60天）
        kline = fetch_kline(code, days=65)
        if kline is None or len(kline) < 25:
            continue

        closes = kline["close"].values.astype(float)

        # 1. 检查涨停基因
        if not has_limit_up_history(kline, 60):
            continue

        # 2. 计算均线
        ma5 = _sma(closes, 5)
        ma10 = _sma(closes, 10)
        ma20 = _sma(closes, 20)

        valid = ~np.isnan(ma5) & ~np.isnan(ma10) & ~np.isnan(ma20)
        ma5_v, ma10_v, ma20_v = ma5[valid], ma10[valid], ma20[valid]

        if len(ma5_v) < 1:
            continue

        c_ma5, c_ma10, c_ma20 = ma5_v[-1], ma10_v[-1], ma20_v[-1]

        # 3. 计算偏离度
        dev_ma10 = (price - c_ma10) / c_ma10 * 100 if c_ma10 > 0 else 999
        dev_ma20 = (price - c_ma20) / c_ma20 * 100 if c_ma20 > 0 else 999

        # 4. 计算RSI
        rsi_val = _calc_rsi(closes, 14)

        # 5. 均线趋势
        if c_ma5 > c_ma10 > c_ma20:
            ma_status = "多头排列 ↑"
            trend_score = 20
        elif c_ma5 > c_ma20:
            ma_status = "短期偏多 ↗"
            trend_score = 10
        elif c_ma10 > c_ma20:
            ma_status = "中期偏弱 →"
            trend_score = 0
        else:
            ma_status = "空头排列 ↓"
            trend_score = -20

        # 6. 评分逻辑
        score = 50  # 基础分

        # 加分：回调到支撑位
        if -3 < dev_ma10 < 1:
            score += 20  # 靠近10日线，好位置
        elif -5 < dev_ma20 < 0:
            score += 15  # 靠近20日线，也不错
        elif -3 < dev_ma20 < 3:
            score += 10
        else:
            score -= 10  # 偏离太远

        # 加分：RSI 适中（不冷不过热）
        if rsi_val is not None:
            if 30 <= rsi_val <= 55:
                score += 15  # 黄金区间
            elif 55 < rsi_val <= 65:
                score += 5  # 偏强但还能接受
            elif rsi_val > 65:
                score -= 10  # 偏高了
            elif 20 <= rsi_val < 30:
                score += 5  # 偏冷但可能反弹
            else:
                score -= 10  # 太冷了

        # 加分：趋势
        score += trend_score

        # 综合判断理由
        reasons = []
        if -3 < dev_ma10 < 1:
            reasons.append(f"回踩10日线(偏离{dev_ma10:+.1f}%)")
        elif -5 < dev_ma20 < 0:
            reasons.append(f"回踩20日线(偏离{dev_ma20:+.1f}%)")
        else:
            reasons.append(f"偏离10日线{dev_ma10:+.1f}%")

        if rsi_val is not None:
            if rsi_val <= 45:
                reasons.append(f"RSI{rsi_val:.0f}偏低")
            else:
                reasons.append(f"RSI{rsi_val:.0f}适中")

        reasons.append(ma_status)
        reasons.append("有涨停基因")

        if score >= 70:
            action = "✅ **重点关注**"
        elif score >= 55:
            action = "👀 **可关注**"
        else:
            action = "⏸ 暂观望"

        results.append({
            "code": code,
            "name": disp_name,
            "price": price,
            "rsi": rsi_val,
            "dev_ma10": dev_ma10,
            "dev_ma20": dev_ma20,
            "ma_status": ma_status,
            "score": min(100, max(0, score)),
            "action": action,
            "reason": "，".join(reasons),
        })

    # 按评分排序
    results.sort(key=lambda r: r["score"], reverse=True)

    # 价格过滤
    if max_price > 0:
        results = [r for r in results if r["price"] <= max_price]

    return results


def scan_close_buy_candidates(max_price: float = 0, tech_only: bool = False) -> list[dict]:
    """尾盘买入扫描 — 14:50-15:00 运行

    根据市场实际情况决定是否推荐：
      - 大盘跌超1.5% → 不推荐（市场太差）
      - 大盘震荡或上涨 → 正常推荐
      - 个股条件同上

    Returns:
        按评分排序的候选列表，如果市场不适合会返回空列表
    """
    # ── 先看大盘环境 ──
    market_ok, market_info = _check_market_condition()
    results = []

    if not market_ok:
        logger.info("尾盘买入跳过：%s", market_info)
        return []

    pool = {k: v for k, v in QUALITY_POOL.items() if not tech_only or k in TECH_STOCKS}

    for code, name in pool.items():
        if code.startswith("30"):
            continue
        logger.info("尾盘扫描 %s (%s)...", name, code)
        realtime = fetch_realtime(code)
        if not realtime:
            continue
        price = realtime["price"]
        chg = realtime.get("change_pct", 0)
        disp_name = realtime.get("name") or name

        # 涨幅过滤：1%~5%（涨势要有强度，但不追涨停）
        if chg < 1.0 or chg > 5.0:
            continue

        kline = fetch_kline(code, days=65)
        if kline is None or len(kline) < 25:
            continue

        closes = kline["close"].values.astype(float)
        volumes = kline["volume"].values.astype(float)
        ff = fetch_fund_flow(code)

        # 计算指标
        from src.scoring import compute_score
        score_info = compute_score(closes, volumes, price, ff)
        score = score_info.get("score", 0)
        if score < 50:
            continue

        # 均线多头排列检查（淘汰杂毛：必须MA5>MA10>MA20）
        ma5 = _sma(closes, 5)
        ma10 = _sma(closes, 10)
        ma20 = _sma(closes, 20)
        valid = ~np.isnan(ma5) & ~np.isnan(ma10) & ~np.isnan(ma20)
        s5, s10, s20 = ma5[valid], ma10[valid], ma20[valid]
        if len(s5) < 1:
            continue
        if s5[-1] > s10[-1] > s20[-1]:
            trend = "多头排列↑"
        elif s5[-1] > s20[-1]:
            trend = "短期偏多"
        else:
            continue  # 趋势偏弱，淘汰杂毛

        # RSI检查（剔除弱势和超买）
        rsi_val = _calc_rsi(closes, 14)
        if rsi_val:
            if rsi_val > 65:
                continue  # 超买不追
            if rsi_val < 40:
                continue  # 弱势不碰（杂毛）

        # 成交量检查（缩量没底气，淘汰）
        avg_v = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
        vol_ratio = volumes[-1] / avg_v if avg_v > 0 else 0
        if vol_ratio < 0.7:
            continue  # 缩量太严重，淘汰

        reasons = [f"涨幅{chg:+.1f}%", f"评分{score}", trend]
        if rsi_val:
            reasons.append(f"RSI{rsi_val:.0f}")
        if vol_ratio > 0.8:
            reasons.append(f"量比{vol_ratio:.1f}")

        if score >= 70:
            action = "✅ **建议买入**"
        else:
            action = "👀 **可关注**"

        results.append({
            "code": code, "name": disp_name, "price": price,
            "chg": chg, "score": score, "rsi": rsi_val,
            "trend": trend, "vol_ratio": round(vol_ratio, 1),
            "action": action, "reason": "，".join(reasons),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    if max_price > 0:
        results = [r for r in results if r["price"] <= max_price]
    return results


def generate_close_buy_report(candidates: list[dict], max_price: float = 0, tech_only: bool = False) -> str:
    """生成尾盘买入推荐报告（列出所有扫描股票的状况）"""
    today_str = datetime.now().strftime("%m/%d")
    price_note = f"≤{max_price:.0f}元 " if max_price > 0 else ""
    pool_note = "科技股" if tech_only else "优质股"

    # 市场环境
    market_ok, market_info = _check_market_condition()

    lines = [f"📋 **尾盘买入推荐** · {today_str} 14:50"]
    lines.append("")
    lines.append(f"📊 大盘环境: {market_info}")
    lines.append("")

    if not candidates:
        lines.append("❌ **今日不建议尾盘买入**")
        lines.append("")
        if "跌幅较大" in market_info:
            lines.append("大盘跌幅较大(-1.5%+)，系统性风险偏高，不建议操作")
        elif "市场偏弱" in market_info:
            lines.append("大盘走势偏弱，个股机会有限，建议观望")
        elif "数据" in market_info:
            lines.append(f"📡 {market_info}")
        else:
            lines.append("当前没有同时满足涨幅1%~5%、多头趋势、RSI 40~65、量比>0.7的个股")
        lines.append("")
        # 总是列出全市场股票评分
        lines.append("")
        lines.append(f"**📊 全市场扫描**")
        lines.append("")
        pool = {k: v for k, v in QUALITY_POOL.items() if not tech_only or k in TECH_STOCKS}
        for s_code, s_name in pool.items():
            if s_code.startswith("30"):
                continue
            rt = fetch_realtime(s_code)
            if not rt:
                continue
            sp = rt.get("price", 0)
            sc = rt.get("change_pct", 0)
            sk = fetch_kline(s_code, days=65)
            if sk is None or len(sk) < 25:
                continue
            s_closes = sk["close"].values.astype(float)
            s_volumes = sk["volume"].values.astype(float)
            s_ff = fetch_fund_flow(s_code)
            from src.scoring import compute_score
            si = compute_score(s_closes, s_volumes, sp, s_ff)
            ss = si.get("score", 0)
            # 简单判断
            s_rsi_v = _calc_rsi(s_closes, 14)
            s_rsi_str = f"RSI{s_rsi_v:.0f}" if s_rsi_v else "RSI?"
            s_ma5 = _sma(s_closes, 5)
            s_ma20 = _sma(s_closes, 20)
            s_valid = ~np.isnan(s_ma5) & ~np.isnan(s_ma20)
            s_trend = "↑" if (len(s_ma5[s_valid]) > 0 and s_ma5[s_valid][-1] > s_ma20[s_valid][-1]) else "↓"
            lines.append(f"  {sc:+.1f}% {s_name}({s_code}) 评分{ss} {s_trend} {s_rsi_str}")
        lines.append("")
        lines.append("💡 明日开盘后重新评估")
        return "\n".join(lines)

    lines.append(f"选股标准：涨幅1%~5%、多头排列、RSI适中(40~65)、量比>0.7")
    lines.append(f"符合条件: {len(candidates)} 只（涨势较好的标的）")
    # 加入明日预判
    for c in candidates[:6]:
        k_pred = fetch_kline(c["code"], 60)
        if k_pred is not None and "high" in k_pred.columns:
            from src.predictor import predict_tomorrow
            try:
                pred = predict_tomorrow(
                    k_pred["close"].values.astype(float),
                    k_pred["high"].values.astype(float),
                    k_pred["low"].values.astype(float),
                    k_pred["volume"].values.astype(float) if "volume" in k_pred.columns else np.array([]),
                    c["price"],
                )
                c["pred_dir"] = pred["direction"]
                c["pred_reason"] = pred["reason"]
                c["pred_target_label"] = pred.get("target_label", "明日")
                c["pred_conf"] = pred.get("confidence", 0)
            except:
                c["pred_dir"] = "?"
                c["pred_reason"] = ""
                c["pred_target_label"] = "明日"
                c["pred_conf"] = 0
        else:
            c["pred_dir"] = "?"
            c["pred_reason"] = ""
            c["pred_target_label"] = "明日"
            c["pred_conf"] = 0
    lines.append(f"**🏆 尾盘关注名单**")
    lines.append("")
    for i, c in enumerate(candidates[:6], 1):
        pred_icon = {"看涨": "📈", "看跌": "📉", "震荡": "➖"}.get(c.get("pred_dir", ""), "")
        pred_text = f"  🔮{c.get('pred_target_label','明日')}{pred_icon}{c.get('pred_dir','?')}" if c.get("pred_dir") else ""
        # 评分星级显示
        score = c.get("score", 0)
        rsi = c.get("rsi", 0)
        vol = c.get("vol_ratio", 0)
        trend = c.get("trend", "")
        lines.append(
            f"{i}. {c['action']}  {c['name']}({c['code']}){get_sector_tag(c['code'])}  {c['price']:.2f}元{pred_text}"
        )
        lines.append(f"   评分{score} · {trend} · RSI{rsi if rsi else '?'} · 量比{vol}")
        if c.get("pred_reason"):
            label = c.get("pred_target_label", "明日")
            lines.append(f"   🔮 {label}预判: {c['pred_reason']}")
        lines.append("")

    lines.append("💡 尾盘买入策略：14:55前考虑建仓，次日盘中择机止盈")
    return "\n".join(lines).strip()


def _check_market_condition() -> tuple[bool, str]:
    """检查当前大盘环境是否适合尾盘买入"""
    try:
        from src.fetcher import fetch_market_index
        idx = fetch_market_index("000001")
        if idx:
            chg = idx.get("change_pct", 0)
            price = idx.get("price", 0)
            if chg <= -1.5:
                return False, f"上证指数 {price:.1f} {chg:+.2f}% ⚠️ 跌幅较大，不建议买入"
            elif chg <= -0.5:
                return False, f"上证指数 {price:.1f} {chg:+.2f}% 🔶 市场偏弱，谨慎操作"
            else:
                return True, f"上证指数 {price:.1f} {chg:+.2f}% ✅ 市场环境正常"
        return True, "大盘数据获取失败，按正常情况处理"
    except Exception as e:
        logger.debug("大盘环境检查异常: %s", e)
        return True, "大盘数据异常，按正常情况处理"

def generate_dip_buy_report(candidates: list[dict], max_price: float = 0, tech_only: bool = False) -> str:
    """生成尾盘低吸推荐报告"""
    lines = []
    pool_note = "科技股"
    pool_note += f" · ≤{max_price:.0f}元" if max_price > 0 else ""
    lines.append(f"📋 **尾盘低吸扫描** · {pool_note} · {datetime.now().strftime('%m/%d %H:%M')}")
    lines.append("")
    pool_name = "科技股" if tech_only else "优质股"
    pool_size = len(QUALITY_POOL) if not tech_only else len(TECH_STOCKS)
    lines.append(f"共扫描 {pool_size} 只{pool_name}")
    lines.append(f"有涨停基因 + 回调到位的：{len([c for c in candidates if c['score'] >= 55])} 只")
    lines.append("")

    # 推荐列表
    top = [c for c in candidates if c["score"] >= 55]
    if top:
        lines.append(f"**🏆 低吸关注名单**")
        lines.append("")
        for i, c in enumerate(top[:8], 1):
            lines.append(
                f"{i}. {c['action']}  {c['name']}({c['code']}){get_sector_tag(c['code'])}  {c['price']:.2f}元"
            )
            lines.append(f"   评分 {c['score']} · {c['reason']}")
            lines.append("")

    # 其余
    rest = [c for c in candidates if 40 <= c['score'] < 55]
    if rest:
        lines.append(f"**📌 还可看看**")
        for c in rest[:5]:
            lines.append(f"  · {c['name']}({c['code']}){get_sector_tag(c['code'])} 评分{c['score']} {c['reason']}")

    return "\n".join(lines).strip()
