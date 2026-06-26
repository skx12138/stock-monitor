"""
模拟交易系统 — 不花真钱，跟踪策略信号自动"买卖"

流程:
  收到买入信号 → 记录买入价和数量
  收到卖出信号 → 记录卖出价，计算盈亏
  每天推送 → 当前持仓和总盈亏
"""
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Optional

from src.signals import Signal, SignalDedup

logger = logging.getLogger(__name__)

TRADE_FILE = "papertrade_data.json"

# ── 数据结构 ──

@dataclass
class Position:
    """持仓"""
    stock_code: str
    stock_name: str
    buy_date: str
    buy_price: float
    shares: int          # 持仓股数
    current_price: float = 0.0
    total_cost: float = 0.0
    market_value: float = 0.0
    profit_pct: float = 0.0
    profit_amount: float = 0.0
    peak_price: float = 0.0
    add_count: int = 0            # 加仓次数

@dataclass
class TradeRecord:
    """历史交易"""
    stock_code: str
    stock_name: str
    action: str          # "buy" / "sell"
    date: str
    price: float
    shares: int
    reason: str = ""
    profit_pct: float = 0.0
    profit_amount: float = 0.0

@dataclass
class Portfolio:
    """账户"""
    cash: float = 100000.0       # 初始资金10万
    total_value: float = 100000.0
    positions: dict = field(default_factory=dict)   # code -> Position
    trades: list = field(default_factory=list)
    daily_values: list = field(default_factory=list)  # 每日净值记录


# ── 交易引擎 ──

class PaperTrading:
    """模拟交易引擎"""

    def __init__(self, initial_cash: float = 100000):
        self.portfolio = Portfolio(cash=initial_cash, total_value=initial_cash)
        self.trade_dedup: dict[str, datetime] = {}
        self.trade_cooldown = 0
        self.max_positions = 999
        self.commission = 0.00025      # 股票佣金万2.5
        self.etf_commission = 0.0001    # ETF/基金佣金万1
        self.stamp_duty = 0.001        # 印花税千1（卖出收）
        self.transfer_fee = 0.00001    # 过户费万0.1
        self.min_commission = 5.0      # 股票最低佣金5元
        self.min_etf_commission = 0.1  # ETF最低佣金0.1元
        self.trail_activate = 4.0
        self.trail_pullback = 4.0
        self.enable_volatility_adjust = True
        self.enable_sector_filter = True
        self._sector_cache = None
        self._sector_cache_time = 0
        self._load()

    def _get_sector_tag(self, code: str) -> str:
        try:
            from src.sectors import get_sector_tag
            return get_sector_tag(code)
        except: return ""

    def refresh_hot_sectors(self):
        """刷新热门板块列表（每日一次）"""
        now = datetime.now()
        if now.timestamp() - self._sector_cache_time < 3600:
            return
        self._sector_cache_time = now.timestamp()
        try:
            from src.fetcher import fetch_sector_performance
            sectors = fetch_sector_performance()
            if sectors:
                # 取涨幅前5的板块
                sectors.sort(key=lambda s: abs(s.get("change_pct", 0)), reverse=True)
                self._sector_cache = [s["name"] for s in sectors[:5]]
        except: pass

    def _save(self):
        """保存交易数据到文件"""
        try:
            data = {
                "cash": self.portfolio.cash,
                "total_value": self.portfolio.total_value,
                "positions": {
                    k: asdict(v) for k, v in self.portfolio.positions.items()
                },
                "trades": [asdict(t) for t in self.portfolio.trades[-100:]],  # 保留最近100条
                "daily_values": self.portfolio.daily_values[-365:],
            }
            with open(TRADE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug("保存模拟交易数据失败: %s", e)

    def _load(self):
        """加载历史交易数据"""
        if not os.path.exists(TRADE_FILE):
            return
        try:
            with open(TRADE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.portfolio.cash = data.get("cash", 100000)
            self.portfolio.total_value = data.get("total_value", 100000)
            for code, p in data.get("positions", {}).items():
                self.portfolio.positions[code] = Position(**p)
            for t in data.get("trades", []):
                self.portfolio.trades.append(TradeRecord(**t))
            for v in data.get("daily_values", []):
                self.portfolio.daily_values.append(v)
        except Exception as e:
            logger.debug("加载模拟交易数据失败: %s", e)

    def process_signal(self, signal: Signal, current_price: float):
        """根据信号执行模拟交易（基于信号方向）"""
        # ... 原有逻辑保持不变 ...

    def process_score(self, code: str, name: str, current_price: float,
                      score_info: dict, kline=None) -> Optional[TradeRecord]:
        """根据评分系统执行模拟交易（优化版V3）

        规则:
          - 动态仓位: 评分50买10%, 55买15%, 60买20%, 65买25%, 70买30%
          - 分批止盈: 盈利10%卖1/3, 15%再卖1/3, 20%清仓
          - 自适应止损: 基于ATR动态计算止损位
          - 卖出条件: 评分<40 或 触发止损/止盈
        """
        score = score_info.get("score", 0)
        action = score_info.get("action", "")
        now = datetime.now()

        last_trade = self.trade_dedup.get(code)
        if last_trade:
            elapsed = (now - last_trade).total_seconds() / 60
            if elapsed < self.trade_cooldown:
                return None
        self.trade_dedup[code] = now

        trade = None
        pos = self.portfolio.positions.get(code)
        now = datetime.now()

        # ── 板块轮动过滤（本项目仅监控固定几只股票，跳过板块过滤） ──
        sector_ok = True

        # ── 波动率调整仓位（高波动降仓位） ──
        vol_ratio = 1.0
        atr_val = score_info.get("atr", 0)
        price_now = current_price
        if self.enable_volatility_adjust and atr_val > 0 and price_now > 0:
            vol_pct = atr_val / price_now * 100
            if vol_pct > 5:
                vol_ratio = 0.5    # 高波动，仓位减半
            elif vol_pct > 3:
                vol_ratio = 0.75   # 中波动，仓位打75折

        # ── 动态仓位 ──
        def get_ratio(s):
            base = 0.30 if s >= 65 else 0.25 if s >= 60 else 0.20 if s >= 55 else 0.15 if s >= 50 else 0.10 if s >= 45 else 0
            return base * vol_ratio

        # ── 移动止盈 ──
        if pos:
            if current_price > pos.peak_price:
                pos.peak_price = current_price
            profit = (current_price / pos.buy_price - 1) * 100
            # 动态移动止盈：根据均线趋势调整回撤容忍度
            ma5_vv = score_info.get("ma5", 0)
            ma20_vv = score_info.get("ma20", 0)
            if ma5_vv > ma20_vv:
                # 多头趋势中，给更大回撤空间
                trail_pull = self.trail_pullback + 2
            else:
                trail_pull = self.trail_pullback
            if profit >= self.trail_activate:
                pullback = (pos.peak_price - current_price) / pos.peak_price * 100
                if pullback >= trail_pull:
                    return self._sell_position(code, current_price, f"移动止盈(从高点回撤{pullback:.1f}%)")

            # ── 分批止盈（让利润多跑一会儿） ──
            sell_shares = 0
            profit_str = ""
            if profit >= 25:
                sell_shares = pos.shares  # 全清
                profit_str = f"止盈{profit:.1f}%清仓"
            elif profit >= 15:
                sell_shares = pos.shares // 3  # 卖1/3
                profit_str = f"止盈{profit:.1f}%卖1/3"
            elif profit >= 8:
                sell_shares = pos.shares // 3  # 卖1/3
                profit_str = f"止盈{profit:.1f}%卖1/3"

            if sell_shares > 0:
                trade = self._sell_partial(code, current_price, sell_shares, profit_str)

        # ── 追涨检测（涨太多不买，跌了才是机会） ──
        chase_penalty = 1.0
        if not trade and code not in self.portfolio.positions:
            # 检查日内涨幅
            intraday_chg = score_info.get("change_pct", 0)
            if intraday_chg > 4:
                chase_penalty = 0  # 涨超4%，不买
            elif intraday_chg > 2:
                chase_penalty = 0.5  # 涨超2%，仓位减半
            # 检查近5日涨幅
            if kline is not None and len(kline) > 5:
                closes_arr = kline["close"].values.astype(float)
                if len(closes_arr) >= 5:
                    recent_chg = (current_price / closes_arr[-5] - 1) * 100
                    if recent_chg > 15:
                        chase_penalty = 0  # 近5日涨超15%，不追
                    elif recent_chg > 8:
                        chase_penalty = min(chase_penalty, 0.5)  # 近5日涨超8%，减半
            # 跌幅是机会：跌超2%时适当增加仓位
            if intraday_chg < -2 and chase_penalty > 0:
                chase_penalty = min(1.0, chase_penalty + 0.25)  # 跌时加仓25%

        # ── 次日涨跌预判（周五自动预测下周一, 加仓前看方向） ──
        prediction = None
        if kline is not None and len(kline) > 20:
            try:
                from src.predictor import predict_tomorrow
                closes_arr = kline["close"].values.astype(float)
                volumes_arr = kline["volume"].values.astype(float) if "volume" in kline.columns else np.array([])
                if "high" in kline.columns and "low" in kline.columns:
                    highs_arr = kline["high"].values.astype(float)
                    lows_arr = kline["low"].values.astype(float)
                else:
                    highs_arr = closes_arr
                    lows_arr = closes_arr
                prediction = predict_tomorrow(closes_arr, highs_arr, lows_arr, volumes_arr, current_price)
            except Exception as e:
                logger.debug("预测失败 %s: %s", name, e)

        # ── 大盘环境判断（跌势时条件性买入） ──
        market_declining = False
        try:
            from src.scoring import _get_market_mode
            m_mode, m_desc, m_chg = _get_market_mode()
            if m_mode == "declining":
                market_declining = True
                logger.info("大盘%s(%+.1f%%), %s 需条件买入", m_desc, m_chg, name)
        except Exception as e:
            logger.debug("大盘判断失败: %s", e)

        # ── 深跌反弹机会：当日跌幅巨大时放宽大盘/预测过滤 ──
        intraday_chg = score_info.get("change_pct", 0)
        deep_drop = intraday_chg < -5  # 当日跌超5%视为深跌机会

        # ── 跌停处理（跌停+预测看涨=抄底，跌停+预测看跌=卖出） ──
        limit_down = intraday_chg <= -9.5  # 接近跌停
        if limit_down and prediction:
            if prediction["direction"] == "看涨":
                chase_penalty = min(chase_penalty + 0.5, 1.5)
                logger.info("跌停(%.1f%%)+预测%s(%.0f%%), %s 抄底机会",
                            intraday_chg, prediction["direction"], prediction["confidence"], name)
            elif prediction["direction"] == "看跌":
                logger.info("跌停(%.1f%%)+预测看跌, %s 暂不参与，等后续反弹机会", intraday_chg, name)
                # 已持仓的等待反弹，不割肉
                # 未持仓则跳过买入
                chase_penalty = 0

        # ── 买入（动态仓位 + 大盘/预测过滤） ──
        if not trade:
            ratio = get_ratio(score) * chase_penalty
            if ratio > 0 and code not in self.portfolio.positions and sector_ok:
                # 跌势时减半仓位+需看涨预测
                if market_declining:
                    if prediction and prediction["direction"] == "看涨":
                        ratio *= 0.5
                        logger.info("大盘跌势但预测看涨，半仓买入 %s", name)
                    elif deep_drop:
                        ratio *= 0.5  # 深跌允许半仓买入
                        logger.info("大盘跌势+深跌反弹机会(%.1f%%)，半仓买入 %s", intraday_chg, name)
                    else:
                        ratio = 0  # 无看涨预测则跳过
                if ratio > 0:
                    trade = self._buy_position(code, name, current_price, ratio, f"评分{score}分·买{ratio*100:.0f}%")
            elif ratio > 0 and code in self.portfolio.positions and score >= 65:
                pos = self.portfolio.positions.get(code)
                if pos and pos.profit_pct > 0:  # 盈利中才加仓
                    # 金字塔加仓：次数越多，加的越少，门槛越高
                    add_ratios = [0.15, 0.10, 0.05]
                    add_scores = [55, 60, 65]
                    if pos.add_count < len(add_ratios):
                        idx = pos.add_count
                        if score >= add_scores[idx]:
                            # 加仓间隔：改用 ATR 动态计算
                            atr_val = score_info.get("atr", 0)
                            if atr_val > 0 and current_price > 0:
                                atr_pct = atr_val / current_price * 100
                                min_pct = max(atr_pct * 1.5, 1.0)
                                min_price = pos.buy_price * (1 + min_pct / 100)
                            else:
                                min_price = pos.buy_price * 1.03
                            if current_price >= min_price:
                                # 大盘+预测检查
                                add_skip = False
                                if prediction and prediction["direction"] == "看跌":
                                    logger.info("预测看跌，跳过 %s 第%s次加仓", name, idx + 1)
                                    add_skip = True
                                elif market_declining and not (prediction and prediction["direction"] == "看涨"):
                                    if deep_drop:
                                        logger.info("大盘跌势+深跌(%.1f%%)，允许 %s 第%s次加仓", intraday_chg, name, idx + 1)
                                    else:
                                        logger.info("大盘跌势无看涨信号，跳过 %s 第%s次加仓", name, idx + 1)
                                        add_skip = True
                                if not add_skip:
                                    # 量价共振：MACD金叉或放量才加仓（分数远超门槛时放宽）
                                    details = score_info.get("details", {})
                                    macd_ok = details.get("MACD", {}).get("score", 0) > 0
                                    vol_ok = details.get("成交量", {}).get("score", 0) > 0
                                    score_margin = score - add_scores[idx]
                                    if not macd_ok and not vol_ok and score_margin < 10:
                                        logger.info("无量价共振(评分仅超门槛%s)，跳过 %s 第%s次加仓", score_margin, name, idx + 1)
                                        add_skip = True
                                if not add_skip:
                                    add_ratio = add_ratios[idx]
                                    trade = self._buy_position(code, name, current_price, add_ratio,
                                        f"第{idx+1}次加仓·评分{score}", add_count=pos.add_count + 1)

                # ── 回踩均线加仓：价格回踩MA10/MA20不破反弹时加仓 ──
                if not trade and pos and pos.profit_pct > 0 and pos.add_count < 3:
                    if kline is not None and len(kline) > 20:
                        closes_arr = kline["close"].values.astype(float)
                        from src.signals import _sma
                        ma10_v = _sma(closes_arr, 10)
                        ma20_v = _sma(closes_arr, 20)
                        valid = ~np.isnan(ma10_v) & ~np.isnan(ma20_v)
                        if len(ma10_v[valid]) > 0:
                            ma10 = ma10_v[valid][-1]
                            ma20 = ma20_v[valid][-1]
                            dev_ma10 = (current_price / ma10 - 1) * 100
                            dev_ma20 = (current_price / ma20 - 1) * 100
                            # 条件：价格在MA10上方0~2%（回踩不破）或在MA20上方0~1%（深度回踩支撑）
                            near_ma = (0 <= dev_ma10 <= 2) or (0 <= dev_ma20 <= 1)
                            if near_ma and score >= 45 and not (prediction and prediction["direction"] == "看跌"):
                                # 检查反弹力度：当前价 > 前一根K线收盘价（正在反弹）
                                if len(closes_arr) >= 2 and current_price > closes_arr[-2]:
                                    add_chg = score_info.get("change_pct", 0)
                                    logger.info("回踩均线加仓: %s MA10=%.2f MA20=%.2f 现价=%.2f 偏离MA10=%.1f%% 评分=%s",
                                                name, ma10, ma20, current_price, dev_ma10, score)
                                    trade = self._buy_position(code, name, current_price, 0.10,
                                        f"回踩MA10加仓·评分{score}", add_count=pos.add_count + 1)

                # ── 放量突破加仓：涨幅>3%+放量>1.5倍+评分↑ ──
                if not trade and pos and pos.profit_pct > 0 and pos.add_count < 3:
                    intraday_chg = score_info.get("change_pct", 0)
                    if kline is not None and intraday_chg >= 3 and score >= 55:
                        volumes_arr = kline["volume"].values.astype(float) if "volume" in kline.columns else np.array([])
                        if len(volumes_arr) >= 5:
                            avg_v = np.mean(volumes_arr[-5:-1])
                            if avg_v > 0:
                                vol_ratio = volumes_arr[-1] / avg_v
                                if vol_ratio >= 1.5 and not (prediction and prediction["direction"] == "看跌"):
                                    logger.info("放量突破加仓: %s 涨幅%.1f%% 量比%.1f 评分=%s",
                                                name, intraday_chg, vol_ratio, score)
                                    trade = self._buy_position(code, name, current_price, 0.10,
                                        f"放量突破加仓·评分{score}", add_count=pos.add_count + 1)

                # ── RSI超卖加仓：RSI<30+均线多头未破，回调机会 ──
                if not trade and pos and pos.profit_pct > 0 and pos.add_count < 3:
                    if kline is not None and len(kline) > 20:
                        from src.signals import _calc_rsi
                        closes_arr = kline["close"].values.astype(float)
                        rsi_val = _calc_rsi(closes_arr, 14)
                        if rsi_val is not None and rsi_val < 30 and score >= 50:
                            # 检查均线趋势未破：MA5>MA20
                            ma5_v = _sma(closes_arr, 5)
                            ma20_v = _sma(closes_arr, 20)
                            valid = ~np.isnan(ma5_v) & ~np.isnan(ma20_v)
                            if len(ma5_v[valid]) > 0 and ma5_v[valid][-1] > ma20_v[valid][-1]:
                                if not (prediction and prediction["direction"] == "看跌"):
                                    logger.info("RSI超卖加仓: %s RSI=%.0f 评分=%s", name, rsi_val, score)
                                    trade = self._buy_position(code, name, current_price, 0.08,
                                        f"RSI超卖加仓·评分{score}", add_count=pos.add_count + 1)

                # 亏损中摊平：亏损>5%且评分仍>=50时，低仓位补仓
                elif pos and pos.profit_pct < -5 and score >= 50:
                    loss = abs(pos.profit_pct)
                    if pos.add_count < 3:  # 最多摊平3次
                        ratio = min(0.05, 0.02 * (loss / 5))  # 亏越多补越多，但最多5%
                        if ratio >= 0.03 and current_price < pos.buy_price * 0.97:
                            # 大盘+预测检查（摊平需要明确看涨）
                            add_skip = False
                            if prediction:
                                if prediction["direction"] != "看涨":
                                    logger.info("预测%s，跳过 %s 摊平", prediction["direction"], name)
                                    add_skip = True
                            elif market_declining:
                                if deep_drop:
                                    logger.info("大盘跌势+深跌(%.1f%%)，允许 %s 摊平", intraday_chg, name)
                                else:
                                    logger.info("大盘跌势无明确看涨信号，跳过 %s 摊平", name)
                                    add_skip = True
                            if not add_skip:
                                trade = self._buy_position(code, name, current_price, ratio,
                                    f"摊平{loss:.0f}%·补仓", add_count=pos.add_count + 1)

        # ── 卖出（评分<40 或 自适应止损）— T+1限制 ──
        if not trade and code in self.portfolio.positions:
            today_str = date.today().isoformat()
            # T+1: 当天买入不能当天卖出
            if pos.buy_date == today_str:
                pass
            elif score < 35:
                # 趋势保护：如果均线多头，评分低也只卖一半
                kline_ma5 = score_info.get("ma5", 0)
                kline_ma20 = score_info.get("ma20", 0)
                if kline_ma5 > kline_ma20 and score >= 30:
                    # 趋势向上但评分略低，卖一半保留另一半
                    sell_shares = pos.shares // 2
                    if sell_shares >= 100:
                        trade = self._sell_partial(code, current_price, sell_shares, f"评分{score}分·减半")
                    else:
                        trade = self._sell_position(code, current_price, f"评分{score}分·回避")
                else:
                    trade = self._sell_position(code, current_price, f"评分{score}分·回避")
            else:
                # 自适应止损（基于ATR）
                atr_value = score_info.get("atr", 0)
                if atr_value > 0:
                    stop_price = pos.buy_price - atr_value * 2.0
                    if current_price <= stop_price:
                        loss = (current_price / pos.buy_price - 1) * 100
                        trade = self._sell_position(code, current_price, f"ATR止损{loss:.1f}%(ATR={atr_value:.2f})")
                else:
                    # 无ATR数据时使用固定止损
                    loss = (current_price / pos.buy_price - 1) * 100
                    if loss <= -8:
                        trade = self._sell_position(code, current_price, f"固定止损{loss:.1f}%")

        self._update_value()
        return trade

    def _sell_partial(self, code, price, shares, reason):
        """部分卖出（含T+1检查）"""
        if code not in self.portfolio.positions:
            return None
        pos = self.portfolio.positions[code]
        # T+1: 当天买入不能当天卖出
        if pos.buy_date == date.today().isoformat():
            return None
        shares = min(shares, pos.shares)
        if shares <= 0:
            return None
        fee = self._calc_commission(shares * price, code) + shares * price * self.stamp_duty
        sell_value = shares * price - fee
        profit_pct = (sell_value / (pos.total_cost * shares / pos.shares) - 1) * 100 if pos.total_cost > 0 else 0
        # 更新持仓
        remaining = pos.shares - shares
        if remaining <= 0:
            return self._sell_position(code, price, reason)
        cost_ratio = remaining / pos.shares
        pos.shares = remaining
        pos.total_cost *= cost_ratio
        pos.market_value = remaining * price
        pos.current_price = price
        self.portfolio.cash += sell_value
        self._update_value()
        trade = TradeRecord(
            stock_code=code, stock_name=pos.stock_name,
            action="卖出(部分)", date=date.today().isoformat(),
            price=round(price, 2), shares=shares,
            profit_pct=round(profit_pct, 2),
            profit_amount=round(sell_value - (pos.total_cost / cost_ratio - pos.total_cost), 2),
            reason=reason,
        )
        self.portfolio.trades.append(trade)
        self._save()
        logger.info("部分卖出: %s %s股 %.2f元 %s", pos.stock_name, shares, price, reason)
        return trade

    def _calc_commission(self, amount: float, code: str = "") -> float:
        """计算佣金（东方财富规则）
        股票：万2.5，最低5元
        ETF/基金：万1，最低0.1元
        """
        is_etf = code.startswith(("51", "52", "15", "16"))
        if is_etf:
            comm = max(amount * self.etf_commission, self.min_etf_commission)
        else:
            comm = max(amount * self.commission, self.min_commission)
        return comm + amount * self.transfer_fee

    def _buy_position(self, code, name, price, ratio, reason, add_count=0):
        """买入"""
        if len(self.portfolio.positions) >= self.max_positions and code not in self.portfolio.positions:
            return None
        amount = self.portfolio.cash * ratio
        if amount < 1000:
            return None
        shares = int(amount / price / 100) * 100
        if shares < 100:
            return None
        total_cost = shares * price + self._calc_commission(shares * price, code)
        self.portfolio.cash -= total_cost

        if code in self.portfolio.positions:
            # 加仓：合并持仓
            old = self.portfolio.positions[code]
            new_shares = old.shares + shares
            avg_price = (old.total_cost + total_cost) / new_shares / (1 + self.commission)
            self.portfolio.positions[code] = Position(
                stock_code=code, stock_name=name,
                buy_date=date.today().isoformat(), buy_price=round(avg_price, 2),
                shares=new_shares, total_cost=old.total_cost + total_cost,
                current_price=price, market_value=new_shares * price,
                peak_price=max(old.peak_price, price),
                add_count=add_count or old.add_count,
            )
        else:
            self.portfolio.positions[code] = Position(
                stock_code=code, stock_name=name,
                buy_date=date.today().isoformat(), buy_price=price,
                shares=shares, total_cost=total_cost,
                current_price=price, market_value=shares * price,
                peak_price=price,
            )
        action_label = "加仓" if code in self.portfolio.positions else "买入"
        logger.info("模拟%s: %s %s股 %.2f元 ratio=%.0f%%", action_label, name, shares, price, ratio * 100)
        trade = TradeRecord(
            stock_code=code, stock_name=name,
            action=action_label, date=date.today().isoformat(),
            price=round(price, 2), shares=shares, reason=reason,
        )
        self.portfolio.trades.append(trade)
        self._update_value()
        self._save()
        return trade

    def _sell_position(self, code, price, reason):
        """卖出（含T+1检查）"""
        if code not in self.portfolio.positions:
            return None
        pos = self.portfolio.positions[code]
        today_str = date.today().isoformat()
        # T+1: 当天买入不能当天卖出
        if pos.buy_date == today_str:
            logger.warning("T+1拦截: %s 今日买入 禁止卖出", pos.stock_name)
            return None
        trade_value = pos.shares * price
        fee = self._calc_commission(trade_value, code) + trade_value * self.stamp_duty
        sell_value = trade_value - fee
        profit_pct = (sell_value / pos.total_cost - 1) * 100 if pos.total_cost > 0 else 0
        profit_amount = sell_value - pos.total_cost
        self.portfolio.cash += sell_value
        del self.portfolio.positions[code]
        logger.info("模拟卖出: %s %.2f元 盈亏%+.2f%%", pos.stock_name, price, profit_pct)
        trade = TradeRecord(
            stock_code=code, stock_name=pos.stock_name,
            action="卖出", date=date.today().isoformat(),
            price=round(price, 2), shares=pos.shares,
            profit_pct=round(profit_pct, 2),
            profit_amount=round(profit_amount, 2),
            reason=reason,
        )
        self.portfolio.trades.append(trade)
        self._update_value()
        self._save()
        return trade

    def update_prices(self, prices: dict[str, float]):
        """更新持仓股票的当前价，计算实时市值，并记录净值"""
        for code, pos in list(self.portfolio.positions.items()):
            price = prices.get(code)
            if price:
                pos.current_price = price
                pos.market_value = pos.shares * price
                pos.profit_pct = round((price / pos.buy_price - 1) * 100, 2)
                pos.profit_amount = round(pos.market_value - pos.total_cost, 2)
                # 更新最高价（用于移动止盈）
                if price > pos.peak_price:
                    pos.peak_price = price
        self._update_value()
        self._record_value()
        self._save()

    def _record_value(self):
        """记录当前净值到时间序列"""
        today_str = date.today().isoformat()
        # 如果今天已记录则更新，否则新增
        for i, entry in enumerate(self.portfolio.daily_values):
            if entry["date"] == today_str:
                self.portfolio.daily_values[i]["value"] = self.portfolio.total_value
                return
        self.portfolio.daily_values.append({
            "date": today_str,
            "value": self.portfolio.total_value,
        })

    def _update_value(self, latest_price: float = 0):
        """更新总资产"""
        pos_value = sum(p.market_value for p in self.portfolio.positions.values())
        self.portfolio.total_value = round(self.portfolio.cash + pos_value, 2)

    def generate_report(self) -> str:
        """生成模拟交易报告"""
        lines = []
        today_str = date.today().strftime("%m/%d")
        lines.append(f"📋 **模拟交易账户** · {today_str}")
        lines.append("")

        # 账户概况
        init_cash = 100000
        total_ret = (self.portfolio.total_value - init_cash) / init_cash * 100
        ret_icon = "📈" if total_ret > 0 else "📉"
        lines.append(f"{ret_icon} **总资产: {self.portfolio.total_value:,.2f}元**")
        lines.append(f"   初始资金: 100,000元")
        lines.append(f"   现金: {self.portfolio.cash:,.2f}元")
        lines.append(f"   持仓市值: {self.portfolio.total_value - self.portfolio.cash:,.2f}元")
        lines.append("")

        # 当前持仓
        if self.portfolio.positions:
            lines.append(f"**当前持仓**")
            for code, pos in self.portfolio.positions.items():
                p_icon = "🟢" if pos.profit_pct >= 0 else "🔴"
                # 计算今日盈亏
                rt = None
                try:
                    from src.fetcher import fetch_realtime
                    rt = fetch_realtime(code)
                    if rt and rt.get("yesterday_close", 0) > 0:
                        yc = rt["yesterday_close"]
                        today_chg = (pos.current_price / yc - 1) * 100
                        today_profit = pos.shares * (pos.current_price - yc)
                    else:
                        today_chg = 0; today_profit = 0
                except: today_chg = 0; today_profit = 0
                today_icon = "📈" if today_chg >= 0 else "📉"
                lines.append(f"  {p_icon} {pos.stock_name}({code})")
                lines.append(f"     买入: {pos.buy_price:.2f}元  |  现价: {pos.current_price:.2f}元")
                lines.append(f"     持仓: {pos.shares}股  |  市值: {pos.market_value:,.0f}元")
                lines.append(f"     总盈亏: {pos.profit_pct:+.2f}%  |  {today_icon} 今日: {today_profit:+.0f}元")
            lines.append("")

        # 最近交易
        recent = [t for t in self.portfolio.trades if t.date == date.today().isoformat()]
        if recent:
            lines.append(f"**今日交易**")
            for t in recent[-5:]:
                icon = "🟢" if t.action == "买入" else ("🟢" if t.profit_pct >= 0 else "🔴")
                if "卖出" in t.action:
                    p_str = f"  {t.profit_amount:+.2f}元" if t.profit_amount else ""
                else:
                    p_str = f"  {t.profit_pct:+.2f}%" if t.profit_pct else ""
                lines.append(f"  {icon} {t.action} {t.stock_name} {t.price:.2f}元 {t.shares}股{p_str}")
            lines.append("")

        # 历史交易统计
        if self.portfolio.trades:
            sells = [t for t in self.portfolio.trades if t.action == "卖出"]
            wins = [t for t in sells if t.profit_pct > 0]
            total_trades = len(sells)
            win_rate = round(len(wins) / total_trades * 100, 1) if total_trades > 0 else 0
            lines.append(f"**历史统计**")
            lines.append(f"   总交易: {total_trades}次  |  胜率: {win_rate}%")
            lines.append(f"   盈利: {len(wins)}次  |  亏损: {total_trades - len(wins)}次")

        return "\n".join(lines).strip()
