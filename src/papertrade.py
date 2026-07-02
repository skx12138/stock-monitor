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

import numpy as np

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
    hold_since: str = ""           # 最早买入日期（用于做T判断）


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

    def __init__(self, initial_cash: float = 500000):
        self.portfolio = Portfolio(cash=initial_cash, total_value=initial_cash)
        self.trade_dedup: dict[str, datetime] = {}
        self.trade_cooldown = 0
        self.max_positions = 999            # 不限制持仓数量
        self.max_total_ratio = 0.80         # 总仓位上限80%
        self.max_sector_ratio = 0.40         # 单板块仓位上限40%
        self.commission = 0.00025      # 股票佣金万2.5
        self.etf_commission = 0.0001    # ETF/基金佣金万1
        self.stamp_duty = 0.001        # 印花税千1（卖出收）
        self.transfer_fee = 0.00001    # 过户费万0.1
        self.min_commission = 5.0      # 股票最低佣金5元
        self.min_etf_commission = 0.1  # ETF最低佣金0.1元
        self.trail_activate = 6.0    # 盈利6%后启动移动止盈（让利润多跑）
        self.trail_pullback = 5.0    # 从高点回撤5%触发止盈
        self.enable_volatility_adjust = True
        self.enable_sector_filter = True
        self._sector_cache = None
        self._sector_cache_time = 0
        self._load()
        self.buy_recommendations: list[str] = []  # 推荐买入但因资金不足未成交的股票
        self._messages = None  # 外部消息列表钩子，用于做T通知

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
        """根据信号执行模拟交易（基于信号方向）

        规则:
          - bullish(看涨) → 未持仓时买入，仓位按信号强度10%~20%
          - bearish(看跌) → 已持仓时卖出
          - neutral(中性) → 不操作
        """
        if signal.direction == "bullish" and signal.stock_code not in self.portfolio.positions:
            ratio = 0.20 if signal.signal_type in ("ma_crossover", "macd") else 0.10
            self._buy_position(signal.stock_code, signal.stock_name, current_price, ratio,
                               f"信号买入·{signal.signal_label}·{signal.suggestion}")
        elif signal.direction == "bearish" and signal.stock_code in self.portfolio.positions:
            self._sell_position(signal.stock_code, current_price,
                                f"信号卖出·{signal.signal_label}·{signal.suggestion}")

    def process_score(self, code: str, name: str, current_price: float,
                      score_info: dict, kline=None) -> Optional[TradeRecord]:
        """根据评分系统执行模拟交易（V5版 — 匹配新评分体系）

        规则:
          - 动态仓位: V5评分60买10%, 65买15%, 70买20%, 75买25%, 80买30%
          - 分批止盈: 盈利12%卖1/4, 20%再卖1/4, 30%清仓
          - 自适应止损: 基于ATR动态计算止损位
          - 卖出条件: 评分<45(回避)或触发止损/止盈
        """
        score = score_info.get("score", 0)
        action = score_info.get("action", "")
        now = datetime.now()
        
        # 加载本股票的个性化策略参数
        try:
            from src.optimizer import get_stock_params
            sp = get_stock_params(code)
            buy_th = sp.get("buy_threshold", 60)
            sell_th = sp.get("sell_threshold", 45)
            stop_loss_pct = sp.get("stop_loss", 8)
        except:
            buy_th = 60
            sell_th = 45
            stop_loss_pct = 8

        # ── 风控1：当日总亏损超过8%时暂停所有新开仓 ──
        daily_loss_limit = -8.0
        current_day_ret = (self.portfolio.total_value - 100000) / 100000 * 100
        if current_day_ret < daily_loss_limit and code not in self.portfolio.positions:
            logger.warning("风控: 当日总亏损%.1f%%超过阈值%.0f%%，暂停新开仓 %s", current_day_ret, daily_loss_limit, name)
            return None

        # ── 风控2：大盘暴跌(>3%)时自动减半仓 ──
        try:
            from src.scoring import get_market_sentiment
            s_lv, s_label = get_market_sentiment()
            if s_lv == -2:  # 恐慌
                if code in self.portfolio.positions:
                    pos = self.portfolio.positions.get(code)
                    if pos and pos.shares > 100:
                        sell_shares = pos.shares // 2
                        if sell_shares >= 100:
                            logger.warning("风控: 市场恐慌[%s]，%s 自动减半仓%d股", s_label, name, sell_shares)
                            return self._sell_partial(code, current_price, sell_shares, f"恐慌减半·{s_label}")
        except: pass

        # ── 风控3：连续3日下跌暂停加仓 ──
        consecutive_days_down = 0
        if kline is not None and len(kline) >= 5 and code in self.portfolio.positions:
            try:
                c_closes = kline["close"].values.astype(float)
                for i in range(1, min(6, len(c_closes))):
                    if c_closes[-i] < c_closes[-i-1]:
                        consecutive_days_down += 1
                    else:
                        break
                if consecutive_days_down >= 3 and code in self.portfolio.positions:
                    logger.info("风控: %s 连续%d日下跌，跳过加仓", name, consecutive_days_down)
            except: pass

        last_trade = self.trade_dedup.get(code)
        if last_trade:
            elapsed = (now - last_trade).total_seconds() / 60
            if elapsed < self.trade_cooldown:
                return None
        self.trade_dedup[code] = now

        trade = None
        pos = self.portfolio.positions.get(code)
        now = datetime.now()
        # 清空之前推荐的记录（每轮开始时清空一次）
        if hasattr(self, '_rec_cleared'):
            pass
        else:
            self.buy_recommendations = []
            self._rec_cleared = True

        # ── 量价形态调节（放量上涨加仓，放量下跌减仓） ──
        vol_adj = 1.0
        try:
            vol_detail = score_info.get("details", {}).get("成交量", {})
            vol_desc = vol_detail.get("desc", "")
            if "放量上涨" in vol_desc:
                vol_adj = 1.2  # 放量上涨，加2成仓位
            elif "缩量回调" in vol_desc or "惜售" in vol_desc:
                vol_adj = 1.1  # 缩量回调/惜售，加1成
            elif "放量下跌" in vol_desc or "资金出逃" in vol_desc:
                vol_adj = 0.4  # 放量下跌，打4折
            elif "缩量上涨" in vol_desc and "动力不足" in vol_desc:
                vol_adj = 0.6  # 缩量上涨动力不足，打6折
            elif "无人参与" in vol_desc:
                vol_adj = 0.3  # 极度缩量无人参与，打3折
        except:
            pass
        sector_ok = True
        sector_adj = 1.0
        try:
            sector = self._get_sector_tag(code)
            if sector:
                from src.fetcher import fetch_sector_performance
                sectors = fetch_sector_performance()
                if sectors:
                    for s in sectors:
                        if s["name"] == sector or sector in s["name"] or s["name"] in sector:
                            chg = s.get("change_pct", 0)
                            if chg > 2:
                                sector_adj = 1.2    # 热点板块+20%仓位
                                logger.info("板块[%s]涨幅%.1f%%，%s 仓位加2成", sector, chg, name)
                            elif chg < -2:
                                sector_adj = 0.5    # 弱势板块减半
                                sector_ok = False if chg < -3 else True
                                logger.info("板块[%s]跌幅%.1f%%，%s 仓位打5折", sector, chg, name)
                            break
        except Exception as e:
            logger.debug("板块过滤失败: %s", e)

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

        # ── 动态仓位（匹配V5评分体系） ──
        #   V5评分 buy_th(60)=10%  65=15%  70=20%  75=25%  80+=30%
        def get_ratio(s):
            if s >= buy_th + 20: return 0.30 * vol_ratio
            if s >= buy_th + 15: return 0.25 * vol_ratio
            if s >= buy_th + 10: return 0.20 * vol_ratio
            if s >= buy_th + 5:  return 0.15 * vol_ratio
            if s >= buy_th:      return 0.10 * vol_ratio
            return 0

        # ── 移动止盈 ──
        if pos:
            if current_price > pos.peak_price:
                pos.peak_price = current_price
            profit = (current_price / pos.buy_price - 1) * 100
            
            # 冲高回落形态：收紧移动止盈
            from src.scoring import get_intraday_trend
            try:
                trend_desc, _ = get_intraday_trend()
                is_reversal = trend_desc.startswith("冲高回落")
                is_recovery = trend_desc.startswith("探底回升")
            except:
                is_reversal = False
                is_recovery = False
            
            # 动态移动止盈：冲高回落时收紧，探底回升时放宽
            if is_reversal:
                trail_activate = min(self.trail_activate, 3.0)   # 冲高回落: 盈利3%就启动
                trail_pull = min(self.trail_pullback, 3.0)       # 回撤3%就卖
            elif is_recovery:
                trail_activate = max(self.trail_activate, 8.0)   # 探底回升: 盈利8%再启动
                trail_pull = self.trail_pullback + 4              # 多给4%回撤空间
            else:
                trail_activate = self.trail_activate
                ma5_vv = score_info.get("ma5", 0)
                ma20_vv = score_info.get("ma20", 0)
                if ma5_vv > ma20_vv:
                    trail_pull = self.trail_pullback + 3  # 多头趋势多给3%回撤空间
                else:
                    trail_pull = self.trail_pullback
            if profit >= trail_activate:
                pullback = (pos.peak_price - current_price) / pos.peak_price * 100
                if pullback >= trail_pull:
                    return self._sell_position(code, current_price, f"移动止盈(从高点回撤{pullback:.1f}%)")

            # ── 分批止盈（让利润多跑一会儿，抬高出货门槛） ──
            sell_shares = 0
            profit_str = ""
            if profit >= 30:
                sell_shares = pos.shares  # 全清
                profit_str = f"止盈{profit:.1f}%清仓"
            elif profit >= 20 and pos.shares >= 200:
                sell_shares = min(100, pos.shares // 2)  # 至少卖100股
                profit_str = f"止盈{profit:.1f}%卖{sell_shares}股"
            elif profit >= 12 and pos.shares >= 200:
                sell_shares = min(100, pos.shares // 2)  # 至少卖100股
                profit_str = f"止盈{profit:.1f}%卖{sell_shares}股"

            if sell_shares > 0:
                trade = self._sell_partial(code, current_price, sell_shares, profit_str)

        # ── 做T策略：先卖后买，赚取日内差价 ──
        T_SHARES = 200  # 每次做T股数
        if not trade and pos and pos.shares >= 200:
            today_str = date.today().isoformat()
            intraday_chg = score_info.get("change_pct", 0)
            # 检查是否有昨日持仓可做T
            can_t_trade = pos.shares  # 全部可T卖出，因为buy_date被更新为今日
            if pos.hold_since and pos.hold_since < today_str:
                can_t_trade = pos.shares
            # 初始化T交易记录
            if not hasattr(self, '_t_records'):
                self._t_records = {}
            t_key = f"{code}_{today_str}"
            t_info = self._t_records.get(t_key, {"sold": 0, "buy_price": 0})
            # T卖点：日内涨超3%，卖出T_SHARES股
            if intraday_chg >= 3.0 and t_info["sold"] == 0 and can_t_trade >= 400:
                from datetime import time as _dt_time
                now_t = datetime.now().time()
                if _dt_time(9, 30) <= now_t <= _dt_time(14, 30):
                    t_info["sold"] = T_SHARES
                    t_info["buy_price"] = current_price
                    self._t_records[t_key] = t_info
                    trade = self._sell_partial(code, current_price, T_SHARES, f"做T卖出+{intraday_chg:.1f}%")
                    logger.info("做T卖出: %s +%.1f%% 卖%d股 等回调接回", name, intraday_chg, T_SHARES)
                    if hasattr(self, '_messages') and self._messages is not None:
                        self._messages.append(f"  🔄 做T卖出 {name}({code}) {current_price:.2f}元×{T_SHARES}股")
            # T买点：日内跌超1.5%或有T仓位未回补，买回
            if not trade and t_info["sold"] > 0:
                should_buy_back = False
                buy_reason = ""
                # 跌超1.5%接回
                if intraday_chg <= -1.5:
                    should_buy_back = True
                    buy_reason = f"做T买入(跌{intraday_chg:.1f}%)"
                # 收盘前强制接回（14:50后）
                from datetime import time as _dt_time
                now_t = datetime.now().time()
                if _dt_time(14, 50) <= now_t <= _dt_time(15, 0):
                    should_buy_back = True
                    buy_reason = "做T买入(收盘强制接回)"
                if should_buy_back:
                    t_info["sold"] = 0
                    self._t_records[t_key] = t_info
                    # 直接买回（用卖出时的金额等量买回）
                    buy_amount = t_info.get("buy_price", current_price) * T_SHARES
                    if self.portfolio.cash >= buy_amount * 1.01:
                        shares = T_SHARES
                        cost = shares * current_price + self._calc_commission(shares * current_price, code)
                        self.portfolio.cash -= cost
                        old = self.portfolio.positions.get(code)
                        if old:
                            new_shares = old.shares + shares
                            avg_price = (old.total_cost + cost) / new_shares
                            self.portfolio.positions[code] = Position(
                                stock_code=code, stock_name=name,
                                buy_date=old.buy_date, buy_price=round(avg_price, 2),
                                shares=new_shares, total_cost=old.total_cost + cost,
                                current_price=current_price, market_value=new_shares * current_price,
                                peak_price=max(old.peak_price, current_price),
                                add_count=old.add_count, hold_since=old.hold_since,
                            )
                        self._update_value()
                        self._save()
                        logger.info("做T买入: %s %.2f元 %s", name, current_price, buy_reason)
                        if hasattr(self, '_messages') and self._messages is not None:
                            self._messages.append(f"  🔄 做T买入 {name}({code}) {current_price:.2f}元×{T_SHARES}股")

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

        # ── 早盘保护（开盘后30分钟内仓位减半，避免追高被套） ──
        morning_adj = 1.0
        try:
            from datetime import time as _dt_time
            now_t = datetime.now().time()
            if _dt_time(9, 30) <= now_t <= _dt_time(10, 0):
                morning_adj = 0.5
                logger.info("早盘保护期(9:30-10:00)，%s 仓位打5折", name)
        except:
            pass
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

        # ── 大盘日内趋势（高开低走/低开高走/单边行情） ──
        intraday_adj = 1.0
        try:
            from src.scoring import get_intraday_trend
            trend_desc, intensity = get_intraday_trend()
            if trend_desc.startswith("高开低走"):
                intraday_adj = 0.5  # 高开低走，尾盘大概率继续弱
                logger.info("大盘%s(强度%.1f)，%s 仓位打5折防尾盘跳水", trend_desc, intensity, name)
            elif trend_desc.startswith("冲高回落"):
                intraday_adj = 0.4  # 早涨下午大跌，最危险形态
                logger.info("大盘%s(强度%.1f)，%s 仓位打4折防下午跳水", trend_desc, intensity, name)
            elif trend_desc.startswith("单边下跌"):
                intraday_adj = 0.4  # 单边下跌不抄底
                logger.info("大盘%s(强度%.1f)，%s 仓位打4折", trend_desc, intensity, name)
            elif trend_desc.startswith("低开高走"):
                intraday_adj = 1.2  # 低开高走，尾盘可积极些
                logger.info("大盘%s(强度%.1f)，%s 仓位加2成", trend_desc, intensity, name)
            elif trend_desc.startswith("探底回升"):
                intraday_adj = 1.3  # 探底回升，最强势形态
                logger.info("大盘%s(强度%.1f)，%s 仓位加3成", trend_desc, intensity, name)
            elif trend_desc.startswith("单边上涨"):
                intraday_adj = 0.8  # 单边上涨不追高
                logger.info("大盘%s(强度%.1f)，%s 仓位打8折防追高", trend_desc, intensity, name)
        except Exception as e:
            logger.debug("日内趋势判断失败: %s", e)

        # ── 市场情绪调节（恐慌降仓，狂热谨慎） ──
        sentiment_adj = 1.0
        try:
            from src.scoring import get_market_sentiment
            s_level, s_label = get_market_sentiment()
            if s_level == -2:    # 恐慌
                sentiment_adj = 0.3
                logger.info("市场情绪[%s]，%s 仓位降至30%%", s_label, name)
            elif s_level == -1:  # 恐惧
                sentiment_adj = 0.6
                logger.info("市场情绪[%s]，%s 仓位打6折", s_label, name)
            elif s_level == 1:   # 贪婪
                sentiment_adj = 0.7
                logger.info("市场情绪[%s]，%s 追高仓位打7折", s_label, name)
            elif s_level == 2:   # 狂热
                sentiment_adj = 0.4
                logger.info("市场情绪[%s]，%s 狂热期仓位降至40%%", s_label, name)
        except Exception as e:
            logger.debug("情绪判断失败: %s", e)

        # ── 深跌反弹机会：当日跌幅巨大时放宽大盘/预测过滤 ──
        intraday_chg = score_info.get("change_pct", 0)
        deep_drop = intraday_chg < -5  # 默认5%（会被ATR动态覆盖）
        heavy_drop = intraday_chg < -8  # 默认8%（会被ATR动态覆盖）
        # 用ATR动态计算暴跌阈值（波动大的股票容忍度更高）
        try:
            from src.signals import calc_atr
            if kline is not None and len(kline) > 20 and "high" in kline.columns and "low" in kline.columns:
                c_atr = kline["close"].values.astype(float)
                h_atr = kline["high"].values.astype(float)
                l_atr = kline["low"].values.astype(float)
                atr_val = calc_atr(c_atr, h_atr, l_atr, 14)
                if atr_val > 0 and current_price > 0:
                    atr_pct = atr_val / current_price * 100  # ATR百分比
                    deep_thresh = max(-atr_pct * 2.0, -10)   # 2倍ATR，最多-10%
                    heavy_thresh = max(-atr_pct * 3.0, -15)  # 3倍ATR，最多-15%
                    deep_drop = intraday_chg < deep_thresh
                    heavy_drop = intraday_chg < heavy_thresh
                    logger.info("ATR动态阈值: %s ATR=%.1f%% 深跌%.0f%% 暴跌%.0f%%", name, atr_pct, deep_thresh, heavy_thresh)
        except:
            pass

        # ── 暴跌加仓分析：检查近几日趋势 ──
        drop_analysis = ""
        if heavy_drop and pos and kline is not None and len(kline) >= 5:
            try:
                closes_arr = kline["close"].values.astype(float)
                # 检查近5日走势
                recent_closes = closes_arr[-5:]
                days_down = sum(1 for i in range(1, len(recent_closes)) if recent_closes[i] < recent_closes[i-1])
                total_chg_5d = (recent_closes[-1] / recent_closes[0] - 1) * 100
                if days_down >= 4 and total_chg_5d < -15:
                    # 连跌4天+累计跌超15%=加速赶底
                    drop_analysis = "连日暴跌加速赶底"
                    logger.info("暴跌分析: %s 连跌%d天累计%.1f%%，加速赶底可加仓", name, days_down, total_chg_5d)
                elif days_down <= 1 and total_chg_5d > -5:
                    # 之前一直在涨/横盘，今天突然暴跌=恐慌错杀
                    drop_analysis = "恐慌错杀"
                    logger.info("暴跌分析: %s 今日暴跌但近5日仅跌%.1f%%，恐慌错杀可抄底", name, total_chg_5d)
                elif days_down >= 3:
                    # 连跌3天=持续下跌中
                    drop_analysis = "持续下跌中"
                    logger.info("暴跌分析: %s 连跌%d天，等企稳再考虑", name, days_down)
                else:
                    drop_analysis = "震荡下跌"
            except Exception as e:
                logger.debug("暴跌分析失败: %s", e)

        # ── 多周期确认：日K线均线多头才买入（减少假信号） ──
        daily_bullish = True
        try:
            from src.fetcher import fetch_kline as _fk
            dk = _fk(code, 365)
            if dk is not None and len(dk) > 20:
                dc = dk["close"].values.astype(float)
                dma5 = _sma(dc, 5)
                dma20 = _sma(dc, 20)
                dv = ~np.isnan(dma5) & ~np.isnan(dma20)
                if len(dma5[dv]) > 0:
                    daily_bullish = dma5[dv][-1] > dma20[dv][-1]
        except:
            pass

        # ── 60分钟K线确认：短线趋势配合才买入（减少盘中假突破） ──
        minute60_bullish = True
        try:
            from src.fetcher import fetch_kline as _fk
            mk = _fk(code, 30, scale=60)  # 60分钟K线，30根≈7.5个交易日
            if mk is not None and len(mk) > 10:
                mc = mk["close"].values.astype(float)
                mma5 = _sma(mc, 5)
                mma20 = _sma(mc, 20)
                mv = ~np.isnan(mma5) & ~np.isnan(mma20)
                if len(mma5[mv]) > 0:
                    minute60_bullish = mma5[mv][-1] > mma20[mv][-1]
                    if not minute60_bullish:
                        logger.info("60分钟K线MA5<MA20，短线偏弱，%s 需更谨慎", name)
        except Exception as e:
            logger.debug("60分钟K线获取失败 %s: %s", name, e)

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

        # ── 买入（动态仓位 + 大盘/预测过滤 + 情绪调节 + 日内趋势） ──
        if not trade:
            ratio = get_ratio(score) * chase_penalty * sentiment_adj * intraday_adj * sector_adj * morning_adj * vol_adj
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
                    # 日K线多头检查（减少假信号）
                    if (not daily_bullish or not minute60_bullish) and not deep_drop and not limit_down:
                        if not daily_bullish:
                            logger.info("日K线趋势向下，%s 跳过买入", name)
                        else:
                            logger.info("60分K线趋势向下，%s 跳过买入", name)
                        ratio = 0
                if ratio > 0:
                    trade = self._buy_position(code, name, current_price, ratio, 
                        f"评分{score}分买入{ratio*100:.0f}%仓位·{prediction['direction'] if prediction else '无预测'}·大盘{'跌' if market_declining else '稳'}",
                        add_count=0)
            elif ratio > 0 and code in self.portfolio.positions and score >= 65:
                pos = self.portfolio.positions.get(code)
                if pos and (pos.profit_pct > 0 or (pos.profit_pct > -5 and pos.add_count < 2)) and consecutive_days_down < 3:  # 盈利或浅亏(<5%)允许加仓
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
                                    # 量价共振作为软提示而非硬阻拦（评分远超门槛(≥10)或深跌时无条件通过）
                                    details = score_info.get("details", {})
                                    macd_ok = details.get("MACD", {}).get("score", 0) > 0
                                    vol_ok = details.get("成交量", {}).get("score", 0) > 0
                                    score_margin = score - add_scores[idx]
                                    if not macd_ok and not vol_ok and score_margin < 10 and not deep_drop:
                                        logger.info("无量价共振但评分超门槛%s，%s 第%s次加仓减半", score_margin, name, idx + 1)
                                        add_ratios[idx] *= 0.5  # 减半而不是跳过
                                    # 日K线向下时加仓需谨慎
                                    if not daily_bullish and not add_skip:
                                        logger.info("日K线趋势向下，%s 第%s次加仓减半", name, idx + 1)
                                        add_ratios[idx] *= 0.5
                                if not add_skip:
                                    add_ratio = add_ratios[idx] * sentiment_adj * intraday_adj * sector_adj * morning_adj * vol_adj
                                    trade = self._buy_position(code, name, current_price, add_ratio,
                                        f"第{idx+1}次加仓·评分{score}", add_count=pos.add_count + 1)

                # ── 回踩均线加仓：价格回踩MA10/MA20不破反弹时加仓（浅亏也允许） ──
                if not trade and pos and (pos.profit_pct > 0 or (pos.profit_pct > -3 and deep_drop)) and pos.add_count < 3 and consecutive_days_down < 3:
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
                            # MA20趋势：MA20上升=中期趋势完好，回踩加仓可靠
                            ma20_rising = (ma20 > ma10) if len(closes_arr) > 20 and len(ma20_v[valid]) >= 2 else True
                            if near_ma and score >= 45 and not (prediction and prediction["direction"] == "看跌"):
                                # MA20向下时，回踩加仓减半（趋势不强）
                                if not ma20_rising:
                                    logger.info("MA20向下，%s 回踩加仓减半", name)
                                # 检查反弹力度：当前价 > 前一根K线收盘价（正在反弹）
                                if len(closes_arr) >= 2 and current_price > closes_arr[-2]:
                                    add_chg = score_info.get("change_pct", 0)
                                    logger.info("回踩均线加仓: %s MA10=%.2f MA20=%.2f 现价=%.2f 偏离MA10=%.1f%% 评分=%s",
                                                name, ma10, ma20, current_price, dev_ma10, score)
                                    adj_ratio = round(0.10 * sentiment_adj * intraday_adj * sector_adj * morning_adj * vol_adj, 2)
                                    if not ma20_rising:
                                        adj_ratio *= 0.5  # MA20向下，趋势不强，减半
                                    trade = self._buy_position(code, name, current_price, max(adj_ratio, 0.03),
                                        f"回踩MA10加仓·评分{score}", add_count=pos.add_count + 1)

                # ── 放量突破加仓：涨幅>3%+放量>1.5倍+评分↑ ──
                if not trade and pos and pos.profit_pct > 0 and pos.add_count < 3 and consecutive_days_down < 3:
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
                                    adj_ratio = round(0.10 * sentiment_adj * intraday_adj * sector_adj * morning_adj * vol_adj, 2)
                                    trade = self._buy_position(code, name, current_price, max(adj_ratio, 0.03),
                                        f"放量突破加仓·评分{score}", add_count=pos.add_count + 1)

                # ── RSI超卖加仓：RSI<30+均线多头未破，回调机会（不要求盈利） ──
                if not trade and pos and pos.add_count < 2:
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
                # ── 亏损中摊平/暴跌加仓 ──
                if not trade and pos and pos.profit_pct < -5 and score >= buy_th + 5:
                    loss = abs(pos.profit_pct)
                    if pos.add_count < 3:  # 最多摊平3次
                        ratio = min(0.05, 0.02 * (loss / 5))  # 亏越多补越多，但最多5%
                        # 暴跌分析结果影响
                        if heavy_drop:
                            if drop_analysis == "恐慌错杀":
                                ratio = min(ratio * 2, 0.08)  # 恐慌错杀可加倍补
                                logger.info("暴跌分析[%s]，%s 加倍摊平至%.0f%%", drop_analysis, name, ratio*100)
                            elif drop_analysis == "连日暴跌加速赶底":
                                ratio = min(ratio * 1.5, 0.06)  # 加速赶底适当加
                                logger.info("暴跌分析[%s]，%s 适当加仓至%.0f%%", drop_analysis, name, ratio*100)
                            elif drop_analysis == "持续下跌中":
                                ratio *= 0.5  # 持续下跌只补一半
                                logger.info("暴跌分析[%s]，%s 减半摊平至%.0f%%", drop_analysis, name, ratio*100)
                        # ── 开盘暴跌策略(跌超7%)：预测看涨则抄底，预测看跌但超卖也抄底 ──
                        if heavy_drop or deep_drop:
                            from src.signals import _calc_rsi
                            rsi_drop = None
                            if kline is not None and len(kline) > 20:
                                c_arr = kline["close"].values.astype(float)
                                rsi_drop = _calc_rsi(c_arr, 14)
                            if prediction and prediction["direction"] == "看涨":
                                ratio = min(ratio * 2, 0.10)  # 预测看涨，加倍抄底
                                logger.info("开盘暴跌+预测看涨(%d%%)，%s 加仓至%.0f%%抄底", prediction["confidence"], name, ratio*100)
                            elif prediction and prediction["direction"] == "看跌" and rsi_drop is not None and rsi_drop < 30:
                                ratio = min(ratio * 1.2, 0.06)  # 看跌但超卖，轻仓试错
                                logger.info("开盘暴跌+RSI%.0f超卖，%s 轻仓试错%.0f%%", rsi_drop, name, ratio*100)
                            elif prediction and prediction["direction"] == "看跌" and deep_drop:
                                ratio *= 0.5
                                logger.info("开盘暴跌+预测看跌，%s 减半等待%.0f%%", name, ratio*100)
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
                                trade = self._buy_position(code, name, current_price, ratio * vol_adj,
                                    f"亏损{loss:.0f}%摊平·{drop_analysis}·评分{score}·预测{prediction['direction'] if prediction else '无'}", add_count=pos.add_count + 1)

                # ── MA趋势加仓：MA5上穿MA20(金叉)确认趋势 + MA20方向向上 ──
                if not trade and pos and pos.add_count < 3 and kline is not None and len(kline) > 25:
                    closes_arr = kline["close"].values.astype(float)
                    from src.signals import _sma
                    ma5_v = _sma(closes_arr, 5)
                    ma20_v = _sma(closes_arr, 20)
                    valid_ma = ~np.isnan(ma5_v) & ~np.isnan(ma20_v)
                    if len(ma5_v[valid_ma]) >= 3:
                        m5_prev, m5_cur = ma5_v[valid_ma][-3], ma5_v[valid_ma][-1]
                        m20_prev, m20_cur = ma20_v[valid_ma][-3], ma20_v[valid_ma][-1]
                        ma20_up = m20_cur > m20_prev  # MA20在上升
                        golden_cross = m5_prev <= m20_prev and m5_cur > m20_cur  # MA5上穿MA20
                        price_above_ma5 = current_price > m5_cur  # 价格在MA5上方
                        # 条件：金叉确认 或 (MA20上升+价格在MA5上方)
                        if (golden_cross or (ma20_up and price_above_ma5)) and score >= 50:
                            if not (prediction and prediction["direction"] == "看跌"):
                                add_ma_ratio = round(0.08 * sector_adj * morning_adj * vol_adj, 2)
                                logger.info("MA趋势加仓: %s 金叉=%s MA20向上=%s 评分=%s", name, golden_cross, ma20_up, score)
                                trade = self._buy_position(code, name, current_price, add_ma_ratio,
                                    f"MA趋势加仓·金叉{'是' if golden_cross else '否'}·评分{score}", add_count=pos.add_count + 1)

        # ── 卖出（评分<40 或 自适应止损）— T+1限制 + 大跌保护 ──
        if not trade and code in self.portfolio.positions:
            today_str = date.today().isoformat()
            # T+1: 当天买入不能当天卖出
            if pos.buy_date == today_str:
                pass
            # ── 大跌日/恐慌错杀保护：不止损等反弹 ──
            elif heavy_drop and drop_analysis in ("恐慌错杀", "加速赶底"):
                logger.info("暴跌分析[%s]，%s 跳过卖出等反弹", drop_analysis, name)
            elif score < sell_th:
                # 大盘大跌日(>1.5%)或市场恐慌时，评分卖出降为卖一半
                from src.scoring import get_market_sentiment
                s_lv, _ = get_market_sentiment()
                market_panic = s_lv <= -1
                if market_panic or market_declining:
                    sell_shares = pos.shares // 2
                    if sell_shares >= 100:
                        trade = self._sell_partial(code, current_price, sell_shares, f"评分{score}·恐慌减半")
                        logger.info("恐慌/跌势中评分%s，%s 减半持仓", score, name)
                # 趋势保护：均线多头时评分低也只卖一半
                if not trade:
                    kline_ma5 = score_info.get("ma5", 0)
                    kline_ma20 = score_info.get("ma20", 0)
                    if kline_ma5 > kline_ma20 and score >= 35:
                        sell_shares = pos.shares // 2
                        if sell_shares >= 100:
                            trade = self._sell_partial(code, current_price, sell_shares, f"评分{score}·趋势向上减半")
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
                    if loss <= -stop_loss_pct:
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
        # 符合100股整数倍
        shares = int(shares / 100) * 100
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
        is_etf = code.startswith(("51", "52", "58", "15", "16"))
        if is_etf:
            comm = max(amount * self.etf_commission, self.min_etf_commission)
        else:
            comm = max(amount * self.commission, self.min_commission)
        return comm + amount * self.transfer_fee

    def _buy_position(self, code, name, price, ratio, reason, add_count=0):
        """买入（带总仓位和板块控制）"""
        if len(self.portfolio.positions) >= self.max_positions and code not in self.portfolio.positions:
            return None

        # ── 总仓位控制：持仓市值占比不超过 max_total_ratio ──
        self._update_value()
        current_pos_ratio = (self.portfolio.total_value - self.portfolio.cash) / self.portfolio.total_value
        if current_pos_ratio >= self.max_total_ratio and code not in self.portfolio.positions:
            logger.info("总仓位已达%.0f%%上限，跳过买入 %s", self.max_total_ratio * 100, name)
            return None

        # ── 板块控制：单板块持仓不超过 max_sector_ratio ──
        if code not in self.portfolio.positions:
            from src.sectors import get_sector_tag
            sector = get_sector_tag(code)
            if sector:
                sector_value = 0
                for pcode, pos in self.portfolio.positions.items():
                    if get_sector_tag(pcode) == sector:
                        sector_value += pos.market_value
                sector_ratio = sector_value / self.portfolio.total_value if self.portfolio.total_value > 0 else 0
                # 加上本次买入后的预估占比
                buy_amount = self.portfolio.cash * ratio
                new_sector_ratio = (sector_value + buy_amount) / self.portfolio.total_value
                if new_sector_ratio > self.max_sector_ratio:
                    logger.info("板块[%s]已达%.0f%%上限(%.0f%%)，跳过买入 %s", 
                                sector, self.max_sector_ratio * 100, new_sector_ratio * 100, name)
                    return None

        amount = self.portfolio.cash * ratio
        if amount < 1000:
            return None
        # 涨停板检查：涨停价买不到
        try:
            from src.fetcher import fetch_realtime
            rt_check = fetch_realtime(code)
            if rt_check:
                yc = rt_check.get("yesterday_close", 0)
                if yc > 0:
                    limit_up = yc * 1.10  # 涨停价±10%
                    if code.startswith(("3", "688")):  # 创业板/科创板±20%
                        limit_up = yc * 1.20
                    if price >= limit_up:
                        logger.info("涨停板(%.2f)>=限价%.2f，%s 无法买入", price, limit_up, name)
                        return None
        except: pass
        shares = int(amount / price / 100) * 100
        if shares < 100:
            # 钱不够100股时，看能买多少
            shares = int(self.portfolio.cash * 0.9 / price / 100) * 100
            if shares < 100:
                logger.info("现金不足(%.0f元)，%s 至少需要%.0f元才能买100股", self.portfolio.cash, name, price*100)
                self.buy_recommendations.append(f"{name}({code}) 需{price*100:.0f}元 现金{self.portfolio.cash:.0f}元")
                return None
        total_cost = shares * price + self._calc_commission(shares * price, code)
        if total_cost > self.portfolio.cash:
            # 现金不够，降仓到可承受范围
            shares = int(shares * 0.8 / 100) * 100
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
                hold_since=old.hold_since or old.buy_date,
            )
        else:
            self.portfolio.positions[code] = Position(
                stock_code=code, stock_name=name,
                buy_date=date.today().isoformat(), buy_price=price,
                shares=shares, total_cost=total_cost,
                current_price=price, market_value=shares * price,
                peak_price=price,
                hold_since=date.today().isoformat(),
            )
        action_label = "加仓" if code in self.portfolio.positions else "买入"
        logger.info("模拟%s: %s %s股 %.2f元 ratio=%.0f%% [%s]", action_label, name, shares, price, ratio * 100, code)
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
        # 跌停板检查：跌停价卖不出
        try:
            from src.fetcher import fetch_realtime
            rt_sc = fetch_realtime(code)
            if rt_sc:
                yc = rt_sc.get("yesterday_close", 0)
                if yc > 0:
                    ld = yc * (0.80 if code.startswith(("3", "688")) else 0.90)
                    if price <= ld:
                        logger.info("跌停板(%.2f)<=限价%.2f，%s 无法卖出", price, ld, pos.stock_name)
                        return None
        except: pass
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
