"""
买卖区间价格系统 — 交易决策不再是单一价格，而是一个价格区间
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PriceRange:
    """买卖区间价格"""
    buy_lower: float       # 最低买入价（低于此价不买）
    buy_upper: float       # 最高买入价（高于此价不追）
    buy_core: float        # 理想买入价（区间中心）
    sell_targets: list     # 分批止盈目标价 [(price, ratio), ...]
    stop_loss: float       # 止损价
    expire_at: Optional[datetime] = None
    filled: bool = False


@dataclass
class PendingOrder:
    """挂单 — 等待价格进入区间后执行"""
    order_id: str
    stock_code: str
    stock_name: str
    direction: str          # "buy" / "sell"
    price_range: PriceRange
    reason: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True


def calc_buy_range_on_breakout(
    breakout_price: float,
    atr: float,
    consolidation_high: float,
) -> PriceRange:
    """在突破点计算买入区间"""
    buy_core = breakout_price
    buy_lower = max(breakout_price - atr * 0.5, consolidation_high)
    buy_upper = breakout_price + atr * 0.8
    sell_targets = [
        (breakout_price + atr * 2.0, 0.33),
        (breakout_price + atr * 3.5, 0.33),
        (breakout_price + atr * 5.0, 0.34),
    ]
    stop_loss = breakout_price - atr * 1.5
    return PriceRange(buy_lower, buy_upper, buy_core, sell_targets, stop_loss)


def calc_sell_range(entry_price: float, atr: float, is_breakout: bool = False) -> PriceRange:
    """持仓后计算卖出区间"""
    if is_breakout:
        sell_targets = [
            (entry_price + atr * 2.0, 0.33),
            (entry_price + atr * 3.5, 0.33),
            (entry_price + atr * 5.0, 0.34),
        ]
    else:
        sell_targets = [
            (entry_price * 1.10, 0.33),
            (entry_price * 1.15, 0.33),
            (entry_price * 1.20, 0.34),
        ]
    stop_loss = entry_price * 0.92
    return PriceRange(0, 0, entry_price, sell_targets, stop_loss)


class OrderManager:
    """挂单管理器 — 每轮检查价格是否进入区间"""

    def __init__(self):
        self._orders: dict[str, PendingOrder] = {}
        self._next_id: int = 0

    def place_order(self, code, name, direction, price_range, reason) -> str:
        """创建挂单"""
        oid = f"ORD_{datetime.now().strftime('%H%M%S')}_{self._next_id}"
        self._next_id += 1
        self._orders[oid] = PendingOrder(oid, code, name, direction, price_range, reason)
        logger.info("挂单 %s: %s %s 区间[%.2f, %.2f]", oid, name, direction,
                    price_range.buy_lower, price_range.buy_upper)
        return oid

    def check_orders(self, current_prices: dict[str, float]) -> list[tuple[str, PendingOrder]]:
        """检查所有活跃挂单，价格进入区间则返回该订单"""
        filled = []
        for oid, order in list(self._orders.items()):
            if not order.is_active:
                continue
            price = current_prices.get(order.stock_code)
            if price is None:
                continue
            if order.direction == "buy":
                l, u = order.price_range.buy_lower, order.price_range.buy_upper
                if l <= price <= u:
                    order.price_range.filled = True
                    order.is_active = False
                    filled.append((oid, order))
        return filled

    def cancel_all(self):
        for o in self._orders.values():
            o.is_active = False
