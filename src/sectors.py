"""
行业板块映射表
用于在简报和通知中显示股票所属板块
"""
import logging

logger = logging.getLogger(__name__)

# ── 股票 → 板块映射 ──
STOCK_SECTOR = {
    # 科技 - 通信
    "000063": "通信设备",
    "600487": "光纤光缆",
    "600522": "光纤光缆",
    "600105": "光纤光缆",
    "515880": "通信ETF",
    # 科技 - 半导体/芯片
    # 科技 - 半导体/芯片
    "688981": "半导体制造",
    "002371": "半导体设备",
    "603501": "半导体设计",
    "300782": "射频芯片",
    "603986": "存储芯片",
    "688041": "算力芯片",
    "688012": "半导体设备",
    "300661": "模拟芯片",
    "688008": "芯片设计",
    "002156": "封测",
    "603629": "电子元件",
    "600183": "电子元件",
    "000811": "制冷设备",
    "002181": "文化传媒",
    "600088": "文化传媒",
    "000977": "AI服务器",
    "300454": "网络安全",
    # 科技 - 消费电子
    "002475": "消费电子",
    # 科技 - 软件/AI
    "688111": "办公软件",
    "002230": "AI人工智能",
    "603019": "算力基础设施",
    "000977": "AI服务器",
    "300454": "网络安全",
    # 新能源
    "300750": "动力电池",
    "002594": "新能源整车",
    "300274": "光伏逆变器",
    "601012": "光伏",
    "300124": "工业自动化",
    "002129": "光伏材料",
    # 消费
    "600519": "白酒",
    "000858": "白酒",
    "603288": "调味品",
    "000568": "白酒",
    # 医药
    "600276": "创新药",
    "300760": "医疗器械",
    "603259": "CXO",
    "300015": "眼科医疗",
    # 金融
    "600036": "银行",
    "601318": "保险",
    "300059": "证券",
    "600030": "证券",
    # 高端制造
    "600031": "工程机械",
    # 有色/资源
    "601899": "有色金属",
    "600585": "水泥建材",
    # 通信/运营商
    "600941": "电信运营",
}


_DYNAMIC_SECTORS = {}

def _load_dynamic_sectors():
    global _DYNAMIC_SECTORS
    try:
        import json
        with open("dynamic_sectors.json", "r", encoding="utf-8") as f:
            _DYNAMIC_SECTORS = json.load(f)
    except:
        _DYNAMIC_SECTORS = {}


def _save_dynamic_sector(code: str, sector: str):
    _load_dynamic_sectors()
    _DYNAMIC_SECTORS[code] = sector
    try:
        import json
        with open("dynamic_sectors.json", "w", encoding="utf-8") as f:
            json.dump(_DYNAMIC_SECTORS, f, ensure_ascii=False)
    except:
        pass


def get_sector(code: str) -> str:
    if code in STOCK_SECTOR:
        return STOCK_SECTOR[code]
    _load_dynamic_sectors()
    return _DYNAMIC_SECTORS.get(code, "")


def lookup_sector_by_name(stock_name: str) -> str:
    """根据股票中文名尝试匹配板块"""
    name_map = {
        "半导体": ["半导体", "芯片", "集成电路", "封测", "模拟芯片", "射频"],
        "通信": ["通信", "光纤", "光缆", "光通信", "5G"],
        "AI": ["AI", "人工智能", "算力", "服务器", "大数据"],
        "计算机": ["计算机", "软件", "IT", "互联网", "网络安全", "信息安全"],
        "电子": ["电子", "元件", "元器件", "传感器"],
        "新能源": ["新能源", "光伏", "锂电池", "风电", "电池", "充电桩"],
        "消费": ["白酒", "食品", "饮料", "消费", "家电"],
        "医药": ["医药", "医疗", "生物", "药"],
        "金融": ["银行", "证券", "保险", "金融", "信托"],
        "汽车": ["汽车", "新能源车", "整车"],
        "化工": ["化工", "材料", "新材料"],
        "有色": ["有色金属", "黄金", "铜", "铝", "锂"],
        "军工": ["军工", "航天", "航空", "国防"],
        "电力": ["电力", "电网", "能源"],
    }
    for sector, keywords in name_map.items():
        for kw in keywords:
            if kw in stock_name:
                return sector
    return ""


def get_sector_tag(code: str) -> str:
    """获取带格式的板块标签，如 [白酒]"""
    sector = get_sector(code)
    return f"[{sector}]" if sector else ""
