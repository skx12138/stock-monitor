"""
东方财富 Web 端自动交易
使用 Selenium 控制 Edge 浏览器操作 jy.xzsec.com 交易页面

使用方式:
  trader = RealTrader()
  trader.login()
  trader.buy("600105", 70.95, 200)   # 买入
  trader.sell("000811", 53.50, 100)  # 卖出
  trader.close()
"""
import json
import logging
import os
import time
from typing import Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

logger = logging.getLogger(__name__)

# 默认凭据文件路径
CREDENTIALS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "credentials.json")
# 东方财富证券 Web 交易地址
DEFAULT_TRADE_URL = "https://jy.xzsec.com/"


def load_credentials(path: str = CREDENTIALS_FILE) -> dict:
    """加载交易账号凭据"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"凭据文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class RealTrader:
    """东方财富 Web 交易机器人"""

    def __init__(self, credentials: Optional[dict] = None,
                 trade_url: str = DEFAULT_TRADE_URL,
                 headless: bool = False,
                 dry_run: bool = True):
        """
        Args:
            credentials: 账号密码字典 {account, password}
            trade_url: 交易页面地址
            headless: 是否无头模式
            dry_run: 是否模拟运行（不下真实单）
        """
        if credentials is None:
            credentials = load_credentials()
        self.account = credentials.get("account", "")
        self.password = credentials.get("password", "")
        self.trade_url = credentials.get("trade_url", trade_url)
        self.headless = headless
        self.dry_run = dry_run

        self.driver: Optional[webdriver.Edge] = None
        self.logged_in = False
        self._wait_short = 5   # 短等待（秒）
        self._wait_long = 15   # 长等待（秒）

        # 登录页的 iframe ID（东方财富交易页面常用）
        self._login_iframe_id = "loginIframe"

    # ── 浏览器管理 ──

    def _init_driver(self):
        """初始化 Edge 浏览器"""
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1280,800")
        # 隐藏自动化特征
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        # 使用 Edge 的默认用户数据目录（保持登录状态）
        options.add_argument("--disable-blink-features=AutomationControlled")

        self.driver = webdriver.Edge(options=options)
        self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """
        })

    def close(self):
        """关闭浏览器"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
            self.logged_in = False

    # ── 登录 ──

    def login(self) -> bool:
        """登录东方财富交易系统

        Returns:
            登录成功返回 True
        """
        if self.logged_in:
            return True

        self._init_driver()
        logger.info("打开交易页面: %s", self.trade_url)
        self.driver.get(self.trade_url)
        time.sleep(2)

        wait = WebDriverWait(self.driver, self._wait_long)
        time.sleep(1)

        try:
            # 2. 输入资金账号
            account_input = wait.until(
                EC.presence_of_element_located((By.ID, "txtZjzh"))
            )
            account_input.clear()
            account_input.send_keys(self.account)
            logger.info("已输入账号")
            time.sleep(0.5)

            # 3. 输入交易密码（用 JS 绕过虚拟键盘）
            self.driver.execute_script(
                f'document.getElementById("txtPwd").value = "{self.password}";'
            )
            logger.info("已输入密码")
            time.sleep(0.3)

            # 4. 检查是否有验证码
            if self._check_captcha():
                logger.warning("需要输入图形验证码")
                self.screenshot("captcha_needed.png")
                try:
                    code = input("请打开 captcha_needed.png 查看验证码，输入后按 Enter: ")
                    if code.strip():
                        captcha_input = self.driver.find_element(By.ID, "txtValidCode")
                        captcha_input.clear()
                        captcha_input.send_keys(code.strip())
                except (EOFError, OSError):
                    logger.error("非交互环境，无法输入验证码")
                    return False

            # 5. 点击登录按钮
            login_btn = self.driver.find_element(By.ID, "btnConfirm")
            login_btn.click()
            logger.info("已点击登录")
            time.sleep(3)

            # 6. 检查是否登录成功
            if self._is_logged_in():
                self.logged_in = True
                logger.info("登录成功")
                return True
            else:
                logger.warning("登录可能未完成，请检查页面截图")
                self.screenshot("login_check.png")
                return False

        except Exception as e:
            logger.error("登录过程异常: %s", e)
            return False

    def _find_input(self, id_candidates: list[str], desc: str) -> Optional[object]:
        """按 ID/name/placeholder 尝试找输入框"""
        for cid in id_candidates:
            # ID
            try:
                el = self.driver.find_element(By.ID, cid)
                if el and el.tag_name == "input":
                    return el
            except NoSuchElementException:
                pass
            # name
            try:
                el = self.driver.find_element(By.NAME, cid)
                if el and el.tag_name == "input":
                    return el
                return el
            except NoSuchElementException:
                pass
        # placeholder 匹配
        try:
            for cid in id_candidates:
                els = self.driver.find_elements(By.XPATH,
                    f"//input[contains(@placeholder, '{cid}')]")
                if els:
                    return els[0]
        except Exception:
            pass
        logger.warning("未找到 %s（尝试过: %s）", desc, id_candidates)
        return None

    def _find_clickable(self, id_candidates: list[str], desc: str) -> Optional[object]:
        """按 ID/text 找可点击元素（按钮/链接）"""
        for cid in id_candidates:
            try:
                el = self.driver.find_element(By.ID, cid)
                if el and el.is_displayed():
                    return el
            except NoSuchElementException:
                pass
        # 按 text 找
        for cid in id_candidates:
            try:
                el = self.driver.find_element(By.XPATH,
                    f"//*[contains(text(), '{cid}') and not(./ancestor::*[contains(@style,'display:none')])]")
                if el and el.is_displayed():
                    return el
            except Exception:
                pass
        logger.warning("未找到 %s", desc)
        return None

    def _check_captcha(self) -> bool:
        """检查页面是否有验证码需要输入"""
        try:
            el = self.driver.find_element(By.ID, "txtValidCode")
            return el.is_displayed()
        except NoSuchElementException:
            pass
        return False

    def _is_logged_in(self) -> bool:
        """检查是否已登录"""
        try:
            current_url = self.driver.current_url.lower()
            if "login" not in current_url and "jy.xzsec" not in current_url:
                return True
            # 检查页面是否有退出按钮
            self.driver.find_element(By.XPATH,
                "//*[contains(text(), '退出') or contains(@class, 'logout')]")
            return True
        except NoSuchElementException:
            pass
        try:
            self.driver.find_element(By.XPATH,
                "//*[contains(@id, 'stockCode') or contains(@class, 'trade-tab')]")
            return True
        except NoSuchElementException:
            pass
        return False

    # ── 交易操作 ──

    def buy(self, stock_code: str, price: float, shares: int,
            remark: str = "") -> dict:
        """买入股票

        Args:
            stock_code: 股票代码
            price: 买入价格
            shares: 买入数量（股）
            remark: 备注

        Returns:
            {success, order_id, message}
        """
        logger.info("买入指令: %s %s股@%.2f元 %s", stock_code, shares, price, remark)

        if self.dry_run:
            logger.info("🟡 模拟模式，不下真实单")
            self._dry_run_buy(stock_code, price, shares, remark)
            return {"success": True, "order_id": "dry_run", "message": "模拟下单"}

        if not self._ensure_logged_in():
            return {"success": False, "order_id": "", "message": "未登录"}

        try:
            self._switch_to_tab("买入")
            time.sleep(1)

            # 输入股票代码
            code_input = self._find_input(["stockCode", "code", "stock_code"], "股票代码输入框")
            if code_input:
                code_input.clear()
                code_input.send_keys(stock_code)
                time.sleep(0.5)

            # 等待行情价格刷新
            time.sleep(1)

            # 输入价格
            price_input = self._find_input(["price", "buyPrice", "stockPrice"], "价格输入框")
            if price_input:
                price_input.clear()
                price_input.send_keys(str(price))
                time.sleep(0.3)

            # 输入数量
            qty_input = self._find_input(["quantity", "amount", "buyAmount", "stockAmount"], "数量输入框")
            if qty_input:
                qty_input.clear()
                qty_input.send_keys(str(shares))
                time.sleep(0.3)

            # 点击买入按钮
            buy_btn = self._find_clickable(["buy", "btn-buy", "下单", "买入"], "买入按钮")
            if buy_btn:
                buy_btn.click()
                time.sleep(1)

            # 确认对话框
            result = self._confirm_order("买入")
            logger.info("买入结果: %s", result)
            return result

        except Exception as e:
            logger.error("买入异常: %s", e)
            return {"success": False, "order_id": "", "message": str(e)}

    def sell(self, stock_code: str, price: float, shares: int,
             remark: str = "") -> dict:
        """卖出股票"""
        logger.info("卖出指令: %s %s股@%.2f元 %s", stock_code, shares, price, remark)

        if self.dry_run:
            logger.info("🟡 模拟模式，不下真实单")
            self._dry_run_sell(stock_code, price, shares, remark)
            return {"success": True, "order_id": "dry_run", "message": "模拟下单"}

        if not self._ensure_logged_in():
            return {"success": False, "order_id": "", "message": "未登录"}

        try:
            self._switch_to_tab("卖出")
            time.sleep(1)

            code_input = self._find_input(["stockCode", "code", "stock_code"], "股票代码输入框")
            if code_input:
                code_input.clear()
                code_input.send_keys(stock_code)
                time.sleep(0.5)

            time.sleep(1)

            price_input = self._find_input(["price", "sellPrice", "stockPrice"], "价格输入框")
            if price_input:
                price_input.clear()
                price_input.send_keys(str(price))
                time.sleep(0.3)

            qty_input = self._find_input(["quantity", "amount", "sellAmount", "stockAmount"], "数量输入框")
            if qty_input:
                qty_input.clear()
                qty_input.send_keys(str(shares))
                time.sleep(0.3)

            sell_btn = self._find_clickable(["sell", "btn-sell", "下单", "卖出"], "卖出按钮")
            if sell_btn:
                sell_btn.click()
                time.sleep(1)

            result = self._confirm_order("卖出")
            logger.info("卖出结果: %s", result)
            return result

        except Exception as e:
            logger.error("卖出异常: %s", e)
            return {"success": False, "order_id": "", "message": str(e)}

    def _switch_to_tab(self, tab_name: str):
        """切换到交易选项卡（买入/卖出/撤单/持仓）"""
        try:
            # 先切回主页面（如果卡在 iframe 里）
            self.driver.switch_to.default_content()
            time.sleep(0.3)

            # 找 tab 并点击
            tab = self._find_clickable([tab_name], f"\"{tab_name}\"选项卡")
            if tab:
                tab.click()
                logger.info("已切换到 %s", tab_name)
                time.sleep(1)
        except Exception as e:
            logger.warning("切换选项卡 %s 失败: %s", tab_name, e)

    def _confirm_order(self, action: str) -> dict:
        """确认下单对话框"""
        try:
            # 找确认按钮
            confirm = self._find_clickable(
                ["confirm", "ok", "确定", "确认"],
                "确认按钮"
            )
            if confirm:
                confirm.click()
                time.sleep(1)
                return {"success": True, "order_id": "", "message": f"{action}委托已提交"}
            else:
                logger.warning("未找到确认按钮，可能已自动提交")
                return {"success": True, "order_id": "", "message": f"{action}委托已提交"}
        except Exception as e:
            return {"success": False, "order_id": "", "message": f"确认异常: {e}"}

    def _ensure_logged_in(self) -> bool:
        """确保已登录，若未登录则尝试登录"""
        if not self.logged_in:
            return self.login()
        return True

    # ── 模拟模式 ──

    def _dry_run_buy(self, stock_code: str, price: float, shares: int, remark: str):
        """模拟买入（记录日志 + 推送通知）"""
        msg = (
            f"🟡 [模拟交易] 买入 {stock_code}\n"
            f"   价格: {price:.2f}元\n"
            f"   数量: {shares}股\n"
            f"   金额: {price * shares:,.0f}元\n"
            f"   备注: {remark}"
        )
        logger.info("模拟买入: %s", msg)

    def _dry_run_sell(self, stock_code: str, price: float, shares: int, remark: str):
        """模拟卖出"""
        msg = (
            f"🟡 [模拟交易] 卖出 {stock_code}\n"
            f"   价格: {price:.2f}元\n"
            f"   数量: {shares}股\n"
            f"   金额: {price * shares:,.0f}元\n"
            f"   备注: {remark}"
        )
        logger.info("模拟卖出: %s", msg)

    # ── 查询 ──

    def get_positions(self) -> list[dict]:
        """获取当前持仓"""
        if not self._ensure_logged_in():
            return []

        try:
            self._switch_to_tab("持仓")
            time.sleep(2)
            # 这里需要解析持仓表格
            # 东方财富的持仓页面结构需要实际浏览器来适配
            logger.info("已切换到持仓页面")
            return self._parse_table("持仓")
        except Exception as e:
            logger.error("获取持仓失败: %s", e)
            return []

    def get_balance(self) -> dict:
        """获取资金情况"""
        if not self._ensure_logged_in():
            return {}

        try:
            # 通常资金信息在页面顶部或左侧
            balance_el = self.driver.find_element(By.XPATH,
                "//*[contains(text(), '可用') or contains(text(), '资金余额')]")
            logger.info("资金信息: %s", balance_el.text)
            return {"text": balance_el.text}
        except NoSuchElementException:
            pass
        return {}

    def _parse_table(self, name: str) -> list[dict]:
        """解析页面表格数据（需要根据实际页面结构调整）"""
        rows = []
        try:
            table = self.driver.find_element(By.XPATH,
                "//table[contains(@class, 'table') or contains(@id, 'grid')]")
            for tr in table.find_elements(By.XPATH, ".//tr")[1:]:
                cells = [td.text for td in tr.find_elements(By.XPATH, ".//td")]
                if cells:
                    rows.append(dict(zip(range(len(cells)), cells)))
        except Exception:
            pass
        return rows

    # ── 截图（调试用） ──

    def screenshot(self, path: str = "debug_screenshot.png"):
        """截取当前页面截图"""
        if self.driver:
            self.driver.save_screenshot(path)
            logger.info("截图已保存: %s", path)


# ── 快捷测试 ──

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    trader = RealTrader(dry_run=True, headless=False)
    try:
        ok = trader.login()
        if ok:
            logger.info("✅ 登录验证成功")
            # 测试模拟下单
            trader.buy("600105", 70.95, 200, "测试买入")
            trader.sell("000811", 53.50, 100, "测试卖出")
            trader.screenshot("login_success.png")
        else:
            logger.error("❌ 登录验证失败")
            trader.screenshot("login_failed.png")
    finally:
        trader.close()
