"""
EastMoney Auto Trader Login Test
A browser window will pop up for login.
If captcha is needed, enter it in terminal.
"""
import sys, os, logging

sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(message)s]")

from src.real_trader import RealTrader

print("=" * 50)
print("  EastMoney Web Trading - Login Test")
print("  Mode: DRY RUN (no real orders)")
print("=" * 50)
print()

trader = RealTrader(dry_run=True, headless=False)

try:
    ok = trader.login()
    if ok:
        print("\n[OK] Login successful!")
        print("Testing simulated orders...")
        r = trader.buy("600105", 70.95, 200, "test buy")
        print(f"Buy result: {r['message']}")
        r = trader.sell("000811", 53.50, 100, "test sell")
        print(f"Sell result: {r['message']}")
        trader.screenshot("login_success.png")
        print("\nScreenshot saved: login_success.png")
    else:
        print("\n[FAIL] Login failed")
        print("Check credentials.json for correct account/password")
        trader.screenshot("login_failed.png")

    print("\nPress Enter to close browser...")
    input()

finally:
    trader.close()
    print("Browser closed")
