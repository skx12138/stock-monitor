"""
A股行情监控 - 快捷启动
"""
import os
import sys

# 确保在项目根目录运行
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from main import main

if __name__ == "__main__":
    main()
