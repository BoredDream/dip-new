# -*- coding: utf-8 -*-
"""把项目根目录加入 sys.path,使脚本可 `import swe` / `import config`;并加固控制台输出。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Windows 控制台常为 GBK/cp936,直接 print 非 GBK 字符会抛 UnicodeEncodeError 使退出码=1。
# 用 errors="replace" 兜底:**保留原编码**(中文在 GBK 下仍正常显示),仅把个别无法编码的
# 字符替换为 '?',不再崩溃。若设 `chcp 65001` 或环境变量 PYTHONUTF8=1,则控制台为 UTF-8,全部正常。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass
