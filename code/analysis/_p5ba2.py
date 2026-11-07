# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""按 p5_stats_final.py 原口径重算：BA = 配对两目录同轮回应相似度。"""
import json, re
from pathlib import Path

S = Path((_REPO / "data" / "sessions"))


def load(n):
    p = S / n / "msgs.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def bg(t):
    t = re.sub(r"\s+", "", str(t))
    return {t[i:i + 2] for i in range(len(t) - 1)}


def jac(a, b):
    x, y = bg(a), bg(b)
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def avg(v):
    return sum(v) / len(v) if v else 0.0


def pair(a_ind, b_ind, tag):
    a = load(a_ind)[30:70]
    b = load(b_ind)[30:70]
    ar = [a[i]["content"] for i in range(1, len(a), 2)]
    br = [b[i]["content"] for i in range(1, len(b), 2)]
    n = min(len(ar), len(br))
    ar, br = ar[:n], br[:n]
    ba = [jac(ar[j], br[j]) for j in range(n)]
    t2 = [jac(ar[j], br[j - 1]) for j in range(1, n)]
    t1 = [jac(ar[j], ar[j - 1]) for j in range(1, n)]
    vals = (tag, len(ar), len(br), avg(ba), avg(t1), avg(t2), len(set(ar)), len(set(br)))
    print("%-22s A=%d B=%d | BA=%.3f | T1=%.3f | T2=%.3f | unique A=%d B=%d" % vals)


print("=== 原报告 8 条件 ===")
pair("AIP5-501", "AIP5-502", "501/502 真共振")
pair("AIP5-507", "AIP5-508", "507/508 复读型")
pair("AIP5-505", "AIP5-506", "505/506 独立基线")
pair("AIP5-601", "AIP5-602", "601/602 3B复读")
pair("AIP5-603", "AIP5-604", "603/604 3B各说各话")
pair("AIP5-605", "AIP5-606", "605/606 3B修复")
pair("AIP5-701", "AIP5-702", "701/702 跨家族")
pair("AIP5-703", "AIP5-704", "703/704 无老师")
pair("AIP5-SC-01", "AIP5-SC-02", "SC-01/02 伪他者")

print("\n=== 多 seed 6 对 ===")
for i in range(21, 27):
    pair("AIP5-MS-%dA" % i, "AIP5-MS-%dB" % i, "seed %d" % i)
