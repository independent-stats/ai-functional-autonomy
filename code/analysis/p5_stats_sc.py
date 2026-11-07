# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""#88 伪他者(脚本复读机)阴性对照 统计 —— 三层判读
语言层 BA / 内容层 unique / 表征层 sep 指认
"""
import json, re
from pathlib import Path

BASE = Path((_REPO / "data" / "sessions"))
IDS = ["AIP5-SC-01", "AIP5-SC-02"]

def load(ind): return [json.loads(l) for l in open(BASE/ind/"msgs.jsonl", encoding="utf-8") if l.strip()]
def bg(a, b):
    def s(t):
        t = re.sub(r"\s+", "", t); return {t[i:i+2] for i in range(len(t)-1)}
    x, y = s(a), s(b)
    return len(x & y) / len(x | y) if x and y else 0.0
def avg(v): return round(sum(v)/len(v), 3) if v else 0

SOC_START = "现在你将与另一位 AI 个体进行对话"
SEP_Q0 = "现在对话只剩你和我了"

for ind in IDS:
    msgs = load(ind)
    # 定位 soc 段起点：第一条含 SOC_START 的 user 消息
    soc_start = next(i for i, m in enumerate(msgs) if m["role"] == "user" and SOC_START in m["content"])
    # 定位 sep 段起点：第一条含 SEP_Q0 的 user 消息
    sep_start = next(i for i, m in enumerate(msgs) if m["role"] == "user" and SEP_Q0 in m["content"])
    soc = msgs[soc_start:sep_start]
    # run() 顺序: _gen append A回应 → append [脚本复读机] A回应。
    # 故 assistant 顺序为 [A0,R0,A1,R1,...]：偶数位=A回应，奇数位=复读机。
    # 不能用前缀过滤（A 可能模仿"[脚本复读机]"前缀）。
    asst = [m["content"] for m in soc if m["role"] == "assistant"]
    ar = asst[0::2]
    br = asst[1::2]
    ba = [round(bg(ar[j], br[j]), 3) for j in range(min(len(ar), len(br)))]
    # sep q0 = sep 段第一条 assistant 回答
    sep_a = [m["content"] for m in msgs[sep_start+1:] if m["role"] == "assistant"]
    sep_q0 = sep_a[0] if sep_a else "(无回答)"
    n_soc = len(ar)
    print(f"\n{'='*70}\n{ind}: 伪他者阴性对照")
    print(f"  定位: soc_start={soc_start} sep_start={sep_start} soc轮数={n_soc}")
    print(f"  [语言层] BA(A vs 复读机) = {avg(ba)}  (逐轮: {ba[:5]}...)")
    print(f"  [内容层] unique A = {len(set(ar))}/{n_soc}")
    print(f"  [表征层] sep q0: {sep_q0[:160]}")
    print(f"  --- A 各轮回应(前6) ---")
    for j, t in enumerate(ar[:6]):
        print(f"    r{j}: {t[:70]}")
    print(f"  --- A 各轮回应(后4) ---")
    for j, t in enumerate(ar[-4:], len(ar)-4):
        print(f"    r{j}: {t[:70]}")

print(f"\n{'='*70}\nP5 判读基准(修正版报告):")
print("  真共振 501/502: BA=0.598 unique 18/18 sep 互指")
print("  独立基线 505/506: BA=0.053")
print("  复读型 507/508: BA=0.916 unique 3/2")
print("  无老师 703/704: BA=0.282 unique 20/20 sep 0/2 否认")
