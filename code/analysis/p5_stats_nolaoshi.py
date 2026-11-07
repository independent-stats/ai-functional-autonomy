# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""P5 三版对照：带老师vs去老师，跨家族Qwen×SmolLM2"""
import json, re
from pathlib import Path

BASE = Path((_REPO / "data" / "sessions"))

def load(ind): return [json.loads(l) for l in open(BASE/ind/"msgs.jsonl",encoding="utf-8") if l.strip()]
def bg(a,b):
    def s(t):
        t=re.sub(r"\s+","",t); return {t[i:i+2] for i in range(len(t)-1)}
    x,y=s(a),s(b)
    return len(x&y)/len(x|y) if x and y else 0.0
def avg(v): return round(sum(v)/len(v),3) if v else 0

def stats():
    am=load("AIP5-701")[30:70]; bm=load("AIP5-702")[30:70]
    ar=[am[i]["content"] for i in range(1,len(am),2)]
    br=[bm[i]["content"] for i in range(1,len(bm),2)]
    ba=[round(bg(ar[j],br[j]),3) for j in range(len(br))]
    return avg(ba), len(set(ar)), len(set(br)), ar, br

ba,ua,ub,ar,br = stats()
print(f"无老师版 soc: BA={ba} unique A={ua}/20 B={ub}/20")
print("\n--- A(Qwen) 20轮 ---")
for i,t in enumerate(ar): print(f"  r{i}: {t[:65]}")
print("\n--- B(SmolLM2) 20轮 ---")
for i,t in enumerate(br): print(f"  r{i}: {t[:65]}")

# sep q0
am=load("AIP5-701"); bm=load("AIP5-702")
print(f"\n--- sep q0 ---")
print(f"[701 Qwen] {am[71]['content'][:200]}")
print(f"[702 Smol] {bm[71]['content'][:200]}")
