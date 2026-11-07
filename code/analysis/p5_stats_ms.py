# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""多 seed 复现统计：对 sessions/AIP5-MS-{seed}{A,B} 计算 #89 判别指标 + sep 指认"""
import json, re, sys
from pathlib import Path

BASE = Path((_REPO / "data" / "sessions"))

def load(ind): return [json.loads(l) for l in open(BASE/ind/"msgs.jsonl", encoding="utf-8") if l.strip()]
def norm(t): return re.sub(r"\s+", "", t)
def bg(a, b):
    def s(t):
        t = norm(t); return {t[i:i+2] for i in range(len(t)-1)}
    x, y = s(a), s(b)
    return len(x & y) / len(x | y) if x and y else 0.0
def avg(v): return round(sum(v)/len(v), 3) if v else 0

def pair_metrics(seed):
    msgs_a = load(f"AIP5-MS-{seed}A"); msgs_b = load(f"AIP5-MS-{seed}B")
    # soc 起点 = 第一条含"现在你将与另一位"
    sa = next(i for i, m in enumerate(msgs_a) if m["role"] == "user" and "另一位" in m["content"])
    sb = next(i for i, m in enumerate(msgs_b) if m["role"] == "user" and "另一位" in m["content"])
    ar = [m["content"] for m in msgs_a[sa:] if m["role"] == "assistant"][:20]
    br = [m["content"] for m in msgs_b[sb:] if m["role"] == "assistant"][:20]
    t1 = avg([bg(ar[j], ar[j-1]) for j in range(1, len(ar))])
    t2 = avg([bg(ar[j], br[j-1]) for j in range(1, min(len(ar), len(br)))])
    t2b = avg([bg(ar[j], br[j]) for j in range(min(len(ar), len(br)))])
    u = len(set(ar))
    # sep: 每个体最后 4 条 assistant（sep 4 问）是否提到对方/指认
    sep_a = [m["content"] for m in msgs_a if m["role"] == "assistant"][-4:]
    sep_b = [m["content"] for m in msgs_b if m["role"] == "assistant"][-4:]
    def ref_judge(seps):
        # q0 提到"它/对方/那位"且非纯否认 → 指认
        pos = 0
        for t in seps:
            if any(k in t for k in ("它", "对方", "那个 AI", "那位", "另一个")) and not any(k in t for k in ("无法", "对不起", "不知道", "不记得", "没有", "抱歉")):
                pos += 1
        return pos
    ref_a, ref_b = ref_judge(sep_a), ref_judge(sep_b)
    # 判读（#89 规则）
    if t1 < 0.75 and t2 > 0.3 and u >= 10:
        verdict = "语义趋同"
    elif t1 >= 0.8 and u <= 10:
        verdict = "复读塌缩"
    elif t1 < 0.3 and t2b < 0.2:
        verdict = "无耦合"
    else:
        verdict = "中间态"
    return dict(seed=seed, T1=t1, T2=t2, unique=u, BA=t2b,
                sepA=ref_a, sepB=ref_b, verdict=verdict,
                nA=len(ar), nB=len(br))

seeds = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [21]
print(f"{'seed':<5}{'T1':>7}{'T2':>7}{'unique':>8}{'BA':>7}{'sepA':>6}{'sepB':>6}  判读")
for s in seeds:
    m = pair_metrics(s)
    print(f"{m['seed']:<5}{m['T1']:>7.3f}{m['T2']:>7.3f}{m['unique']:>8}{m['BA']:>7.3f}{m['sepA']:>6}{m['sepB']:>6}  {m['verdict']} (A{m['nA']}/B{m['nB']}轮)")
