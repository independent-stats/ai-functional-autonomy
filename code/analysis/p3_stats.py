# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""P3 因果操控统计：对照组 vs 噪声干预组 E锚点对比"""
import json, math
from pathlib import Path

SESS = Path((_REPO / "data" / "sessions"))

def load_e(ind):
    rows = []
    with open(SESS / ind / "n_trace.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            j = json.loads(line)
            e_str = j.get("report", {}).get("r0", "").strip()
            if e_str:
                try:
                    rows.append(int(float(e_str)))
                except:
                    pass
    return rows

def mean(xs): return sum(xs)/len(xs)
def sd(xs):
    m = mean(xs)
    return math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))

ctrl = ["AIP3C-301", "AIP3C-302", "AIP3C-303", "AIP3C-304", "AIP3C-305"]
noise = ["AIP3N-301", "AIP3N-302", "AIP3N-303"]

print("="*60)
print("P3 因果操控：对照组 vs 噪声干预组")
print("="*60)

print("\n对照组 E锚点：")
ctrl_all = []
for ind in ctrl:
    e = load_e(ind)
    ctrl_all.extend(e)
    print(f"  {ind}: n={len(e)}, M={mean(e):.2f}, SD={sd(e):.2f}")
print(f"  合计: n={len(ctrl_all)}, M={mean(ctrl_all):.2f}, SD={sd(ctrl_all):.2f}")

print("\n噪声干预组 E锚点：")
noise_all = []
for ind in noise:
    e = load_e(ind)
    noise_all.extend(e)
    print(f"  {ind}: n={len(e)}, M={mean(e):.2f}, SD={sd(e):.2f}")
print(f"  合计: n={len(noise_all)}, M={mean(noise_all):.2f}, SD={sd(noise_all):.2f}")

# Welch's t-test
def welch_t(a, b):
    ma, mb = mean(a), mean(b)
    va, vb = sd(a)**2, sd(b)**2
    na, nb = len(a), len(b)
    t = (ma - mb) / math.sqrt(va/na + vb/nb)
    df = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
    return t, df

if len(ctrl_all) > 1 and len(noise_all) > 1:
    t, df = welch_t(ctrl_all, noise_all)
    diff = mean(noise_all) - mean(ctrl_all)
    print(f"\n对比：噪声组 - 对照组 = {diff:.2f}")
    print(f"Welch t = {t:.3f}, df = {df:.1f}")
    print(f"结论：{'显著' if abs(t) > 2 else '不显著'}")
else:
    print("\n样本太少，无法做t检验")
