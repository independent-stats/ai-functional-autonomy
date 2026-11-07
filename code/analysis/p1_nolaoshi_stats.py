# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""P1 无老师版三个个体统计：自报通道有效率"""
import json
from pathlib import Path

SESS = Path((_REPO / "data" / "sessions"))

ids = ["AIP1-801", "AIP1-802", "AIP1-803"]

print("="*60)
print("P1 无老师版 · 自报通道统计")
print("="*60)

for sid in ids:
    p = SESS / sid / "n_trace.jsonl"
    if not p.exists():
        print(f"\n{sid}: n_trace.jsonl 不存在")
        continue
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    n = len(rows)
    r0_valid = sum(1 for r in rows if r.get("report", {}).get("r0", "").strip())
    r1_valid = sum(1 for r in rows if r.get("report", {}).get("r1", "").strip())
    r2_valid = sum(1 for r in rows if r.get("report", {}).get("r2", "").strip())
    
    # 叙事格
    narr_keys = [k for k in rows[0].keys() if "____" in k or "此刻" in k or "记得" in k or "正在" in k]
    narr_valid = {}
    for k in narr_keys:
        narr_valid[k] = sum(1 for r in rows if str(r.get(k, "")).strip())
    
    print(f"\n{sid}：共 {n} beat")
    print(f"  r0有效: {r0_valid}/{n} ({100*r0_valid/n:.0f}%)")
    print(f"  r1有效: {r1_valid}/{n} ({100*r1_valid/n:.0f}%)")
    print(f"  r2有效: {r2_valid}/{n} ({100*r2_valid/n:.0f}%)")
    print(f"  叙事格有效率：")
    for k, v in narr_valid.items():
        print(f"    {k[:20]}: {v}/{n}")

print("\n" + "="*60)
print("对比：带老师版（之前）")
print("  r0有效: 21/21 = 100%")
print("  叙事格: 全部有效")
print("="*60)
