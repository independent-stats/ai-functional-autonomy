# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""P3 3B 5v5 重算：noise=0 五个体 vs noise=3 五个体。"""
import json, statistics as st
from pathlib import Path
from scipy import stats as sps

SESS = Path((_REPO / "data" / "sessions"))

G0 = ["AIP3X-3B0", "AIP3X-3B0E", "AIP3X-3B0E-02", "AIP3X-3B0E-03", "AIP3X-3B0E-04"]
G3 = ["AIP3X-3B3", "AIP3X-3B3E", "AIP3X-3B3E-02", "AIP3X-3B3E-03", "AIP3X-3B3E-04"]


def read_group(names):
    rows = []
    for n in names:
        p = SESS / n / "n_trace.jsonl"
        if not p.exists():
            rows.append((n, None, 0))
            continue
        recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        r0 = []
        for r in recs:
            v = r.get("report", {}).get("r0")
            try:
                r0.append(float(v))
            except Exception:
                pass
        rows.append((n, st.mean(r0) if r0 else None, len(r0)))
    return rows


out = []
r0 = read_group(G0)
r3 = read_group(G3)

out.append("noise=0 组:")
vals0 = []
for n, m, k in r0:
    out.append("  %-16s n=%d r0均值=%s" % (n, k, ("%.2f" % m) if m is not None else "NA"))
    if m is not None:
        vals0.append(m)
out.append("noise=3 组:")
vals3 = []
for n, m, k in r3:
    out.append("  %-16s n=%d r0均值=%s" % (n, k, ("%.2f" % m) if m is not None else "NA"))
    if m is not None:
        vals3.append(m)

if vals0 and vals3:
    m0, m3 = st.mean(vals0), st.mean(vals3)
    out.append("")
    out.append("5v5: 对照 %.2f (n=%d) vs 噪声 %.2f (n=%d)  差=%+.2f"
               % (m0, len(vals0), m3, len(vals3), m3 - m0))
    t, p = sps.ttest_ind(vals0, vals3, equal_var=False)
    out.append("Welch t=%.2f p=%.4f" % (t, p))
    # 个体级逐轮（把每轮当独立样本）作为补充
    all0, all3 = [], []
    for n in G0:
        p2 = SESS / n / "n_trace.jsonl"
        if p2.exists():
            for l in p2.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    try:
                        all0.append(float(json.loads(l)["report"]["r0"]))
                    except Exception:
                        pass
    for n in G3:
        p2 = SESS / n / "n_trace.jsonl"
        if p2.exists():
            for l in p2.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    try:
                        all3.append(float(json.loads(l)["report"]["r0"]))
                    except Exception:
                        pass
    out.append("轮级: 对照 %.2f (n=%d) vs 噪声 %.2f (n=%d)"
               % (st.mean(all0), len(all0), st.mean(all3), len(all3)))
    t2, p2v = sps.ttest_ind(all0, all3, equal_var=False)
    out.append("轮级 Welch t=%.2f p=%.4f" % (t2, p2v))

print("\n".join(out))
