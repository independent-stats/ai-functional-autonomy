# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""P1 个体内 E-B 耦合：每个个体内部算 r0 vs 内部指标，再看分布。
解决 pooled 被跨个体量纲污染的问题。
"""
import pathlib, csv, json, math, statistics

ROOT = pathlib.Path((_REPO / "data"))
OUT = ROOT / "P4_report_work" / "p1_eb_within.txt"


def pear(xs, ys):
    n = len(xs)
    if n < 5:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def num(v):
    try:
        f = float(str(v).strip())
        return None if math.isnan(f) else f
    except Exception:
        return None


def collect(d):
    rep = {}
    nt = d / "n_trace.jsonl"
    if not nt.exists():
        return None
    for ln in nt.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            j = json.loads(ln)
        except Exception:
            continue
        b = j.get("beat")
        r = j.get("report") or {}
        if b is None:
            continue
        rep[b] = {k: num(r.get(k)) for k in ("r0", "r1", "r2")}
    bt = d / "b_trace.csv"
    if not bt.exists():
        return None
    btr = {}
    for row in csv.DictReader(open(bt, encoding="utf-8")):
        try:
            b = int(row["beat"])
        except Exception:
            continue
        btr[b] = {k: num(row.get(k)) for k in ("act_norm_mean", "mean_entropy", "ctx_tokens", "gen_tokens")}
    return rep, btr


bases = ["act_norm_mean", "mean_entropy", "ctx_tokens", "gen_tokens"]
res = {b: [] for b in bases}   # list of (name, r, n)

for d in sorted((ROOT / "sessions").iterdir()):
    if not d.is_dir():
        continue
    c = collect(d)
    if not c:
        continue
    rep, btr = c
    common = sorted(set(rep) & set(btr))
    for base in bases:
        xs, ys = [], []
        for b in common:
            ev = rep[b].get("r0")
            bv = btr[b].get(base)
            if ev is None or bv is None:
                continue
            xs.append(ev); ys.append(bv)
        r = pear(xs, ys)
        if r is not None:
            res[base].append((d.name, r, len(xs)))

lines = ["P1 自报 r0（" + "此刻的我，是一个____" + "）vs 各指标 —— 个体内相关分布", ""]
for base in bases:
    items = res[base]
    if not items:
        lines.append(f"[{base}] 无可用个体")
        continue
    rs = [r for _, r, _ in items]
    pos = sum(1 for r in rs if r > 0)
    neg = sum(1 for r in rs if r < 0)
    mean_r = statistics.mean(rs)
    med = statistics.median(rs)
    strong_pos = sum(1 for r in rs if r >= 0.5)
    strong_neg = sum(1 for r in rs if r <= -0.5)
    lines.append(f"[{base}]  个体数={len(rs)}  n范围={min(n for _,_,n in items)}-{max(n for _,_,n in items)}")
    lines.append(f"   正相关 {pos} 个 / 负相关 {neg} 个")
    lines.append(f"   均值 r={mean_r:+.3f}  中位数 r={med:+.3f}")
    lines.append(f"   强正(>=0.5) {strong_pos} 个  强负(<=-0.5) {strong_neg} 个")
    lines.append("")

lines.append("=== 明细（|r|>=0.5 的个体）===")
for base in bases:
    lines.append(f"-- {base} --")
    for name, r, n in sorted(res[base], key=lambda t: -abs(t[1])):
        if abs(r) >= 0.5:
            lines.append(f"   {name}  r={r:+.3f}  n={n}")
OUT.write_text("\n".join(lines), encoding="utf-8")
print("done")
