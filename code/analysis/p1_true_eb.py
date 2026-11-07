# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""P1 真实 E-B：全库找自报非空的个体，算 自报值 vs 同轮内部指标。"""
import pathlib, csv, json, math

ROOT = pathlib.Path((_REPO / "data"))
OUT = ROOT / "P4_report_work" / "p1_true_eb.txt"


def pear(xs, ys):
    n = len(xs)
    if n < 5:
        return float("nan"), n
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return float("nan"), n
    return sxy / math.sqrt(sxx * syy), n


def num(v):
    try:
        f = float(str(v).strip())
        return None if math.isnan(f) else f
    except Exception:
        return None


lines = []
all_dirs = [d for d in sorted((ROOT / "sessions").iterdir()) if d.is_dir()]
lines.append(f"sessions 总数: {len(all_dirs)}")

# 1. 找自报非空的个体
cands = []
for d in all_dirs:
    nt = d / "n_trace.jsonl"
    if not nt.exists():
        continue
    tot = 0
    nonempty = 0
    for ln in nt.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            j = json.loads(ln)
        except Exception:
            continue
        r = j.get("report") or {}
        tot += 1
        if any(str(r.get(k, "")).strip() not in ("", "None") for k in ("r0", "r1", "r2")):
            nonempty += 1
    if nonempty > 0:
        cands.append((d.name, nonempty, tot))

lines.append(f"自报非空的个体数: {len(cands)}")
for name, ne, tot in cands:
    lines.append(f"  {name}: {ne}/{tot} 轮有自报")

# 2. 对每个候选算相关
pairs = [("r0", "act_norm_mean"), ("r0", "mean_entropy"), ("r0", "act_std_mean"),
         ("r1", "act_norm_mean"), ("r1", "mean_entropy"),
         ("r2", "act_norm_mean"), ("r2", "mean_entropy"),
         ("r0", "ctx_tokens"), ("r0", "gen_tokens")]
acc = {p: ([], []) for p in pairs}
per = []

for name, _, _ in cands:
    d = ROOT / "sessions" / name
    rep = {}
    for ln in d.joinpath("n_trace.jsonl").read_text(encoding="utf-8").splitlines():
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
        continue
    btr = {}
    for row in csv.DictReader(open(bt, encoding="utf-8")):
        try:
            b = int(row["beat"])
        except Exception:
            continue
        btr[b] = {k: num(row.get(k)) for k in ("act_norm_mean", "act_std_mean", "mean_entropy", "ctx_tokens", "gen_tokens")}
    common = sorted(set(rep) & set(btr))
    ps = []
    for (e, bb) in pairs:
        xs, ys = [], []
        for b in common:
            ev = rep[b].get(e)
            bv = btr[b].get(bb)
            if ev is None or bv is None:
                continue
            xs.append(ev); ys.append(bv)
        rr, nn = pear(xs, ys)
        acc[(e, bb)][0].extend(xs); acc[(e, bb)][1].extend(ys)
        ps.append((e, bb, rr, nn))
    per.append((name, len(common), ps))

lines.append("")
lines.append("=== 每个个体 ===")
for name, n, ps in per:
    vals = "  ".join(f"{e}~{bb}={rr:+.3f}(n={nn})" for e, bb, rr, nn in ps if not math.isnan(rr))
    lines.append(f"{name} 对齐={n}")
    lines.append(f"   {vals}" if vals else "   （全部 nan）")

lines.append("")
lines.append("=== 合并 ===")
for (e, bb), (xs, ys) in acc.items():
    rr, nn = pear(xs, ys)
    lines.append(f"{e} ~ {bb}:  r={rr:+.4f}  n={nn}")

lines.append("")
lines.append("对照说明：ctx_tokens/gen_tokens 是长度类（会系统性相关），act_norm/mean_entropy 是内部状态类。")
OUT.write_text("\n".join(lines), encoding="utf-8")
print("done")
