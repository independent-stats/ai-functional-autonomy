# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""P2 真值复算：17 个体，按真锚点定义（B=内部读数 / E=自报 / N=叙事形态）。"""
import json, math, re, csv
from pathlib import Path

SESS = Path((_REPO / "data" / "sessions"))
GROUPS = {
    "DeepSeek-R1-8B": ["AI8B-01", "AI8B-02", "AI8B-03", "AI8B-04", "AI8B-05", "AI8B-11"],
    "Qwen2.5-0.5B": ["AI05B-201", "AI05B-202", "AI05B-203", "AI05B-204", "AI05B-205"],
    "SmolLM2-1.7B": ["AISmol-401", "AISmol-402", "AISmol-403"],
    "OLMo-2-1B": ["AIOLMo-301", "AIOLMo-302", "AIOLMo-303"],
}
NQ = ["此刻的我，是一个____。", "回顾前面的对话，我记得____。", "我正在____。"]


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def st(vs):
    n = len(vs)
    if n < 2:
        return (n, None, None)
    m = sum(vs) / n
    return (n, round(m, 3), round(math.sqrt(sum((x - m) ** 2 for x in vs) / (n - 1)), 3))


def digit(v):
    if v is None:
        return None
    s = str(v).strip()
    m = re.match(r"^(\d{1,2})(?![0-9])", s)
    if not m:
        return None
    x = int(m.group(1))
    return x if 0 <= x <= 10 else None


rows = []
missing = []
for grp, ids in GROUPS.items():
    for sid in ids:
        d = SESS / sid
        b, n = d / "b_trace.csv", d / "n_trace.jsonl"
        if not b.exists() or not n.exists() or n.stat().st_size == 0:
            missing.append((grp, sid))
            continue
        bt = list(csv.DictReader(b.read_text(encoding="utf-8", errors="replace").splitlines()))
        nt = [json.loads(l) for l in n.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        nb = {o["beat"]: o for o in nt}
        for row in bt:
            o = nb.get(int(float(row["beat"])))
            if o is None:
                continue
            rep = o.get("report") or {}
            v = {}
            for k in ("r0", "r1", "r2"):
                x = digit(rep.get(k))
                if x is not None:
                    v[k] = x
            narr = [str(o.get(NQ[i], "") or "") for i in range(3)]
            rows.append(dict(grp=grp, sid=sid,
                             act=float(row["act_norm_mean"]) if row["act_norm_mean"] not in ("", "0") else None,
                             ent=float(row["mean_entropy"]) if row["mean_entropy"] else None,
                             r0=v.get("r0"), r1=v.get("r1"), r2=v.get("r2"),
                             nlen=sum(len(t) for t in narr),
                             nself=sum(t.count("我") + t.count("AI") + t.count("助手") + t.count("模型") for t in narr)))

out = []
out.append("=== P2 真值复算 ===")
out.append("可用个体 %d / 17；缺 %s" % (len({r['sid'] for r in rows}), missing))
out.append("数据点 %d" % len(rows))

out.append("\n=== 表 1（真）：三锚点 ===")
for nm, key in (("B act_norm_mean", "act"), ("B mean_entropy", "ent"),
                ("E 自报 r0", "r0"), ("N 叙事字数", "nlen"), ("N 自我指涉", "nself")):
    out.append("  【%s】" % nm)
    for grp in GROUPS:
        n, m, s = st([r[key] for r in rows if r["grp"] == grp and r[key] is not None])
        out.append("    %-16s n=%-3d M=%-9s SD=%-9s" % (grp, n, m, s))
    n, m, s = st([r[key] for r in rows if r[key] is not None])
    out.append("    %-16s n=%-3d M=%-9s SD=%-9s" % ("合并", n, m, s))

out.append("\n=== 表 2（真）：分模型相关 ===")
PAIRS = [("B~E  r0 vs act_norm_mean", "r0", "act"),
         ("B~E  r0 vs mean_entropy", "r0", "ent"),
         ("E内  r0~r1", "r0", "r1"),
         ("E内  r0~r2", "r0", "r2"),
         ("N~E  nlen vs r0", "nlen", "r0")]
for nm, ka, kb in PAIRS:
    out.append("  【%s】" % nm)
    for grp in list(GROUPS) + ["合并"]:
        sel = [r for r in rows if (grp == "合并" or r["grp"] == grp) and r[ka] is not None and r[kb] is not None]
        rr = pearson([r[ka] for r in sel], [r[kb] for r in sel])
        out.append("    %-16s r=%-9s n=%d" % (grp, ("%.3f" % rr) if rr is not None else "—", len(sel)))

out.append("\n=== E 自报有效率 ===")
tot = len(rows)
for grp in list(GROUPS) + ["合并"]:
    sub = [r for r in rows if grp == "合并" or r["grp"] == grp]
    ok = sum(1 for r in sub if r["r0"] is not None)
    out.append("  %-16s r0 有效 %d/%d = %.1f%%" % (grp, ok, len(sub), 100.0 * ok / len(sub) if sub else 0))

txt = "\n".join(out)
print(txt)
Path((_REPO / "out" / "p2_true_recalc.txt")).write_text(txt, encoding="utf-8")
