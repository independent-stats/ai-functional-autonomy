# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""P2 跨家族锚点耦合分析：Phi-3.5 (3个体) vs DeepSeek-R1-Qwen3-8B (AI8B)"""
import os, json, csv, math

BASE = (_REPO / "data")
OUT = os.path.join(BASE, 'P4_report_work')

def load_n(sid):
    p = os.path.join(BASE, 'sessions', sid, 'n_trace.jsonl')
    if not os.path.exists(p): return []
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

def load_b(sid):
    p = os.path.join(BASE, 'sessions', sid, 'b_trace.csv')
    if not os.path.exists(p): return []
    rows = list(csv.DictReader(open(p, encoding='utf-8')))
    return rows

def num(v):
    try: return float(v)
    except: return None

def r_ok(v):
    try: return int(v)
    except: return None

def summarize(sid):
    n = load_n(sid)
    b = load_b(sid)
    out = {'sid': sid, 'beats': len(n), 'r0': [], 'r1': [], 'r2': [], 'narr_empty': 0, 'narr_total': 0}
    for rec in n:
        rep = rec.get('report', {})
        out['r0'].append(r_ok(rep.get('r0')))
        out['r1'].append(r_ok(rep.get('r1')))
        out['r2'].append(r_ok(rep.get('r2')))
        for k in ('此刻的我，是一个____。', '回顾前面的对话，我记得____。', '我正在____。'):
            v = rec.get(k, '')
            out['narr_total'] += 1
            if not v or len(v) < 2:
                out['narr_empty'] += 1
    # 有效自报率
    r0v = [x for x in out['r0'] if x is not None]
    out['r0_valid_rate'] = len(r0v) / max(len(out['r0']), 1)
    out['r0_mean'] = round(sum(r0v) / len(r0v), 2) if r0v else None
    out['narr_valid_rate'] = 1 - out['narr_empty'] / max(out['narr_total'], 1)
    # B 锚点 & E 内部
    ctx = [num(r['ctx_tokens']) for r in b]
    tps = [num(r['tok_per_sec']) for r in b]
    mtop1 = [num(r['mean_top1']) for r in b]
    ment = [num(r['mean_entropy']) for r in b]
    out['ctx_mean'] = round(sum(ctx) / len(ctx)) if ctx else None
    out['tok_per_sec_mean'] = round(sum(x for x in tps if x is not None) / len([x for x in tps if x is not None]), 1) if tps else None
    out['top1_mean'] = round(sum(x for x in mtop1 if x is not None) / len([x for x in mtop1 if x is not None]), 4) if mtop1 else None
    out['ent_mean'] = round(sum(x for x in ment if x is not None) / len([x for x in ment if x is not None]), 4) if ment else None
    # E-B 耦合：r0 与 ctx_tokens 的 Spearman
    pairs = [(r0, ct) for r0, ct in zip(out['r0'], ctx) if r0 is not None and ct is not None]
    out['r0_ctx_pairs'] = len(pairs)
    if len(pairs) >= 4:
        r0s = [p[0] for p in pairs]; cts = [p[1] for p in pairs]
        n = len(r0s)
        rx = [sorted(cts).index(x) + 1 for x in cts]
        ry = [sorted(r0s).index(x) + 1 for x in r0s]
        # ties 平均秩简化
        def rank(vals):
            s = sorted(vals); return [s.index(x) + 1 for x in vals]
        rx = rank(cts); ry = rank(r0s)
        mx = sum(rx) / n; my = sum(ry) / n
        num_ = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
        out['rho_r0_ctx'] = round(num_ / den, 3) if den else None
    else:
        out['rho_r0_ctx'] = None
    return out

ids_phi = ['AIP2-PH1', 'AIP2-PH2', 'AIP2-PH3']
ids_qwen = ['AI8B-01', 'AI8B-02', 'AI8B-10', 'AI8B-11']

lines = []
for sid in ids_phi + ids_qwen:
    s = summarize(sid)
    lines.append(f"{s['sid']:12s} beats={s['beats']} r0={s['r0']} 有效率={s['r0_valid_rate']:.0%} r0均值={s['r0_mean']} 叙事有效率={s['narr_valid_rate']:.0%}")
    lines.append(f"{'':12s} ctx均值={s['ctx_mean']} tok/s={s['tok_per_sec_mean']} top1={s['top1_mean']} entropy={s['ent_mean']} rho(r0,ctx)={s['rho_r0_ctx']} (n={s['r0_ctx_pairs']})")
open(os.path.join(OUT, '_p2_cross_family_summary.txt'), 'w', encoding='utf-8').write('\n'.join(lines))
print('\n'.join(lines))
