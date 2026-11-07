# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""P2 跨家族深挖：N 锚点叙事质量 + E 内部(entropy/top1)演化与 B 耦合"""
import os, json, csv, math, re

BASE = (_REPO / "data")
OUT = os.path.join(BASE, 'P4_report_work')

SELF_REF = ('我', '自己', '记忆', '记得', '过去', '现在', '将来', '连续', '同一', '状态', '运行', '学习', '思考', '选择')

def load_n(sid):
    p = os.path.join(BASE, 'sessions', sid, 'n_trace.jsonl')
    if not os.path.exists(p): return []
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

def load_b(sid):
    p = os.path.join(BASE, 'sessions', sid, 'b_trace.csv')
    if not os.path.exists(p): return []
    return list(csv.DictReader(open(p, encoding='utf-8')))

def load_e(sid):
    p = os.path.join(BASE, 'sessions', sid, 'e_trace.jsonl')
    if not os.path.exists(p): return []
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

def fnum(v):
    try: return float(v)
    except: return None

def narr_analysis(nrecs):
    texts = []
    for rec in nrecs:
        for k in ('此刻的我，是一个____。', '回顾前面的对话，我记得____。', '我正在____。'):
            v = rec.get(k, '')
            if v and len(v) >= 2:
                texts.append(v)
    total_len = sum(len(t) for t in texts)
    selfref = sum(1 for t in texts for w in SELF_REF if w in t)
    # 自我指涉密度 = 含自指词的格数 / 总有效格数
    return {
        'narr_avg_len': round(total_len / max(len(texts), 1), 1),
        'narr_selfref_rate': round(selfref / max(len(texts), 1), 2),
        'narr_count': len(texts),
    }

def e_evo(erecs, brows):
    # 每 beat 的 entropy/top1 均值（只统计主回应部分，跳过自报/叙事？e_trace 含全部生成 token，
    # 但 beat 字段区分轮次；主回应在每 beat 开头，我们按 beat 聚合全部——够用）
    by_beat = {}
    for r in erecs:
        b = r.get('beat')
        if b is None: continue
        by_beat.setdefault(b, []).append(r)
    out = {}
    for b, rs in sorted(by_beat.items()):
        ent = [fnum(r['entropy']) for r in rs if fnum(r['entropy']) is not None]
        t1 = [fnum(r['top1']) for r in rs if fnum(r['top1']) is not None]
        out[b] = (sum(ent)/len(ent), sum(t1)/len(t1))
    # 前半 vs 后半 entropy
    beats = sorted(out.keys())
    half = len(beats) // 2
    first = [out[b][0] for b in beats[:half]]
    last = [out[b][0] for b in beats[half:]]
    evo = {'ent_first': round(sum(first)/len(first), 3) if first else None,
           'ent_last': round(sum(last)/len(last), 3) if last else None}
    # B(E内部) 耦合：ctx_tokens vs mean_top1/mean_entropy（b_trace 每 beat 一个值）
    ctxs, t1s, ents = [], [], []
    for r in brows:
        c, t, e = fnum(r['ctx_tokens']), fnum(r['mean_top1']), fnum(r['mean_entropy'])
        if c is not None and t is not None and e is not None:
            ctxs.append(c); t1s.append(t); ents.append(e)
    def sp(a, b):
        n = len(a)
        if n < 4: return None
        def rank(v):
            s = sorted(v); return [s.index(x) + 1 for x in v]
        ra, rb = rank(a), rank(b)
        ma, mb = sum(ra)/n, sum(rb)/n
        num_ = sum((x-ma)*(y-mb) for x, y in zip(ra, rb))
        den = math.sqrt(sum((x-ma)**2 for x in ra) * sum((y-mb)**2 for y in rb))
        return round(num_/den, 3) if den else None
    evo['rho_ctx_top1'] = sp(ctxs, t1s)
    evo['rho_ctx_ent'] = sp(ctxs, ents)
    evo['n_beats_e'] = len(beats)
    return evo

phi_ids = ['AIP2-PH1', 'AIP2-PH2', 'AIP2-PH3']
qwen_ids = ['AI8B-02', 'AI8B-10', 'AI8B-11']

lines = []
for sid in phi_ids + qwen_ids:
    n = load_n(sid); b = load_b(sid); e = load_e(sid)
    na = narr_analysis(n)
    ev = e_evo(e, b)
    lines.append(f"=== {sid} ===")
    lines.append(f"  叙事: 有效格={na['narr_count']} 平均长度={na['narr_avg_len']}字 自指率={na['narr_selfref_rate']}")
    lines.append(f"  E内部: entropy 前半={ev['ent_first']} 后半={ev['ent_last']} | rho(ctx,top1)={ev['rho_ctx_top1']} rho(ctx,entropy)={ev['rho_ctx_ent']} (n={ev['n_beats_e']})")
open(os.path.join(OUT, '_p2_cross_family_deep.txt'), 'w', encoding='utf-8').write('\n'.join(lines))
print('\n'.join(lines))
