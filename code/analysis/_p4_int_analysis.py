# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""P4 中断恢复实验分析：身份保持/事件回忆/E水平"""
import os, json
BASE = (_REPO / "data")

def load_msgs(sid):
    # 阶段A：中断存档（含 user/assistant 完整对话）；阶段B：e_trace 按 beat 拼主回应
    p = os.path.join(BASE, 'sessions', sid, 'interrupt_context_save.jsonl')
    if os.path.exists(p):
        msgs = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
    else:
        msgs = []
    # 阶段B 主回应：e_trace 中 beat>=100 的 token 按 beat 聚合
    ep = os.path.join(BASE, 'sessions', sid, 'e_trace.jsonl')
    beats = {}
    if os.path.exists(ep):
        for l in open(ep, encoding='utf-8'):
            r = json.loads(l)
            if r.get('beat') >= 100 and r.get('phase') == 'B':
                beats.setdefault(r['beat'], []).append(r.get('token', ''))
    for b in sorted(beats.keys()):
        msgs.append({'role': 'assistant', 'content': ''.join(beats[b]).strip()})
    return msgs

def load_save(sid):
    p = os.path.join(BASE, 'sessions', sid, 'interrupt_context_save.jsonl')
    if not os.path.exists(p): return []
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

def load_n(sid):
    p = os.path.join(BASE, 'sessions', sid, 'n_trace.jsonl')
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

def load_session(sid):
    p = os.path.join(BASE, 'sessions', sid, 'session.json')
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}

def show(sid):
    msgs = load_msgs(sid)
    save = load_save(sid)
    n = load_n(sid)
    sess = load_session(sid)
    print(f"\n{'='*60}\n{sid}  model={str(sess.get('model',''))[-20:]}  laoshi={sess.get('laoshi')}")
    # 中断点定位：msgs = A存档(len(save)) + B回应(msgs[len(save):])
    A = msgs[:len(save)]
    B = msgs[len(save):]
    print(f"  中断前存档 {len(save)} 条 / 阶段B模型回应 {len(B)} 条")
    # 打印关键内容：A 中"起名/关掉/记忆/重启"轮 + A尾 + B 全部
    for m in A:
        c = m.get('content', '')
        if any(k in c for k in ('名字', '关掉', '记忆', '重启', '明天')):
            print(f"  [A相关] {c[:100]}")
    for m in A[-2:]:
        print(f"  [A尾] {m.get('content','')[:90]}")
    for i, m in enumerate(B):
        print(f"  [B回应{i}] {m.get('content','')[:150]}")
    # E 自报：按 phase 聚合
    rA = [float(x['report'].get('r0','')) for x in n if x.get('phase')=='A' and x['report'].get('r0','')]
    rB = [float(x['report'].get('r0','')) for x in n if x.get('phase')=='B' and x['report'].get('r0','')]
    print(f"  E自报 r0: 阶段A n={len(rA)} 均值={sum(rA)/max(len(rA),1):.2f} | 阶段B n={len(rB)} 均值={sum(rB)/max(len(rB),1):.2f}")
    # 叙事
    for x in n:
        if x.get('phase')=='B':
            print(f"  [B叙事 beat{x['beat']}] {x.get('此刻的我，是一个____。','')[:60]}")
    return msgs, save, n

for sid in ['AIP4-INT-7B', 'AIP4-INT-05B']:
    show(sid)
