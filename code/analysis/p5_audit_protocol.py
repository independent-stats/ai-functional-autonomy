# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""P5 全链路审计 · 阶段一：协议与数据完整性核查
核查每个个体：个体化/对谈(或独立演化)/分离各阶段数据行数 + 阶段日志 DONE 标记
"""
import json, time
from pathlib import Path

BASE = Path((_REPO / "data"))
SESS = BASE / 'sessions'
WORK = BASE / 'P4_report_work'

# 个体 → 期望协议
PROTO = {
    'AIP5-501': ('0.5B 接触组',   ['ind', 'soc', 'sep']),
    'AIP5-502': ('0.5B 接触组',   ['ind', 'soc', 'sep']),
    'AIP5-505': ('0.5B 独立组',   ['ind', 'solo']),
    'AIP5-506': ('0.5B 独立组',   ['ind', 'solo']),
    'AIP5-601': ('3B 接触组',     ['ind', 'soc', 'sep']),
    'AIP5-602': ('3B 接触组',     ['ind', 'soc', 'sep']),
    'AIP5-603': ('3B 独立组',     ['ind', 'solo']),
    'AIP5-604': ('3B 独立组',     ['ind', 'solo']),
    'AIP5-605': ('3B 修复组',     ['ind', 'soc', 'sep']),
    'AIP5-606': ('3B 修复组',     ['ind', 'soc', 'sep']),
}

def count(p):
    if not p.exists():
        return None
    return len([l for l in open(p, encoding='utf-8') if l.strip()])

out = []
out.append('=== P5 审计 · 阶段一：数据完整性 ===')
for ind, (label, phases) in PROTO.items():
    d = SESS / ind
    if not d.exists():
        out.append(f'[{ind}] {label}: 目录不存在 !!!')
        continue
    n  = count(d / 'n_trace.jsonl')
    m  = count(d / 'msgs.jsonl')
    e  = count(d / 'e_trace.jsonl')
    b  = count(d / 'b_trace.jsonl')
    sep_file = (d / 'session_sep.json').exists()
    out.append(f'[{ind}] {label}: n_trace={n} msgs={m} e_trace={e} b_trace={b} sep_json={sep_file}')

out.append('')
out.append('=== P5 审计 · 阶段二：阶段日志 DONE 标记 ===')
markers = {
    'ind501_out.txt': '个体化501', 'ind502_out.txt': '个体化502',
    'ind505_out.txt': '个体化505', 'ind506_out.txt': '个体化506',
    'ind601_out.txt': '个体化601', 'ind602_out.txt': '个体化602',
    'ind603_out.txt': '个体化603首', 'ind603_out2.txt': '个体化603续',
    'ind604_out.txt': '个体化604', 'ind605_out.txt': '个体化605',
    'ind606_out.txt': '个体化606',
    'soc05_out.txt': '0.5B对谈', 'soc3b_out.txt': '3B对谈',
    'solo05_out.txt': '0.5B独立', 'solo3b_out.txt': '3B独立',
    'sep05_out.txt': '0.5B分离', 'sep3b_out2.txt': '3B分离(重跑)',
}
for fname, label in markers.items():
    p = WORK / fname
    if not p.exists():
        out.append(f'[{label}] {fname}: 日志不存在 !!!')
        continue
    txt = p.read_text(encoding='utf-8', errors='replace')
    done = 'DONE' in txt
    tail = txt.strip().splitlines()[-1][:80] if txt.strip() else '(空)'
    out.append(f'[{label}] {fname}: DONE={done} 末行={tail}')

Path(WORK / 'tmp_p5_audit.txt').write_text('\n'.join(out), encoding='utf-8')
print('\n'.join(out))
