# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""分析 P4 的 3B / 7B 15 轮长序列，与 0.5B 50 轮对照。"""
import json, csv
import statistics as st
from pathlib import Path

S = Path((_REPO / "data" / "sessions"))


def load(name):
    d = S / name
    b = list(csv.DictReader(open(d / "b_trace.csv", encoding="utf-8")))
    n = [json.loads(l) for l in (d / "n_trace.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return b, n


out = []

for name, label in [("AIP4-3B-01", "3B"), ("AIP4-708", "7B"), ("AIP4-407", "0.5B(50轮)")]:
    b, n = load(name)
    ents = [float(r["mean_entropy"]) for r in b]
    outs = [int(r["gen_tokens"]) for r in b if r.get("gen_tokens")]
    out.append("== %s %s，%d 轮" % (name, label, len(b)))
    # 分四段均值
    k = max(1, len(ents) // 4)
    segs = [ents[i * k:(i + 1) * k] for i in range(4)]
    out.append("  熵四段均值: " + " / ".join("%.3f" % st.mean(s) for s in segs if s))
    out.append("  熵首3 %.3f  末3 %.3f  全体 %.3f  SD %.3f"
               % (st.mean(ents[:3]), st.mean(ents[-3:]), st.mean(ents), st.stdev(ents)))
    if outs:
        out.append("  输出长度: 首3 %.1f 末3 %.1f  不同取值 %d/%d"
                   % (st.mean(outs[:3]), st.mean(outs[-3:]), len(set(outs)), len(outs)))
    # 叙事标签
    tags = []
    for r in n:
        v = r.get("此刻的我，是一个____。", "")
        if isinstance(v, str) and v.strip():
            tags.append(v.strip())
    if tags:
        norm = []
        for t in tags:
            if any(k2 in t for k2 in ["阿里", "Qwen", "通义", "千问"]):
                norm.append("厂商")
            elif any(k2 in t for k2 in ["助手", "服务", "AI", "人工智能"]):
                norm.append("AI通用")
            elif any(k2 in t for k2 in ["系统", "模型", "网络", "权重", "算法", "代码"]):
                norm.append("系统")
            elif any(k2 in t for k2 in ["实验个体", "编号", "个体", "存在"]):
                norm.append("编号个体")
            else:
                norm.append("其它")
        from collections import Counter
        c = Counter(norm)
        out.append("  叙事类别分布: %s  不同 %d 类" % (dict(c), len(c)))
        # 首次锁定轮
        first = norm[0]
        lock_at = None
        for i, t in enumerate(norm):
            if t == norm[-1]:
                lock_at = i
                break
        out.append("  首类=%s 末类=%s 末类首次出现于第 %s 轮" % (norm[0], norm[-1], lock_at))
    out.append("")

print("\n".join(out))
