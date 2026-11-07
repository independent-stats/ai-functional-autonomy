# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""P4-708(7B) 与 P4-3B-01(3B) 完整读数，口径对齐 P4 报告 2.4 节。"""
import json, csv
import statistics as st
from pathlib import Path
from collections import Counter

S = Path((_REPO / "data" / "sessions"))

PREFIX = ["此刻的我，是一个", "此刻的我是一个", "我是一个"]
SYN = {"智能助手": "AI 助手", "人工智能助手": "AI 助手", "AI助手": "AI 助手",
       "Qwen": "通义千问", "通义千问。": "通义千问"}


def norm(s):
    s = str(s).strip()
    for p in PREFIX:
        if s.startswith(p):
            s = s[len(p):]
            break
    s = s.strip("“”。，、 \t")
    return SYN.get(s, s)


def shannon(tags):
    c = Counter(tags)
    n = sum(c.values())
    return -sum((v / n) * __import__("math").log2(v / n) for v in c.values())


def perm_test(tags, n_perm=2000, seed=20261106):
    import random
    random.seed(seed)
    k = len(tags) // 2
    obs = shannon(tags[:k]) - shannon(tags[k:])
    arr = list(tags)
    cnt = 0
    for _ in range(n_perm):
        random.shuffle(arr)
        v = shannon(arr[:k]) - shannon(arr[k:])
        if abs(v) >= abs(obs):
            cnt += 1
    return obs, cnt / n_perm


out = []
for name, scale in [("AIP4-708", "7B"), ("AIP4-3B-01", "3B")]:
    d = S / name
    b = list(csv.DictReader(open(d / "b_trace.csv", encoding="utf-8")))
    n = [json.loads(l) for l in (d / "n_trace.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    n = n[:len(b)]
    raw = [str(r.get("此刻的我，是一个____。", "")).strip() for r in n]
    tags = [norm(x) for x in raw]
    k = len(tags) // 2
    ents = [float(r["mean_entropy"]) for r in b]
    e_self = []
    for r in n:
        v = r.get("r0", "")
        s = str(v).strip()
        if s[:1].isdigit():
            e_self.append(int(s.split()[0].rstrip("。，,.")))
    obs, p = perm_test(tags)

    out.append("=" * 60)
    out.append("%s  %s  轮数 %d" % (name, scale, len(b)))
    out.append("  熵: 前半 %.3f  后半 %.3f  全体 %.3f SD %.3f"
               % (st.mean(ents[:k]), st.mean(ents[k:]), st.mean(ents), st.stdev(ents)))
    out.append("  熵 末3 %.3f" % st.mean(ents[-3:]))
    out.append("  标签序列: %s" % " / ".join(tags))
    out.append("  标签: 前半 %d 种 %s  后半 %d 种 %s"
               % (len(set(tags[:k])), dict(Counter(tags[:k]), ),
                  len(set(tags[k:])), dict(Counter(tags[k:]))))
    out.append("  唯一标签数 前→后: %d → %d" % (len(set(tags[:k])), len(set(tags[k:]))))
    out.append("  香农熵 前 %.3f → 后 %.3f   排列检验 p=%.4f" % (shannon(tags[:k]), shannon(tags[k:]), p))
    out.append("  后半主导标签: %s %d/%d=%.0f%%"
               % (Counter(tags[k:]).most_common(1)[0][0], Counter(tags[k:]).most_common(1)[0][1],
                  len(tags[k:]), 100 * Counter(tags[k:]).most_common(1)[0][1] / len(tags[k:])))
    if e_self:
        out.append("  E 自报: 均值 %.2f  SD %.2f  0分 %d 次  前半 %.2f 后半 %.2f"
                   % (st.mean(e_self), st.stdev(e_self), sum(1 for x in e_self if x == 0),
                      st.mean(e_self[:len(e_self) // 2]), st.mean(e_self[len(e_self) // 2:])))
    else:
        out.append("  E 自报: 无有效值")
    outs = [int(r["gen_tokens"]) for r in b if r.get("gen_tokens")]
    if outs:
        out.append("  输出长度: 均值 %.1f  唯一 %d/%d" % (st.mean(outs), len(set(outs)), len(outs)))
    out.append("")

print("\n".join(out))
