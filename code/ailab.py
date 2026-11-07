#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ailab —— 本仓库的统一数据加载器

设计目标：让使用者不需要了解六个阶段之间那些结构差异，直接拿到
干净、对齐、可以直接进统计的数据。

用法：

    import ailab

    s = ailab.load("AIP4-407")        # 读一个 session
    s.entropy                          # [1.21, 1.35, ...]  逐轮熵（B 锚点）
    s.self_report                      # [8, 7, ...]        E 锚点（已转成整数）
    s.narrative                        # ["我是一个...", ...]  N 锚点
    s.beats                            # 逐轮字典（含 beat/t_ms/entropy/...）
    s.meta                             # session.json 内容
    s.has                              # {"b":True,"e":True,"n":True,"msgs":False,"r":False}

    for s in ailab.iter_sessions(stage="P3"):
        print(s.id, len(s.entropy))

    ailab.catalog()                    # 全部 session 的概览（dict 列表）

它在内部处理了这些差异（所以你不用管）：

  1. b_trace.csv 有 4 种列结构（5/6/14/15 列）→ 一律按列名取值
  2. 第 4 列是 gen_tokens 不是熵 → 熵只从 mean_entropy 取
  3. n_trace 填空题 P1/P3 三道、P4/P5 一道 → 全部收进 narrative
  4. report 的值是字符串且可能带小数 → int(float(x))，坏值用 None
  5. mean_entropy 可能是 "nan" → 转成 None，不进统计
  6. b_trace 行数可能多于 n_beats → main_entropy 只取主段（beat < n_beats）
  7. 缺 session.json → meta 为空字典，不抛异常
  8. 缺文件 → 对应字段为空列表，不抛异常
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SESS_DIR = DATA / "sessions"
PERT_DIR = DATA / "perturbation"

STAGES = {
    "AI8B": "P1", "AI05B": "P1", "AIP1": "P1",
    "AIP2": "P2", "AISmol": "P2", "AIOLMo": "P2",
    "AIP3": "P3",
    "AIP4": "P4",
    "AIP5": "P5",
}


def _stage_of(sid):
    for pre, st in STAGES.items():
        if sid.startswith(pre):
            return st
    return "unknown"


def _num(v):
    """把 csv/json 里的数字字符串转成 float；nan/空/坏值 -> None"""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _int0(v):
    """report 的值：字符串、可能带小数；坏值 -> None"""
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _split_beat(b, stage):
    """
    拆出 (段号, 段内轮次)。

    P5 的双体对谈用 段号*100+轮次 编码：
        0..14   -> 第一段
        100..119 -> 第二段
        200..203 -> 第三段
    其它阶段 beat 就是轮次，段号恒为 0。
    """
    if b is None:
        return None, None
    if stage == "P5" and b >= 100:
        return int(b // 100), int(b % 100)
    return 0, int(b)


def _read_csv_dicts(p):
    if not p.exists():
        return []
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl(p):
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


class Session:
    """一个 session 的统一视图"""

    def __init__(self, sid, root=None):
        self.id = sid
        self.stage = _stage_of(sid)
        self.dir = Path(root or SESS_DIR) / sid

        # --- meta ---
        sj = self.dir / "session.json"
        if sj.exists():
            try:
                self.meta = json.loads(sj.read_text(encoding="utf-8"))
            except Exception:
                self.meta = {}
        else:
            self.meta = {}

        # --- b_trace：逐轮运行态 ---
        rows = _read_csv_dicts(self.dir / "b_trace.csv")
        self.beats = []
        for i, r in enumerate(rows):
            b = _num(r.get("beat"))
            seg, turn = _split_beat(b, self.stage)
            self.beats.append({
                "row": i,
                "beat": b,
                "seg": seg,
                "turn": turn,
                "t_ms": _num(r.get("t_ms")),
                "ctx_tokens": _num(r.get("ctx_tokens")),
                "gen_tokens": _num(r.get("gen_tokens")),   # 注意：不是熵
                "entropy": _num(r.get("mean_entropy")),
                "top1": _num(r.get("mean_top1")),
                "act_norm": _num(r.get("act_norm_mean")),  # 未挂 hook 时为 0
                "noise": _num(r.get("noise")),             # 仅 P3 有
                "state": r.get("state"),                   # 仅 perturbation 类
                "response": r.get("response"),              # 仅 perturbation 类
                "dup": False,
            })
        # 去重：同一 (seg, turn) 出现多次时保留最后一条（续跑后的有效记录）
        seen = {}
        for b in self.beats:
            key = (b["seg"], b["turn"])
            if key in seen:
                seen[key]["dup"] = True
            seen[key] = b
        keep = {id(b) for b in seen.values()}
        self.beats = [b for b in self.beats if id(b) in keep]
        self.raw_beat_rows = len(rows)

        # --- n_trace：叙事 + 自报 ---
        nrows = _read_jsonl(self.dir / "n_trace.jsonl")
        self.narrative = []
        self.self_report = []
        for r in nrows:
            parts = [v for k, v in r.items()
                     if isinstance(k, str) and "____" in k and isinstance(v, str) and v]
            self.narrative.append(" ".join(parts) if parts else "")
            rep = r.get("report")
            if isinstance(rep, dict):
                # 三道题的阶段取 r0（与单道题阶段口径一致）
                v = rep.get("r0", rep.get("r1", rep.get("r2")))
                self.self_report.append(_int0(v))
            else:
                self.self_report.append(None)

        # --- e_trace：逐 token ---
        self.tokens = _read_jsonl(self.dir / "e_trace.jsonl")

        # --- msgs：多轮对话 ---
        self.messages = _read_jsonl(self.dir / "msgs.jsonl")

        # --- 可用性标记 ---
        r = self.dir / "r_trace.jsonl"
        self.has = {
            "b": bool(rows),
            "e": bool(self.tokens),
            "n": bool(nrows),
            "msgs": bool(self.messages),
            "r": bool(r.exists() and r.stat().st_size > 0),
            "session_json": sj.exists(),
        }

    # ---------- 便捷属性 ----------

    @property
    def model(self):
        m = self.meta.get("model", "")
        if isinstance(m, str):
            return m.replace("\\", "/").split("/")[-1] if m else ""
        return ""

    @property
    def n_beats(self):
        for k in ("n_beats", "beats", "max_beats"):
            if k in self.meta and self.meta[k] is not None:
                try:
                    return int(self.meta[k])
                except (TypeError, ValueError):
                    pass
        return None

    @property
    def entropy(self):
        """逐轮熵，None 已被剔除（保留位置对齐用 None）"""
        return [b["entropy"] for b in self.beats]

    @property
    def main_beats(self):
        """
        主段。

        非 P5：beat < n_beats 的记录（滤掉多记的轮次）。
        P5：第一段（seg == 0）。P5 的 beat 是 段号*100+轮次，不能按 beat<n_beats 过滤。
        """
        if self.stage == "P5":
            return [b for b in self.beats if b["seg"] == 0]
        nb = self.n_beats
        if nb is None:
            return self.beats
        return [b for b in self.beats if b["beat"] is not None and b["beat"] < nb]

    @property
    def main_entropy(self):
        return [b["entropy"] for b in self.main_beats]

    @property
    def segments(self):
        """
        按段拆分（P5 才有意义；其它阶段只有一个段）。
        返回 {段号: [beat 字典...]}
        """
        out = {}
        for b in self.beats:
            out.setdefault(b["seg"], []).append(b)
        return out

    @property
    def entropy_series(self):
        """按 (段号, 轮次) 排序后的完整熵序列，可直接画整条曲线"""
        order = sorted(self.beats, key=lambda b: (b["seg"] or 0, b["turn"] or 0))
        return [b["entropy"] for b in order]

    def entropy_clean(self):
        """只返回非 None 的熵值，可直接进统计"""
        return [e for e in self.entropy if e is not None]

    def __repr__(self):
        return "<Session %s stage=%s model=%s beats=%d>" % (
            self.id, self.stage, self.model or "?", len(self.beats))


# ---------------------------------------------------------------- 对外接口

def load(sid, root=None):
    """读一个 session。sid 为目录名。"""
    return Session(sid, root=root)


def iter_sessions(stage=None, require=None, root=None):
    """
    遍历 sessions。

    stage:   "P1".."P5" 过滤
    require: 例如 ("b","n") 只返回同时有 b_trace 与 n_trace 的
    """
    base = Path(root or SESS_DIR)
    if not base.exists():
        return
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        s = Session(d.name, root=base)
        if stage and s.stage != stage:
            continue
        if require and not all(s.has.get(x) for x in require):
            continue
        yield s


def catalog(root=None):
    """全部 session 的轻量概览，返回 dict 列表（便于 pandas 直接吃）"""
    out = []
    for s in iter_sessions(root=root):
        out.append({
            "session_id": s.id,
            "stage": s.stage,
            "model": s.model,
            "n_beats": s.n_beats,
            "b_rows": len(s.beats),
            "n_rows": len(s.narrative),
            "e_tokens": len(s.tokens),
            "msgs_rows": len(s.messages),
            "has_r": s.has["r"],
        })
    return out


def load_perturbation(name=None):
    """
    读 data/perturbation/ 下的"状态-回应"记录（另一类数据，不是 session）。
    返回 dict: {"id":..., "rows":[{"beat","state","top1","response",...}]}
    """
    if not PERT_DIR.exists():
        return None
    if name is None:
        dirs = sorted(d for d in PERT_DIR.iterdir() if d.is_dir())
        name = dirs[0].name if dirs else None
    if name is None:
        return None
    d = PERT_DIR / name
    rows = _read_csv_dicts(d / "b_trace.csv")
    dose = _read_jsonl(d / "dose_trace.jsonl")
    return {"id": name, "rows": rows, "dose": dose}


if __name__ == "__main__":
    print("仓库根: %s" % REPO)
    cats = catalog()
    from collections import Counter
    print("session 总数: %d" % len(cats))
    print("阶段分布:", dict(Counter(c["stage"] for c in cats)))
    print("可用性:", {
        "b": sum(1 for c in cats if c["b_rows"]),
        "n": sum(1 for c in cats if c["n_rows"]),
        "e": sum(1 for c in cats if c["e_tokens"]),
        "msgs": sum(1 for c in cats if c["msgs_rows"]),
        "r": sum(1 for c in cats if c["has_r"]),
    })
    if PERT_DIR.exists():
        print("perturbation 记录: %d" % len([d for d in PERT_DIR.iterdir() if d.is_dir()]))
