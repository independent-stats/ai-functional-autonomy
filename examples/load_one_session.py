#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最小示例：用 ailab 读一个 session，打印它的结构摘要。

用法：
    python examples/load_one_session.py                 # 默认读 AIP4-407
    python examples/load_one_session.py AIP5-501       # 指定目录
    python examples/load_one_session.py --list         # 列出全部 session

推荐姿势就是这几行：导入 ailab，拿属性，不碰原始文件的列位置。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import ailab  # noqa: E402


def show(sid):
    s = ailab.load(sid)

    print("=" * 68)
    print("session: %s" % sid)
    print("=" * 68)
    print("  阶段            %s" % s.stage)
    print("  模型            %s" % (s.model or "（未知）"))
    print("  n_beats         %s" % s.n_beats)
    print("  b_trace 行数    %d%s" % (
        len(s.beats), "" if s.raw_beat_rows == len(s.beats)
        else "（原始 %d 行，已去重）" % s.raw_beat_rows))
    print("  可用锚点        %s" % ", ".join(
        k for k in ("b", "e", "n", "msgs", "r") if s.has.get(k)))
    print()

    ent = s.entropy_clean()
    if ent:
        print("B 锚点（mean_entropy）")
        print("  首轮 %.6f   末轮 %.6f   范围 [%.6f, %.6f]"
              % (ent[0], ent[-1], min(ent), max(ent)))

    rep = [x for x in s.self_report if x is not None]
    if rep:
        print("E 锚点（report.r0，已转 int）")
        print("  共 %d 个取值，首个 %s" % (len(rep), rep[0]))

    if s.narrative:
        first = next((x for x in s.narrative if x), "")
        print("N 锚点（叙事填空）")
        print("  %d 条，首条非空内容: %s" % (len(s.narrative), first[:40] or "（该轮为空）"))

    if s.stage == "P5" and len(s.segments) > 1:
        print("P5 分段（beat 编码为 段号*100+轮次）")
        for seg, beats in sorted(s.segments.items()):
            print("  段 %d: %d 轮，熵 %.4f -> %.4f"
                  % (seg, len(beats),
                     beats[0]["entropy"], beats[-1]["entropy"]))

    if s.messages:
        print("对话记录: %d 行" % len(s.messages))
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        cat = ailab.catalog()
        for c in cat:
            print("%-24s %-3s %-26s beats=%d" % (
                c["session_id"], c["stage"], c["model"] or "?", c["b_rows"]))
        print("\n共 %d 个 session" % len(cat))
    else:
        sid = sys.argv[1] if len(sys.argv) > 1 else "AIP4-407"
        show(sid)
