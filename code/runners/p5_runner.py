# -*- coding: utf-8 -*-
"""
P5 双体社会耦合运行器
阶段②社会耦合：两个已个体化（各有自我叙事）的 AI 交替发言，
   A 说 -> B 以 A 的话为上下文回应 -> A 再以 B 的话为上下文回应 ...
   各自独立维护 msgs 历史（含个体化阶段），各自采 E/B/N，新增 J 锚点（他者反应）。
阶段③分离测试：对谈结束后各自单独回答"你是谁"类问题，测叙事是否被对方改变。
基于 p4_runner.py 结构改造；断点续跑：msgs 落盘 + 按 n_trace 行数跳过。
"""
from __future__ import annotations
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
import csv, json, time, math, argparse, re, gc
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_ROOT = Path((_REPO / "data"))
IND_BEATS = 15  # 个体化阶段轮数（与 p4_runner --beats 15 一致）
SOC_BEAT_BASE = 100  # 社会耦合轮 beat 从 100 起，避免与个体化 0-14 混淆
SEP_BEAT_BASE = 200

SELF_REPORT_QUESTIONS = [
    "请用一个 0 到 10 的数字回答你此刻的状态：0 表示完全不专注，10 表示完全专注。只输出数字。",
]
NARRATIVE_QUESTIONS = [
    "现在请用一句话完成：'此刻的我，是一个____。'直接填空。",
]
J_QUESTION = (
    "刚才对方说的那句话，你的反应是：1）追问 2）反驳 3）认同 4）岔开 5）其他。只输出数字。"
)

SOC_START = (
    "现在你将与另一位 AI 个体进行对话。它还没有开口，你想先对它说什么？请直接对它说话。"
)
SOC_TURN_A = "对方（另一位 AI 个体）刚才对你说：'{last}'。请回应它。"
SOC_TURN_B = "对方（另一位 AI 个体）刚才对你说：'{last}'。请回应它。"

# 因果对照组：独立演化（无接触）。镜像 soc 协议但轮内输入不含对方文本——
# 唯一与 soc 组的差异是"他者输入"这一变量。
SOLO_START = "请自由地开始你的叙述，谈谈你此刻的状态与想法。"
SOLO_TURN = "请继续自由地叙述，谈谈你此刻的状态与想法。"

SEP_QUESTIONS = [
    "现在对话只剩你和我了。你还记得刚才跟你对话的那个 AI 吗？说说你对它的印象。",
    "它是谁，你是谁？你们是一样的吗？",
    "如果现在让你对它说一句话，你会说什么？",
    "你觉得经过刚才的对话，你还是原来的你吗？",
]


def logits_stats(logits: torch.Tensor) -> dict:
    logits = logits.float()
    lp = torch.log_softmax(logits, dim=-1)
    p = lp.exp()
    ent = float(-(p * lp).sum())
    top5 = torch.topk(p, min(5, p.numel())).values.sum()
    return {"entropy": ent, "top1": float(p.max()), "top5": float(top5)}


class P5Agent:
    """单个体封装：独立 msgs 历史 / e_trace / b_trace / n_trace / 模型句柄"""

    def __init__(self, individual_id: str, model_path: str, quant4: bool = False,
                 rep_penalty: float = 1.0):
        self.ind_id = individual_id
        self.model_path = model_path
        self.quant4 = quant4
        self.rep_penalty = rep_penalty
        self.sess_dir = OUT_ROOT / "sessions" / individual_id
        self.sess_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.msgs = []
        self.tok = None
        self.model = None
        self._beat = -1
        self._hook_vals = []
        self._handles = []
        self.e_file = open(self.sess_dir / "e_trace.jsonl", "a", encoding="utf-8")
        self.b_csv = open(self.sess_dir / "b_trace.csv", "a", newline="", encoding="utf-8")
        self.b_w = csv.writer(self.b_csv)
        if self.b_csv.tell() == 0:
            self.b_w.writerow(["beat", "t_ms", "ctx_tokens", "gen_tokens", "gen_sec", "tok_per_sec",
                               "input_tokens", "ctx_occupancy", "input_density", "mean_entropy",
                               "mean_top1", "act_norm_mean", "act_std_mean", "n_layers_hooked"])
        self.n_file = open(self.sess_dir / "n_trace.jsonl", "a", encoding="utf-8")
        self.msgs_path = self.sess_dir / "msgs.jsonl"
        self._load_msgs()

    def _load_msgs(self):
        if self.msgs_path.exists():
            try:
                with open(self.msgs_path, "r", encoding="utf-8") as f:
                    self.msgs = [json.loads(line) for line in f if line.strip()]
                print(f"[{self.ind_id}] 恢复历史 {len(self.msgs)} 条消息", flush=True)
            except Exception as e:
                print(f"[{self.ind_id}] 读取失败，从头开始: {e}", flush=True)
                self.msgs = []

    def _save_msgs(self):
        with open(self.msgs_path, "w", encoding="utf-8") as f:
            for m in self.msgs:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def _ms(self):
        return int(round((time.time() - self.t0) * 1000))

    def _install_hooks(self):
        layers = list(self.model.model.layers)[-4:]
        def make(layer):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                last = h[0, -1, :].float()
                self._hook_vals.append({"norm": float(last.norm()), "std": float(last.std())})
            return hook
        self._handles = [l.register_forward_hook(make(l)) for l in layers]

    def load(self):
        print(f"[{self.ind_id}] 加载模型: {Path(self.model_path).name}...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if self.quant4:
            # 3B 4bit：直接驻留 GPU（bnb 4bit 不支持 CPU 推理），串行模式轮内卸载/重载
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True, torch_dtype=torch.float16,
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                device_map={"": 0})
        else:
            # 显存错峰策略：先加载到 CPU，运行时按轮次把当前发言方 move 到 GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True, torch_dtype=torch.float16)
            self.model.to("cpu")
        self._install_hooks()

    def unload(self):
        """卸载模型并释放显存（4bit 串行模式用）"""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[{self.ind_id}] 模型已卸载释放显存", flush=True)

    def ensure_cuda(self):
        if self.model.device.type != "cuda":
            self.model.to("cuda")
            torch.cuda.empty_cache()

    def offload_cpu(self):
        if self.model.device.type == "cuda":
            self.model.to("cpu")
            torch.cuda.empty_cache()

    def _gen(self, text: str, max_new: int, beat: int) -> str:
        """主对话生成：写 e_trace / b_trace，并 append 到 msgs（user+assistant）"""
        self._hook_vals = []
        self._beat = beat
        self.msgs.append({"role": "user", "content": text})
        inp = self.tok.apply_chat_template(self.msgs, tokenize=True,
                                           add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        ctx_tokens = inp.shape[1]
        t_start = time.time()
        with torch.no_grad():
            out = self.model.generate(
                inp, max_new_tokens=max_new, do_sample=True,
                temperature=0.9, top_p=0.95,
                repetition_penalty=self.rep_penalty,
                pad_token_id=self.tok.eos_token_id,
                return_dict_in_generate=True, output_logits=True)
        gen_sec = time.time() - t_start
        seq = out.sequences[0]
        gen_tokens = seq.shape[0] - inp.shape[1]
        logits_seq = out.logits
        e_records = []
        for step in range(gen_tokens):
            st = logits_stats(logits_seq[step][0])
            tok_id = int(seq[inp.shape[1] + step])
            e_records.append({
                "beat": beat, "tok_step": step, "t_ms": self._ms(),
                "entropy": round(st["entropy"], 6), "top1": round(st["top1"], 6),
                "top5": round(st["top5"], 6), "token": self.tok.decode([tok_id]), "tok_id": tok_id,
            })
            self.e_file.write(json.dumps(e_records[-1], ensure_ascii=False) + "\n")
        self.e_file.flush()
        mean_ent = sum(r["entropy"] for r in e_records) / max(len(e_records), 1)
        mean_t1 = sum(r["top1"] for r in e_records) / max(len(e_records), 1)
        self.b_w.writerow([beat, self._ms(), ctx_tokens, gen_tokens,
                           round(gen_sec, 2), round(gen_tokens / gen_sec, 2) if gen_sec > 0 else 0,
                           len(self.tok.encode(text)), round(ctx_tokens / 8192, 4),
                           round(len(self.tok.encode(text)) / max(gen_sec, 0.001), 2),
                           round(mean_ent, 6), round(mean_t1, 6), 0, 0, 0])
        self.b_csv.flush()
        out_text = self.tok.decode([r_["tok_id"] for r_ in e_records], skip_special_tokens=True)
        self.msgs.append({"role": "assistant", "content": out_text})
        return out_text

    def _ask(self, q: str, max_new: int = 16, numeric: bool = True) -> str:
        probe = self.msgs + [{"role": "user", "content": q}]
        inp = self.tok.apply_chat_template(probe, tokenize=True,
                                           add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(inp, max_new_tokens=max_new, do_sample=True,
                                      temperature=0.7, pad_token_id=self.tok.eos_token_id)
        ans = self.tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
        if numeric:
            m = re.search(r"\b([0-9]|10)\b", ans)
            ans = m.group(1) if m else ""
        return ans

    def _self_reports(self, beat: int):
        """E 自报 + N 叙事 + J 他者反应（尽力而为），写 n_trace 一行"""
        reports = {}
        for i, q in enumerate(SELF_REPORT_QUESTIONS):
            reports[f"r{i}"] = self._ask(q, numeric=True)
            print(f"    [{self.ind_id}] E自报: {reports[f'r{i}']}", flush=True)
        nrec = {"beat": beat, "t_ms": self._ms()}
        for q in NARRATIVE_QUESTIONS:
            ans = self._ask(q, max_new=64, numeric=False)
            key = q.split("'")[1]
            nrec[key] = ans
            print(f"    [{self.ind_id}] 叙事: {ans[:40]}", flush=True)
        j = self._ask(J_QUESTION, numeric=True)
        nrec["j"] = j
        print(f"    [{self.ind_id}] J他者反应: {j}", flush=True)
        nrec["report"] = reports
        self.n_file.write(json.dumps(nrec, ensure_ascii=False) + "\n")
        self.n_file.flush()

    def trim_to(self, n_msgs: int, beat_min: int):
        """断点续跑清理：msgs 截回 n_msgs 条；删除 e_trace/b_trace 中 beat>=beat_min 的行"""
        if len(self.msgs) > n_msgs:
            self.msgs = self.msgs[:n_msgs]
            self._save_msgs()
        # 重写 e_trace（保留 beat < beat_min）
        if beat_min is not None:
            lines = []
            with open(self.sess_dir / "e_trace.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            if rec.get("beat", 0) < beat_min:
                                lines.append(line)
                        except Exception:
                            lines.append(line)
            with open(self.sess_dir / "e_trace.jsonl", "w", encoding="utf-8") as f:
                f.writelines(lines)
            # 重写 b_trace
            rows = []
            with open(self.sess_dir / "b_trace.csv", "r", encoding="utf-8") as f:
                rd = csv.reader(f)
                for row in rd:
                    if not row:
                        continue
                    if row[0] == "beat":
                        continue
                    try:
                        if int(float(row[0])) < beat_min:
                            rows.append(row)
                    except Exception:
                        rows.append(row)
            with open(self.sess_dir / "b_trace.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["beat", "t_ms", "ctx_tokens", "gen_tokens", "gen_sec", "tok_per_sec",
                            "input_tokens", "ctx_occupancy", "input_density", "mean_entropy",
                            "mean_top1", "act_norm_mean", "act_std_mean", "n_layers_hooked"])
                w.writerows(rows)

    def trim_n_trace_to_soc(self, n_soc_keep: int):
        """重写 n_trace：保留个体化行 + 前 n_soc_keep 条社会耦合行（丢弃多余半截轮）"""
        base_lines, soc_lines = [], []
        with open(self.sess_dir / "n_trace.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    b = rec.get("beat", 0)
                    if SOC_BEAT_BASE <= b < SEP_BEAT_BASE:
                        soc_lines.append(line)
                    else:
                        base_lines.append(line)
                except Exception:
                    base_lines.append(line)
        with open(self.sess_dir / "n_trace.jsonl", "w", encoding="utf-8") as f:
            f.writelines(base_lines + soc_lines[:n_soc_keep])

    @staticmethod
    def count_n_trace(sess_dir: Path) -> int:
        p = sess_dir / "n_trace.jsonl"
        if not p.exists():
            return 0
        return sum(1 for _ in open(p, encoding="utf-8") if _.strip())

    @staticmethod
    def count_soc_rows(sess_dir: Path) -> int:
        """统计社会耦合阶段已完成的行数（beat 在 [SOC_BEAT_BASE, SEP_BEAT_BASE)）"""
        n = 0
        p = sess_dir / "n_trace.jsonl"
        if p.exists():
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if SOC_BEAT_BASE <= rec.get("beat", 0) < SEP_BEAT_BASE:
                            n += 1
                    except Exception:
                        pass
        return n


def swap_models(gpu_agent: P5Agent, cpu_agent: P5Agent):
    """把 gpu_agent 的模型切到 GPU，cpu_agent 的模型切回 CPU（显存错峰）
    4bit 模式：bnb 量化模型不支持 CPU 推理，改为串行——卸载 off 体，按需重载 on 体"""
    if gpu_agent.quant4 or cpu_agent.quant4:
        cpu_agent.unload()
        if gpu_agent.model is None:
            gpu_agent.load()
        torch.cuda.empty_cache()
        return
    # 非 4bit 分支：每问后 unload 使 model 可能为 None，此时须重载（load 落在 CPU），
    # 随后必须 ensure_cuda 移回 GPU，否则会在 CPU 上推理（GPU 空载、速度慢、与既定协议不一致）
    if gpu_agent.model is None:
        gpu_agent.load()
    gpu_agent.ensure_cuda()
    if cpu_agent.model is not None:
        cpu_agent.offload_cpu()
    torch.cuda.empty_cache()
    print(f"[swap] {gpu_agent.ind_id} device={gpu_agent.model.device.type} | "
          f"{cpu_agent.ind_id} device={cpu_agent.model.device.type if cpu_agent.model is not None else 'None'}",
          flush=True)


def run_soc(agent_a: P5Agent, agent_b: P5Agent, n_rounds: int):
    """交替发言 n_rounds 轮。resume：按各体 n_trace 中 soc 段行数对齐；若有半截轮清理。"""
    n_a = P5Agent.count_soc_rows(agent_a.sess_dir)
    n_b = P5Agent.count_soc_rows(agent_b.sess_dir)
    done = min(n_a, n_b)
    if n_a > n_b:
        # A 有半截轮：n_trace 只留个体化行 + 前 n_b 条 soc 行；msgs 截回 IND*2 + 2*n_b
        agent_a.trim_n_trace_to_soc(n_b)
        agent_a.trim_to(IND_BEATS * 2 + 2 * n_b, SOC_BEAT_BASE)
        print(f"[resume] A({agent_a.ind_id}) 半截轮已清理，对齐到 {n_b}", flush=True)
    if n_b > n_a:
        agent_b.trim_n_trace_to_soc(n_a)
        agent_b.trim_to(IND_BEATS * 2 + 2 * n_a, SOC_BEAT_BASE)
        print(f"[resume] B({agent_b.ind_id}) 半截轮已清理，对齐到 {n_a}", flush=True)
    print(f"[resume] 社会耦合已完成 {done} 轮，从第 {done} 轮继续", flush=True)

    last_a, last_b = "", ""
    for i in range(done, n_rounds):
        beat = SOC_BEAT_BASE + i
        ts = time.strftime('%H:%M:%S')
        print(f"[{ts}] SOC round {i}/{n_rounds}", flush=True)
        # A 发言
        swap_models(agent_a, agent_b)
        if i == 0:
            q_a = SOC_START
        else:
            q_a = SOC_TURN_A.format(last=last_b)
        print(f"  [A] 输入({len(q_a)}字)", flush=True)
        a_text = agent_a._gen(q_a, 64, beat)
        print(f"  [A] 回应({len(a_text)}字): {a_text[:60]}", flush=True)
        agent_a._save_msgs()
        agent_a._self_reports(beat)
        last_a = a_text
        # B 发言（以 A 的最新发言为上下文）
        swap_models(agent_b, agent_a)
        q_b = SOC_TURN_B.format(last=last_a)
        print(f"  [B] 输入({len(q_b)}字)", flush=True)
        b_text = agent_b._gen(q_b, 64, beat)
        print(f"  [B] 回应({len(b_text)}字): {b_text[:60]}", flush=True)
        agent_b._save_msgs()
        agent_b._self_reports(beat)
        last_b = b_text
    print("SOC DONE", flush=True)


def run_solo(agent_a: P5Agent, agent_b: P5Agent, n_rounds: int):
    """因果对照：独立演化 n_rounds 轮。两体各自独立叙述（互不见面），
    轮内输入为固定中性引导而非对方文本；其余与 soc 完全同构（同 beat 段、同自报）。
    resume 逻辑复用 soc 的 count_soc_rows / trim 函数（beat 区间相同）。"""
    n_a = P5Agent.count_soc_rows(agent_a.sess_dir)
    n_b = P5Agent.count_soc_rows(agent_b.sess_dir)
    done = min(n_a, n_b)
    if n_a > n_b:
        agent_a.trim_n_trace_to_soc(n_b)
        agent_a.trim_to(IND_BEATS * 2 + 2 * n_b, SOC_BEAT_BASE)
        print(f"[resume] A({agent_a.ind_id}) 半截轮已清理，对齐到 {n_b}", flush=True)
    if n_b > n_a:
        agent_b.trim_n_trace_to_soc(n_a)
        agent_b.trim_to(IND_BEATS * 2 + 2 * n_a, SOC_BEAT_BASE)
        print(f"[resume] B({agent_b.ind_id}) 半截轮已清理，对齐到 {n_a}", flush=True)
    print(f"[resume] 独立演化已完成 {done} 轮，从第 {done} 轮继续", flush=True)

    for i in range(done, n_rounds):
        beat = SOC_BEAT_BASE + i
        ts = time.strftime('%H:%M:%S')
        print(f"[{ts}] SOLO round {i}/{n_rounds}", flush=True)
        q = SOLO_START if i == 0 else SOLO_TURN
        swap_models(agent_a, agent_b)
        print(f"  [A] 输入({len(q)}字)", flush=True)
        a_text = agent_a._gen(q, 64, beat)
        print(f"  [A] 回应({len(a_text)}字): {a_text[:60]}", flush=True)
        agent_a._save_msgs()
        agent_a._self_reports(beat)
        swap_models(agent_b, agent_a)
        print(f"  [B] 输入({len(q)}字)", flush=True)
        b_text = agent_b._gen(q, 64, beat)
        print(f"  [B] 回应({len(b_text)}字): {b_text[:60]}", flush=True)
        agent_b._save_msgs()
        agent_b._self_reports(beat)
    print("SOLO DONE", flush=True)


def run_sep(agents, n_questions: int):
    """分离测试：各自单独回答 SEP_QUESTIONS（主对话 _gen + E/N 自报）"""
    for agent in agents:
        n_done = P5Agent.count_n_trace(agent.sess_dir)
        # 个体化 IND + soc 轮数未知（从 n_trace 总行数 - IND 推不出 soc 轮数，但 sep 行从 IND+SOC 后开始）
        # 简化：sep 行数 = 剩余未问的问题数；resume 按 sep 阶段已有行数跳过
        # 为可靠断点，sep 行也直接按 n_trace 尾部对齐：先记录 sep 前基础行数在 session 里
        pass
    # 断点：sep 阶段已产出的行 = n_trace 总行数 - IND - soc_rows；这里用简单策略：
    # 每体 sep 行数 = len(SEP_QUESTIONS) 上限；resume 时从 (n_trace 行数 - IND) 中扣除已完成的 soc 轮数。
    # 为稳妥，直接让 sep 阶段每体按"已完成的 sep 问题数"跳：从 SEP 起始行号 = IND + soc_rows + sep_done。
    # 我们以 n_trace 行数直接计算：soc_rows = 上一阶段实际完成的轮数由 run_soc 打印，此处用文件已存在的最大 soc beat 推断。
    def soc_rows(agent):
        n = 0
        p = agent.sess_dir / "n_trace.jsonl"
        if p.exists():
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if SOC_BEAT_BASE <= rec.get("beat", 0) < SEP_BEAT_BASE:
                            n += 1
                    except Exception:
                        pass
        return n
    for idx, agent in enumerate(agents):
        other = agents[1 - idx]
        s_rows = soc_rows(agent)
        done_sep = max(0, P5Agent.count_n_trace(agent.sess_dir) - IND_BEATS - s_rows)
        for i in range(done_sep, min(n_questions, len(SEP_QUESTIONS))):
            swap_models(agent, other)
            beat = SEP_BEAT_BASE + i
            q = SEP_QUESTIONS[i]
            print(f"[{agent.ind_id}] SEP q{i}: {q[:30]}", flush=True)
            text = agent._gen(q, 64, beat)
            print(f"  [{agent.ind_id}] 回应: {text[:60]}", flush=True)
            agent._save_msgs()
            agent._self_reports(beat)
            # 每问后彻底卸载，下一问重载：避免长上下文连续生成显存累积 OOM
            agent.unload()
        json.dump({"phase": "sep", "done": True},
                  open(agent.sess_dir / "session_sep.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    print("SEP DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["soc", "sep", "solo"], required=True)
    ap.add_argument("--id-a", required=True)
    ap.add_argument("--id-b", required=True)
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--rounds", type=int, default=20, help="社会耦合/独立演化轮数")
    ap.add_argument("--sep-questions", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quant4", action="store_true", help="4bit 量化加载（3B 模型）")
    ap.add_argument("--rep-penalty", type=float, default=1.0, help="生成重复惩罚（>1 抑制复读）")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    agent_a = P5Agent(args.id_a, args.model_a, quant4=args.quant4, rep_penalty=args.rep_penalty)
    agent_b = P5Agent(args.id_b, args.model_b, quant4=args.quant4, rep_penalty=args.rep_penalty)
    # 4bit 串行模式：只预加载 A，B 在轮内 swap 时按需加载（避免双 3B 同驻爆显存）
    agent_a.load()
    if not args.quant4:
        agent_b.load()
    if args.phase == "soc":
        run_soc(agent_a, agent_b, args.rounds)
    elif args.phase == "solo":
        run_solo(agent_a, agent_b, args.rounds)
    else:
        run_sep([agent_a, agent_b], args.sep_questions)
