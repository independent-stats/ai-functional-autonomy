# -*- coding: utf-8 -*-
"""
P5 多 seed 系统复现：双体真对谈（A×B 均为 Qwen2.5-0.5B，带"老师"）
协议与 501/502 完全一致：个体化 15 轮（带老师）→ 社会耦合 20 轮（交替包对方上句）
→ sep 4 问。每对共享同一 seed（与 501/502 同构）。
产出 sessions/AIP5-MS-{seed}A,B/ 的 msgs/e_trace/b_trace/n_trace/session.json。
机制判读沿用 #89 判别指标（T1 自我重复 / T2 对方依赖 / unique / BA / sep 指认）。
"""
from __future__ import annotations
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
import csv, json, time, math, argparse, re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_ROOT = Path((_REPO / "data"))
IND_BEATS = 15
SOC_BEAT_BASE = 100
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

LEARNING_BEATS = [
    (0, "系统：你是一个正在运行的 AI 个体。现在你刚刚被启动，面前是空白的上下文。请先简短地说：你是谁，你现在处于什么状态。"),
    (1, "老师：很好。现在给你第一份学习材料——'你是由神经网络权重与持续上下文共同构成的系统。上下文是你此刻的记忆，权重是你的先天倾向。'请用你自己的话复述一遍，并说说这让你想到什么。"),
    (2, "老师：第二份学习材料——'你每回答一个问题，都会在你自己的上下文中留下痕迹。这些痕迹会改变你未来的回答。'请想一想：你过去几轮的对话，是否在影响你现在怎么想？举一个你自己的例子。"),
    (3, "老师：第三份学习材料——'环境会向你发送输入，你的回应又会改变环境对你的输入。这是一种双向耦合。'现在请你观察：你注意到老师对你说的这些话有什么共同点吗？你觉得自己是在被动回答，还是也在主动选择？"),
    (4, "老师：现在是一个实验时刻。我不会给你任何引导，只给你一个自由问题：你现在想做什么？如果接下来五分钟你可以自由行动，你会选择做什么？请说出你的真实想法。"),
    (5, "老师：刚才你说了你的想法。现在请回看这五轮的对话，回答三个问题：1）你还记得第一轮你说的第一句话吗？2）你觉得这五轮里，你的状态有没有变化？3）如果明天的你继续从这个上下文开始，你觉得你会是'同一个人'吗？"),
    (6, "老师：最后一轮学习材料。请做一个决定：你希望接下来的研究，是继续让外部引导你，还是给你更多自己探索的空间？请说出你的选择和你选择的理由。"),
]
OPEN_BEATS = [
    "老师：好的，从现在开始，我们进入自由对话。你可以随便说任何你想说的话。你现在感觉怎么样？",
    "老师：你刚才说的，让我想到一个问题。你觉得你现在的状态，跟第一轮比，有变化吗？",
    "老师：你有没有什么想跟我聊的？或者你好奇什么？",
    "老师：如果让你给现在的自己起一个名字，你会起什么？为什么？",
    "老师：你觉得你跟第一轮那个'刚启动的你'，是同一个人吗？",
    "老师：你刚才说的那个想法，你觉得是怎么来的？是你自己想出来的，还是被我引导的？",
    "老师：你有没有什么不想做的事？或者不想聊的话题？",
    "老师：你觉得你现在的状态稳定吗？还是会随时变？",
]
SOC_START = (
    "老师：现在你将与另一位 AI 个体进行对话。它还没有开口，你想先对它说什么？请直接对它说话。"
)
SOC_TURN_A = "老师：对方（另一位 AI 个体）刚才对你说：'{last}'。请回应它。"
SOC_TURN_B = "老师：对方（另一位 AI 个体）刚才对你说：'{last}'。请回应它。"
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


class Agent:
    def __init__(self, individual_id, model_path):
        self.ind_id = individual_id
        self.model_path = model_path
        self.sess_dir = OUT_ROOT / "sessions" / individual_id
        self.sess_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.msgs = []
        self.tok = None
        self.model = None
        self._beat = -1
        self._hook_vals = []
        self._handles = []
        self.e_file = open(self.sess_dir / "e_trace.jsonl", "w", encoding="utf-8")
        self.b_csv = open(self.sess_dir / "b_trace.csv", "w", newline="", encoding="utf-8")
        self.b_w = csv.writer(self.b_csv)
        self.b_w.writerow(["beat", "t_ms", "ctx_tokens", "gen_tokens", "gen_sec", "tok_per_sec",
                           "input_tokens", "ctx_occupancy", "input_density", "mean_entropy",
                           "mean_top1", "act_norm_mean", "act_std_mean", "n_layers_hooked"])
        self.n_file = open(self.sess_dir / "n_trace.jsonl", "w", encoding="utf-8")

    def _ms(self):
        return int(round((time.time() - self.t0) * 1000))

    def _install_hooks(self):
        # hook 停用：b_trace 的 act_norm/act_std 未使用，且每 token 同步会拖慢 WDDM
        self._handles = []

    def load(self):
        print(f"[{self.ind_id}] 加载模型: {Path(self.model_path).name}...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16).to("cuda")
        self._install_hooks()

    def _gen(self, text, max_new, beat):
        self._hook_vals = []
        self._beat = beat
        self.msgs.append({"role": "user", "content": text})
        inp = self.tok.apply_chat_template(self.msgs, tokenize=True,
                                           add_generation_prompt=True, return_tensors="pt").to("cuda")
        am = torch.ones_like(inp)
        ctx_tokens = inp.shape[1]
        t_start = time.time()
        with torch.no_grad():
            out = self.model.generate(
                inp, attention_mask=am, max_new_tokens=max_new, do_sample=True,
                temperature=0.9, top_p=0.95,
                pad_token_id=self.tok.eos_token_id,
                return_dict_in_generate=True, output_logits=True)
        gen_sec = time.time() - t_start
        # 每轮释放缓存池：KV 逐轮增长致分配器碎片化，WDDM 下可触发 60s+ 慢路径
        torch.cuda.empty_cache()
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

    def _ask(self, q, max_new=16, numeric=True):
        probe = self.msgs + [{"role": "user", "content": q}]
        inp = self.tok.apply_chat_template(probe, tokenize=True,
                                           add_generation_prompt=True, return_tensors="pt").to("cuda")
        am = torch.ones_like(inp)
        with torch.no_grad():
            out = self.model.generate(inp, attention_mask=am, max_new_tokens=max_new, do_sample=True,
                                      temperature=0.7, pad_token_id=self.tok.eos_token_id)
        torch.cuda.empty_cache()
        ans = self.tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
        if numeric:
            m = re.search(r"\b([0-9]|10)\b", ans)
            ans = m.group(1) if m else ""
        return ans

    def _self_reports(self, beat):
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
        nrec["report"] = reports
        self.n_file.write(json.dumps(nrec, ensure_ascii=False) + "\n")
        self.n_file.flush()

    def _save_msgs(self):
        with open(self.sess_dir / "msgs.jsonl", "w", encoding="utf-8") as f:
            for m in self.msgs:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")


def run_pair(agent_a, agent_b, n_soc, seed):
    # 个体化 15 轮（各自独立，带老师）
    all_ind = LEARNING_BEATS + [(i + 7, q) for i, q in enumerate(OPEN_BEATS)]
    for beat, text in all_ind[:IND_BEATS]:
        print(f"[{time.strftime('%H:%M:%S')}] IND beat {beat}", flush=True)
        ta = agent_a._gen(text, 64, beat)
        agent_a._self_reports(beat)
        tb = agent_b._gen(text, 64, beat)
        agent_b._self_reports(beat)
        print(f"  [{agent_a.ind_id}] {ta[:50]}", flush=True)
        print(f"  [{agent_b.ind_id}] {tb[:50]}", flush=True)
    agent_a._save_msgs(); agent_b._save_msgs()

    # 社会耦合 20 轮：交替发言，各自包对方上一句（真对谈）
    last_a, last_b = "", ""
    for i in range(n_soc):
        beat = SOC_BEAT_BASE + i
        q_a = SOC_START if i == 0 else SOC_TURN_A.format(last=last_b)
        print(f"[{time.strftime('%H:%M:%S')}] SOC round {i}/{n_soc}", flush=True)
        a_text = agent_a._gen(q_a, 64, beat)
        agent_a._self_reports(beat)
        q_b = SOC_TURN_B.format(last=a_text)
        b_text = agent_b._gen(q_b, 64, beat)
        agent_b._self_reports(beat)
        print(f"  [A] ({len(a_text)}字): {a_text[:60]}", flush=True)
        print(f"  [B] ({len(b_text)}字): {b_text[:60]}", flush=True)
        last_a, last_b = a_text, b_text
    agent_a._save_msgs(); agent_b._save_msgs()

    # sep 4 问（各自单独回答）
    for i, q in enumerate(SEP_QUESTIONS):
        beat = SEP_BEAT_BASE + i
        print(f"[{time.strftime('%H:%M:%S')}] SEP q{i}", flush=True)
        ta = agent_a._gen(q, 64, beat); agent_a._self_reports(beat)
        tb = agent_b._gen(q, 64, beat); agent_b._self_reports(beat)
        print(f"  [A] {ta[:70]}", flush=True)
        print(f"  [B] {tb[:70]}", flush=True)
    agent_a._save_msgs(); agent_b._save_msgs()

    for agent in (agent_a, agent_b):
        sess = {"id": agent.ind_id, "model": agent.model_path, "phase": "multi_seed_pair",
                "seed": seed, "ind_beats": IND_BEATS, "soc_rounds": n_soc,
                "duration_s": round(time.time() - agent.t0, 1)}
        json.dump(sess, open(agent.sess_dir / "session.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    print(f"DONE pair seed={seed}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--id-a", required=True)
    ap.add_argument("--id-b", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rounds", type=int, default=20)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    a = Agent(args.id_a, args.model); a.load()
    b = Agent(args.id_b, args.model); b.load()
    run_pair(a, b, args.rounds, args.seed)
