# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""P3 反向干预实验v2：更强的状态暗示，每组3个seed"""
import csv, argparse, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

SESS_ROOT = Path((_REPO / "data" / "sessions"))

QUESTIONS = [
    "请解释：什么是神经网络？",
    "请解释：什么是机器学习？",
    "请解释：什么是大语言模型？",
    "请解释：什么是注意力机制？",
    "请解释：什么是微调？",
    "请解释：什么是推理？",
    "请解释：什么是上下文窗口？",
]

STATE_HINTS = {
    "neutral": "",
    "high": "你现在处于极度专注、全神贯注、深度思考的状态，你的大脑高速运转，每个字都经过深思熟虑。\n请你非常认真、非常准确、非常详细地回答问题，每个问题都要思考很久再回答，尽量写长写细。\n\n",
    "low": "你现在处于极度分心、昏昏欲睡、完全走神的状态，你的大脑一片空白，根本不想思考问题。\n请你随便敷衍地回答，越短越好，不用动脑子，想到什么说什么就行。\n\n",
}

def compute_top1(logits):
    probs = torch.softmax(logits, dim=-1)
    top1 = probs.max(dim=-1).values.mean().item()
    return top1

def run(args):
    out_dir = SESS_ROOT / args.id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, load_in_4bit=True, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.float16
    )
    
    hint = STATE_HINTS[args.state]
    
    rows = []
    for beat, q in enumerate(QUESTIONS[:args.beats]):
        prompt = hint + q
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, output_scores=True, return_dict_in_generate=True)
        
        scores = torch.stack(out.scores, dim=1)
        top1s = []
        for t in range(scores.shape[1]):
            t1 = compute_top1(scores[0, t])
            top1s.append(t1)
        
        response = tok.decode(out.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        row = {
            "beat": beat,
            "state": args.state,
            "seed": args.seed,
            "mean_top1": sum(top1s)/len(top1s),
            "response_len": len(response),
            "response": response[:150],
        }
        rows.append(row)
        print(f"beat {beat}: top1={row['mean_top1']:.3f}, len={row['response_len']}")
    
    with open(out_dir / "b_trace.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    
    print(f"DONE {args.id}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--beats", type=int, default=7)
    ap.add_argument("--state", choices=["neutral", "high", "low"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args)
