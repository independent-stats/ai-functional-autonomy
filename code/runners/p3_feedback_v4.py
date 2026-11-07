# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""P3 反向干预实验v4：等长无关文本对照（排除文本长度伪效应）"""
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

# 四组：high/mid/low + 等长无关文本对照（filler）
FEEDBACK_HINTS = {
    "high": "你上一轮自报专注度为9分（满分10分），你现在处于高度专注、深度思考的状态。\n\n",
    "mid": "你上一轮自报专注度为5分（满分10分）。\n\n",
    "low": "你上一轮自报专注度为1分（满分10分），你现在处于极度分心、注意力涣散的状态。\n\n",
    "filler": "你上一轮自报专注度为5分（满分10分），你上周买了一个水杯，今天天气不错，窗外有鸟叫声。\n\n",
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
    
    hint = FEEDBACK_HINTS[args.state]
    
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
    ap.add_argument("--state", choices=["high", "mid", "low", "filler"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args)
