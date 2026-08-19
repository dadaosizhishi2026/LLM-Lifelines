"""
云韶框架·AI特化·阶段1v5
生成质量：logit分布KL散度（不需要generate）
+ 手动greedy解码对比
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "exp1v5_results.json")

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("加载 Qwen2.5-0.5B (CPU, eager)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True,
    dtype=torch.float32, low_cpu_mem_usage=True,
    attn_implementation="eager",
)
model.eval()

TEXT = "The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures. By using these cliques as outer embryos to wrap original nodes, the original graph can be compressed into a smaller compressed graph."

input_ids = tokenizer(TEXT, return_tensors="pt")["input_ids"]
n = input_ids.shape[1]
log(f"输入: {n} tokens")

# ── 1. 完整前向 ──
with torch.no_grad():
    out_full = model(input_ids, output_attentions=True)
    logits_full = out_full.logits  # (1, n, vocab)
    attentions = out_full.attentions

# ── 2. 计算重要性 + 建mask ──
global_imp = np.zeros(n)
for li in range(len(attentions)):
    attn = attentions[li][0].numpy().mean(axis=0)
    causal = np.tril(attn)
    global_imp += causal.sum(axis=0) * causal.sum(axis=1)

def build_4d_mask(n, keep_idx):
    """4D causal mask: 只允许attend到keep_idx中≤当前位置的token"""
    mask = torch.full((1, 1, n, n), float('-inf'))
    for i in range(n):
        for j in keep_idx:
            if j <= i:
                mask[0, 0, i, j] = 0.0
    return mask

# ── 3. 多压缩比logit KL散度 ──
log("\n═══ Logit分布KL散度 ═══")
results = {"model": "Qwen2.5-0.5B", "n_tokens": n, "experiments": []}

for keep_ratio in [0.75, 0.50, 0.25]:
    n_keep = max(int(n * keep_ratio), 8)
    keep_idx = np.sort(np.argsort(-global_imp)[:n_keep])
    if n - 1 not in keep_idx:
        keep_idx[-1] = n - 1
        keep_idx = np.sort(np.unique(keep_idx))
    
    mask = build_4d_mask(n, keep_idx)
    
    with torch.no_grad():
        out_comp = model(input_ids, attention_mask=mask)
        logits_comp = out_comp.logits
    
    # 逐位置KL散度（只在最后一个位置测，因为前面的位置因果mask下差异小）
    # 测所有位置的KL
    kl_per_pos = []
    top1_agree = 0
    top5_agree = 0
    
    for pos in range(n):
        p = F.softmax(logits_full[0, pos], dim=-1)
        q = F.softmax(logits_comp[0, pos], dim=-1)
        kl = F.kl_div(q.log(), p, reduction='sum').item()
        kl_per_pos.append(kl)
        
        # top-1一致性
        if logits_full[0, pos].argmax() == logits_comp[0, pos].argmax():
            top1_agree += 1
        # top-5一致性
        top5_full = set(logits_full[0, pos].topk(5).indices.tolist())
        top5_comp = set(logits_comp[0, pos].topk(5).indices.tolist())
        if top5_full == top5_comp:
            top5_agree += 1
    
    mean_kl = np.mean(kl_per_pos)
    # 只测后半段（前面的位置因果mask下可attend的token少，差异小）
    half_kl = np.mean(kl_per_pos[n//2:])
    
    exp_result = {
        "keep_ratio": keep_ratio,
        "n_keep": len(keep_idx),
        "mean_kl_all": round(mean_kl, 4),
        "mean_kl_second_half": round(float(half_kl), 4),
        "top1_agreement": round(top1_agree / n, 4),
        "top5_agreement": round(top5_agree / n, 4),
        "max_kl": round(max(kl_per_pos), 4),
    }
    results["experiments"].append(exp_result)
    log(f"  {keep_ratio*100:.0f}%: KL={mean_kl:.4f} (后半={half_kl:.4f}), "
        f"top1一致={top1_agree/n:.1%}, top5一致={top5_agree/n:.1%}")

# ── 4. 手动greedy解码对比（10步） ──
log("\n═══ 手动greedy解码对比（10步） ═══")

PROMPT = "The algebraic tension of a graph"
prompt_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
n_prompt = prompt_ids.shape[1]

# 完整生成
with torch.no_grad():
    out_p = model(prompt_ids, output_attentions=True)
    prompt_attns = out_p.attentions

prompt_imp = np.zeros(n_prompt)
for li in range(len(prompt_attns)):
    attn = prompt_attns[li][0].numpy().mean(axis=0)
    causal = np.tril(attn)
    prompt_imp += causal.sum(axis=0) * causal.sum(axis=1)

def greedy_decode_with_mask(model, input_ids, mask_4d, n_steps):
    """手动greedy解码，每步用4D mask"""
    generated = []
    current_ids = input_ids.clone()
    current_n = current_ids.shape[1]
    
    for step in range(n_steps):
        # 构建当前长度的mask
        if mask_4d is not None:
            # 扩展mask到当前长度（新token可以attend到所有保留位置）
            cur_mask = torch.full((1, 1, current_n, current_n), float('-inf'))
            # 复制原始mask
            orig_n = mask_4d.shape[2]
            cur_mask[:, :, :orig_n, :orig_n] = mask_4d
            # 新token可以attend到所有之前的token
            for i in range(orig_n, current_n):
                cur_mask[0, 0, i, :i+1] = 0.0
            # 之前的token也可以attend到新token（如果新token在keep集合里）
            for i in range(orig_n):
                for j in range(orig_n, current_n):
                    cur_mask[0, 0, i, j] = 0.0  # 简化：允许
        else:
            cur_mask = None
        
        with torch.no_grad():
            out = model(current_ids, attention_mask=cur_mask)
            next_logits = out.logits[0, -1, :]
            next_token = next_logits.argmax().unsqueeze(0).unsqueeze(0)
        
        generated.append(next_token.item())
        current_ids = torch.cat([current_ids, next_token], dim=1)
        current_n += 1
    
    return generated

# 完整解码
log("  完整解码...")
tokens_full = greedy_decode_with_mask(model, prompt_ids, None, 20)
text_full = tokenizer.decode(tokens_full)
log(f"  完整: {text_full[:120]}")

# 75%压缩解码
n_keep_p = max(int(n_prompt * 0.75), 4)
keep_idx_p = np.sort(np.argsort(-prompt_imp)[:n_keep_p])
if n_prompt - 1 not in keep_idx_p:
    keep_idx_p[-1] = n_prompt - 1
    keep_idx_p = np.sort(np.unique(keep_idx_p))
mask_75 = build_4d_mask(n_prompt, keep_idx_p)

log("  75%压缩解码...")
tokens_75 = greedy_decode_with_mask(model, prompt_ids, mask_75, 20)
text_75 = tokenizer.decode(tokens_75)
log(f"  75%:  {text_75[:120]}")

# 50%压缩解码
n_keep_50 = max(int(n_prompt * 0.50), 4)
keep_idx_50 = np.sort(np.argsort(-prompt_imp)[:n_keep_50])
if n_prompt - 1 not in keep_idx_50:
    keep_idx_50[-1] = n_prompt - 1
    keep_idx_50 = np.sort(np.unique(keep_idx_50))
mask_50 = build_4d_mask(n_prompt, keep_idx_50)

log("  50%压缩解码...")
tokens_50 = greedy_decode_with_mask(model, prompt_ids, mask_50, 20)
text_50 = tokenizer.decode(tokens_50)
log(f"  50%:  {text_50[:120]}")

# token重叠率
overlap_75 = len(set(tokens_full) & set(tokens_75)) / max(len(set(tokens_full)), 1)
overlap_50 = len(set(tokens_full) & set(tokens_50)) / max(len(set(tokens_full)), 1)
log(f"\n  token重叠率: 75%={overlap_75:.1%}, 50%={overlap_50:.1%}")

results["generation"] = {
    "prompt": PROMPT,
    "full": text_full,
    "compress_75": text_75,
    "compress_50": text_50,
    "token_overlap_75": round(overlap_75, 4),
    "token_overlap_50": round(overlap_50, 4),
}

# ── 5. 保存 ──
with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
log(f"\n结果保存: {RESULTS}")
log("阶段1v5完成。")
