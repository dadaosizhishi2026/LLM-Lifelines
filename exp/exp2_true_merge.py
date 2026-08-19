"""
云韶框架·AI特化·阶段2
真正的团合并KV生成：prefill→团合并KV→用合并KV做generate
不是mask屏蔽，是真的替换KV cache
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "exp2_results.json")

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

# ── 工具函数 ──
def build_cliques(attn_matrix, n_tokens, k=32, max_clique=8, min_clique=3):
    causal = np.tril(attn_matrix)
    sym = (causal + causal.T) / 2
    np.fill_diagonal(sym, 0)
    adj = np.zeros((n_tokens, n_tokens), dtype=np.int8)
    for i in range(n_tokens):
        top_idx = np.argsort(sym[i])[-k:]
        for j in top_idx:
            adj[i, j] = 1
            adj[j, i] = 1
    degrees = adj.sum(axis=1)
    order = np.argsort(-degrees)
    cliques = []
    covered = set()
    for start in order:
        if start in covered:
            continue
        clique = [start]
        candidates = set(np.where(adj[start])[0]) - {start}
        while len(clique) < max_clique and candidates:
            valid = [c for c in candidates if all(adj[c, m] for m in clique)]
            if not valid:
                break
            best = max(valid, key=lambda x: degrees[x])
            clique.append(best)
            candidates = candidates & set(np.where(adj[best])[0]) - set(clique)
        if len(clique) >= min_clique:
            cliques.append(sorted(clique))
            covered.update(clique)
    for i in range(n_tokens):
        if i not in covered:
            cliques.append([i])
            covered.add(i)
    return cliques

def adjust_cliques(cliques, target_n):
    cliques = [list(c) for c in cliques]
    while len(cliques) > target_n:
        sizes = [len(c) for c in cliques]
        min_idx = int(np.argmin(sizes))
        min_clique = cliques[min_idx]
        min_pos = np.mean(min_clique)
        best_dist = float('inf')
        best_idx = -1
        for ci, c in enumerate(cliques):
            if ci == min_idx:
                continue
            dist = abs(np.mean(c) - min_pos)
            if dist < best_dist:
                best_dist = dist
                best_idx = ci
        if best_idx >= 0:
            cliques[best_idx] = sorted(cliques[best_idx] + min_clique)
            cliques.pop(min_idx)
        else:
            break
    return cliques

def merge_kv(kv_tuple, cliques, weights):
    """
    团内KV加权平均
    kv_tuple: (key, value) each (batch, heads, seq, dim)
    返回: (merged_key, merged_value) each (batch, heads, n_cliques, dim)
    """
    key, value = kv_tuple
    batch, heads, seq, dim = key.shape
    n_merged = len(cliques)
    
    mk = torch.zeros(batch, heads, n_merged, dim, dtype=key.dtype)
    mv = torch.zeros(batch, heads, n_merged, dim, dtype=value.dtype)
    
    for ci, clique in enumerate(cliques):
        if len(clique) == 1:
            mk[:, :, ci, :] = key[:, :, clique[0], :]
            mv[:, :, ci, :] = value[:, :, clique[0], :]
        else:
            w = np.array([weights[t] for t in clique])
            w = w / w.sum()
            w_t = torch.tensor(w, dtype=key.dtype).view(1, 1, len(clique), 1)
            mk[:, :, ci, :] = (key[:, :, clique, :] * w_t).sum(dim=2)
            mv[:, :, ci, :] = (value[:, :, clique, :] * w_t).sum(dim=2)
    
    return mk, mv

def greedy_generate_from_kv(model, merged_kv_list, n_new_tokens, first_token_id=None):
    """
    用合并后的KV cache做greedy生成
    merged_kv_list: list of (key, value) per layer, each (1, heads, n_merged, dim)
    """
    generated = []
    kv = merged_kv_list
    n_kv = kv[0][0].shape[2]  # 合并后的序列长度
    
    # 第一个输入token
    if first_token_id is not None:
        input_id = torch.tensor([[first_token_id]])
    else:
        # 用最后一个合并位置的logit决定第一个token
        # 需要一个dummy forward...简化：用eos作为起始
        input_id = torch.tensor([[tokenizer.eos_token_id]])
    
    for step in range(n_new_tokens):
        # 构建position_ids（从n_kv开始）
        pos_ids = torch.arange(n_kv + step, n_kv + step + 1).unsqueeze(0)
        
        # 构建attention mask: 新token可以attend到所有KV位置
        # (1, 1, 1, n_kv+step) 全0（允许attend）
        attn_mask = torch.zeros(1, 1, 1, n_kv + step)
        
        # 构建DynamicCache
        cache = DynamicCache()
        for li in range(len(kv)):
            cache.update(kv[li][0], kv[li][1], li)
        
        with torch.no_grad():
            out = model(
                input_id,
                past_key_values=cache,
                position_ids=pos_ids,
                use_cache=True,
            )
        
        next_logits = out.logits[0, -1, :]
        next_token = next_logits.argmax().item()
        generated.append(next_token)
        
        # 更新KV（把新token的KV追加）
        new_kv = out.past_key_values
        if hasattr(new_kv, 'key_cache'):
            kv = [(new_kv.key_cache[i], new_kv.value_cache[i]) for i in range(len(new_kv.key_cache))]
        else:
            kv = [(t[0], t[1]) for t in new_kv]
        
        input_id = torch.tensor([[next_token]])
        
        # 停止条件
        if next_token == tokenizer.eos_token_id:
            break
    
    return generated

# ── 实验 ──
PROMPT = "The algebraic tension of a graph is defined as"
prompt_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
n_prompt = prompt_ids.shape[1]
log(f"Prompt: '{PROMPT}' ({n_prompt} tokens)")

GEN_STEPS = 30

# 1. 完整生成（基线）
log("\n── 完整KV生成（基线）──")
with torch.no_grad():
    gen_full = model.generate(prompt_ids, max_new_tokens=GEN_STEPS, do_sample=False)
text_full = tokenizer.decode(gen_full[0], skip_special_tokens=True)
log(f"  {text_full}")

# 2. Prefill拿KV + 注意力
log("\n── Prefill + 团合并 ──")
with torch.no_grad():
    out_pf = model(prompt_ids, output_attentions=True, use_cache=True)
    pf_attns = out_pf.attentions
    pf_kv = out_pf.past_key_values

if hasattr(pf_kv, 'key_cache'):
    kv_list = [(pf_kv.key_cache[i], pf_kv.value_cache[i]) for i in range(len(pf_kv.key_cache))]
else:
    kv_list = [(t[0], t[1]) for t in pf_kv]

# 计算重要性权重
attn_avg = pf_attns[0][0].numpy().mean(axis=0)
causal_attn = np.tril(attn_avg)
in_degree = causal_attn.sum(axis=0) + 1e-6

# 建团（多压缩比）
results = {
    "model": "Qwen2.5-0.5B",
    "prompt": PROMPT,
    "n_prompt_tokens": n_prompt,
    "full_generation": text_full,
    "experiments": []
}

for target_ratio in [0.75, 0.50]:
    n_target = max(int(n_prompt * target_ratio), 4)
    
    # 建团
    cliques = build_cliques(attn_avg, n_prompt, k=min(32, n_prompt-1), max_clique=6)
    cliques = adjust_cliques(cliques, n_target)
    
    actual_ratio = len(cliques) / n_prompt
    clique_sizes = [len(c) for c in cliques]
    log(f"\n── 压缩→{actual_ratio:.2f} ({len(cliques)}/{n_prompt} tokens) ──")
    log(f"  团: {len(cliques)}个, 大小: mean={np.mean(clique_sizes):.1f}, max={max(clique_sizes)}")
    
    # 逐层合并KV
    merged_kv = []
    for li in range(len(kv_list)):
        mk, mv = merge_kv(kv_list[li], cliques, in_degree)
        merged_kv.append((mk, mv))
    
    log(f"  合并KV shape: {merged_kv[0][0].shape}")
    
    # 用合并KV生成
    log(f"  生成中...")
    # 用prompt最后一个token的logit作为起始
    last_logit = out_pf.logits[0, -1, :]
    first_token = last_logit.argmax().item()
    
    tokens_gen = greedy_generate_from_kv(model, merged_kv, GEN_STEPS, first_token_id=first_token)
    text_gen = tokenizer.decode([first_token] + tokens_gen, skip_special_tokens=True)
    log(f"  合并: {PROMPT} {text_gen}")
    
    # 对比
    full_continuation = text_full[len(PROMPT):]
    merge_continuation = text_gen
    
    # token级重叠
    full_tokens = set(tokenizer.encode(full_continuation))
    merge_tokens = set(tokenizer.encode(merge_continuation))
    overlap = len(full_tokens & merge_tokens) / max(len(full_tokens), 1)
    
    exp = {
        "target_ratio": target_ratio,
        "actual_ratio": round(actual_ratio, 4),
        "n_merged_tokens": len(cliques),
        "mean_clique_size": round(float(np.mean(clique_sizes)), 2),
        "generation": f"{PROMPT} {text_gen}",
        "token_overlap_with_full": round(overlap, 4),
    }
    results["experiments"].append(exp)
    log(f"  token重叠率: {overlap:.1%}")

# 3. 逐层差异化压缩（T(G)诊断）
log("\n── 逐层差异化压缩 ──")
# 对每层独立建团，T(G)低的层激进压缩，高的层保守
layer_cliques = []
for li in range(len(pf_attns)):
    attn_li = pf_attns[li][0].numpy().mean(axis=0)
    # 快速T(G)估计
    causal_li = np.tril(attn_li)
    sym_li = (causal_li + causal_li.T) / 2
    np.fill_diagonal(sym_li, 0)
    k_li = min(32, n_prompt - 1)
    adj_li = np.zeros((n_prompt, n_prompt), dtype=np.int8)
    for i in range(n_prompt):
        top_idx = np.argsort(sym_li[i])[-k_li:]
        for j in top_idx:
            adj_li[i, j] = 1
            adj_li[j, i] = 1
    degrees_li = adj_li.sum(axis=1)
    # 贪心T(G)
    visited = np.zeros(n_prompt, dtype=bool)
    start = np.argmax(degrees_li)
    path = [start]
    visited[start] = True
    for _ in range(n_prompt - 1):
        curr = path[-1]
        neighbors = np.where(adj_li[curr] & ~visited)[0]
        if len(neighbors) == 0:
            unvisited = np.where(~visited)[0]
            if len(unvisited) == 0:
                break
            nxt = unvisited[np.argmax(degrees_li[unvisited])]
        else:
            nxt = neighbors[np.argmax(degrees_li[neighbors])]
        path.append(nxt)
        visited[nxt] = True
    d = sum(1 for i in range(len(path)-1) if adj_li[path[i], path[i+1]] == 0)
    t_g = d / max(n_prompt - 1, 1)
    
    # 根据T(G)决定压缩比
    if t_g <= 1/64:
        layer_target = int(n_prompt * 0.50)  # 激进
    elif t_g <= 1/8:
        layer_target = int(n_prompt * 0.75)  # 适度
    else:
        layer_target = n_prompt  # 保留全部
    
    if layer_target < n_prompt:
        cl = build_cliques(attn_li, n_prompt, k=k_li, max_clique=6)
        cl = adjust_cliques(cl, layer_target)
    else:
        cl = [[i] for i in range(n_prompt)]
    
    layer_cliques.append(cl)

# 逐层合并
merged_kv_adaptive = []
for li in range(len(kv_list)):
    mk, mv = merge_kv(kv_list[li], layer_cliques[li], in_degree)
    merged_kv_adaptive.append((mk, mv))

# 统计
layer_sizes = [len(cl) for cl in layer_cliques]
log(f"  逐层压缩: min={min(layer_sizes)}, max={max(layer_sizes)}, mean={np.mean(layer_sizes):.1f}")
log(f"  总KV节省: {1 - np.mean(layer_sizes)/n_prompt:.1%}")

# 生成
first_token_adaptive = out_pf.logits[0, -1, :].argmax().item()
tokens_adaptive = greedy_generate_from_kv(model, merged_kv_adaptive, GEN_STEPS, first_token_id=first_token_adaptive)
text_adaptive = tokenizer.decode([first_token_adaptive] + tokens_adaptive, skip_special_tokens=True)
log(f"  自适应: {PROMPT} {text_adaptive}")

results["adaptive_per_layer"] = {
    "layer_compressed_sizes": layer_sizes,
    "mean_compression": round(float(np.mean(layer_sizes) / n_prompt), 4),
    "generation": f"{PROMPT} {text_adaptive}",
}

# ── 保存 ──
with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
log(f"\n结果保存: {RESULTS}")
log("阶段2完成。")
