"""
云韶框架·AI特化·阶段2v2
修正：用89 token完整文本做prefill（不是10 token的短prompt）
从最后一个位置继续生成，测团合并KV的生成质量
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "exp2v2_results.json")

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

# 长文本prefill
TEXT = """The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures. By using these cliques as outer embryos to wrap original nodes, the original graph can be compressed into a smaller compressed graph. After solving the Hamiltonian path on the compressed graph and expanding back, the number of virtual edges remains stable at zero to two, with coverage above ninety-nine percent. This discovery means that the Hamiltonian path problem on dense graphs can be solved in polynomial time through dimensional compression."""

input_ids = tokenizer(TEXT, return_tensors="pt")["input_ids"]
n = input_ids.shape[1]
log(f"Prefill: {n} tokens")

GEN_STEPS = 40

# ── 工具 ──
def build_cliques(attn_matrix, n_tokens, k=32, max_clique=6, min_clique=3):
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

def generate_from_kv(model, kv_list, n_steps, first_token_id):
    """用KV cache做greedy生成"""
    generated = [first_token_id]
    input_id = torch.tensor([[first_token_id]])
    n_kv = kv_list[0][0].shape[2]
    
    for step in range(n_steps):
        cache = DynamicCache()
        for li in range(len(kv_list)):
            cache.update(kv_list[li][0], kv_list[li][1], li)
        
        pos_ids = torch.arange(n_kv + step, n_kv + step + 1).unsqueeze(0)
        
        with torch.no_grad():
            out = model(input_id, past_key_values=cache, position_ids=pos_ids, use_cache=True)
        
        next_token = out.logits[0, -1, :].argmax().item()
        generated.append(next_token)
        
        # 更新KV
        new_kv = out.past_key_values
        if hasattr(new_kv, 'key_cache'):
            kv_list = [(new_kv.key_cache[i], new_kv.value_cache[i]) for i in range(len(new_kv.key_cache))]
        else:
            kv_list = [(t[0], t[1]) for t in new_kv]
        
        input_id = torch.tensor([[next_token]])
        if next_token == tokenizer.eos_token_id:
            break
    
    return generated

# ── 1. 完整生成（基线） ──
log("\n── 完整KV生成（基线）──")
with torch.no_grad():
    gen_full = model.generate(input_ids, max_new_tokens=GEN_STEPS, do_sample=False)
text_full = tokenizer.decode(gen_full[0], skip_special_tokens=True)
continuation_full = text_full[len(TEXT):]
log(f"  续写: {continuation_full[:150]}")

# ── 2. Prefill + 团合并 ──
log("\n── Prefill + 团合并 ──")
with torch.no_grad():
    out_pf = model(input_ids, output_attentions=True, use_cache=True)
    pf_attns = out_pf.attentions
    pf_kv = out_pf.past_key_values
    last_logit = out_pf.logits[0, -1, :]

if hasattr(pf_kv, 'key_cache'):
    kv_list = [(pf_kv.key_cache[i], pf_kv.value_cache[i]) for i in range(len(pf_kv.key_cache))]
else:
    kv_list = [(t[0], t[1]) for t in pf_kv]

# 重要性权重
attn_avg = pf_attns[0][0].numpy().mean(axis=0)
causal_attn = np.tril(attn_avg)
in_degree = causal_attn.sum(axis=0) + 1e-6

# 第一个生成token（从完整prefill的logit）
first_token = last_logit.argmax().item()
log(f"  首token: '{tokenizer.decode([first_token])}'")

results = {
    "model": "Qwen2.5-0.5B",
    "n_prefill_tokens": n,
    "full_continuation": continuation_full[:200],
    "experiments": []
}

for target_ratio in [0.90, 0.75, 0.50]:
    n_target = max(int(n * target_ratio), 10)
    
    cliques = build_cliques(attn_avg, n, k=min(32, n-1), max_clique=6)
    cliques = adjust_cliques(cliques, n_target)
    
    actual_ratio = len(cliques) / n
    clique_sizes = [len(c) for c in cliques]
    log(f"\n── 压缩→{actual_ratio:.2f} ({len(cliques)}/{n} tokens, 均大小{np.mean(clique_sizes):.1f}) ──")
    
    # 逐层合并
    merged_kv = []
    for li in range(len(kv_list)):
        mk, mv = merge_kv(kv_list[li], cliques, in_degree)
        merged_kv.append((mk, mv))
    
    # 生成
    tokens_gen = generate_from_kv(model, merged_kv, GEN_STEPS, first_token)
    text_gen = tokenizer.decode(tokens_gen, skip_special_tokens=True)
    log(f"  续写: {text_gen[:150]}")
    
    # token重叠
    full_toks = set(tokenizer.encode(continuation_full))
    gen_toks = set(tokens_gen)
    overlap = len(full_toks & gen_toks) / max(len(full_toks), 1)
    
    results["experiments"].append({
        "target_ratio": target_ratio,
        "actual_ratio": round(actual_ratio, 4),
        "n_merged": len(cliques),
        "mean_clique_size": round(float(np.mean(clique_sizes)), 2),
        "continuation": text_gen[:200],
        "token_overlap": round(overlap, 4),
    })
    log(f"  token重叠: {overlap:.1%}")

# ── 3. 逐层自适应 ──
log("\n── 逐层自适应压缩 ──")
layer_cliques = []
for li in range(len(pf_attns)):
    attn_li = pf_attns[li][0].numpy().mean(axis=0)
    causal_li = np.tril(attn_li)
    sym_li = (causal_li + causal_li.T) / 2
    np.fill_diagonal(sym_li, 0)
    k_li = min(32, n - 1)
    adj_li = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        top_idx = np.argsort(sym_li[i])[-k_li:]
        for j in top_idx:
            adj_li[i, j] = 1
            adj_li[j, i] = 1
    degrees_li = adj_li.sum(axis=1)
    visited = np.zeros(n, dtype=bool)
    start = int(np.argmax(degrees_li))
    path = [start]
    visited[start] = True
    for _ in range(n - 1):
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
    t_g = d / max(n - 1, 1)
    
    if t_g <= 1/64:
        layer_target = int(n * 0.50)
    elif t_g <= 1/8:
        layer_target = int(n * 0.75)
    else:
        layer_target = n
    
    if layer_target < n:
        cl = build_cliques(attn_li, n, k=k_li, max_clique=6)
        cl = adjust_cliques(cl, layer_target)
    else:
        cl = [[i] for i in range(n)]
    layer_cliques.append(cl)

merged_kv_adaptive = []
for li in range(len(kv_list)):
    mk, mv = merge_kv(kv_list[li], layer_cliques[li], in_degree)
    merged_kv_adaptive.append((mk, mv))

layer_sizes = [len(cl) for cl in layer_cliques]
log(f"  逐层: min={min(layer_sizes)}, max={max(layer_sizes)}, mean={np.mean(layer_sizes):.1f}, 节省={1-np.mean(layer_sizes)/n:.1%}")

tokens_adaptive = generate_from_kv(model, merged_kv_adaptive, GEN_STEPS, first_token)
text_adaptive = tokenizer.decode(tokens_adaptive, skip_special_tokens=True)
log(f"  续写: {text_adaptive[:150]}")

results["adaptive"] = {
    "layer_sizes": layer_sizes,
    "mean_compression": round(float(np.mean(layer_sizes) / n), 4),
    "continuation": text_adaptive[:200],
}

# ── 保存 ──
with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
log(f"\n结果保存: {RESULTS}")
log("阶段2v2完成。")
