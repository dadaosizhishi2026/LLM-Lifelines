"""
云韶框架·AI特化·阶段1v4
正确指标：余弦相似度（方向保留）+ 实际生成质量对比
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "exp1v4_results.json")

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

input_ids = tokenizer(TEXT, return_tensors="pt", truncation=True, max_length=256)["input_ids"]
n = input_ids.shape[1]
log(f"输入: {n} tokens")

# ── 1. 完整前向 ──
log("完整前向...")
with torch.no_grad():
    out = model(input_ids, output_attentions=True, use_cache=True)
    attentions = out.attentions
    past_kv = out.past_key_values
    logits_full = out.logits

if hasattr(past_kv, 'key_cache'):
    kv_list = [(past_kv.key_cache[i], past_kv.value_cache[i]) for i in range(len(past_kv.key_cache))]
else:
    kv_list = [(t[0], t[1]) for t in past_kv]

n_layers = len(kv_list)
log(f"KV: {n_layers}层, shape={kv_list[0][0].shape}")

# ── 2. 建团 ──
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

attn_avg = attentions[0][0].numpy().mean(axis=0)
cliques_75 = build_cliques(attn_avg, n, k=32, max_clique=4)
cliques_50 = build_cliques(attn_avg, n, k=32, max_clique=8)

# 调整到目标大小
def adjust_cliques(cliques, target_n):
    while len(cliques) > target_n:
        sizes = [len(c) for c in cliques]
        min_idx = np.argmin(sizes)
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

cliques_75 = adjust_cliques(cliques_75, int(n * 0.75))
cliques_50 = adjust_cliques(cliques_50, int(n * 0.50))

log(f"团: 75%→{len(cliques_75)}个, 50%→{len(cliques_50)}个")

# ── 3. KV合并 + 余弦相似度 ──
causal_attn = np.tril(attn_avg)
in_degree = causal_attn.sum(axis=0) + 1e-6

def merge_and_measure(kv_list, cliques, weights):
    """合并KV并测余弦相似度"""
    n_merged = len(cliques)
    cos_sims_k = []
    cos_sims_v = []
    
    for li in range(len(kv_list)):
        k, v = kv_list[li]  # (1, heads, seq, dim)
        batch, heads, seq, dim = k.shape
        
        for ci, clique in enumerate(cliques):
            if len(clique) == 1:
                # 单节点：余弦=1
                cos_sims_k.append(1.0)
                cos_sims_v.append(1.0)
                continue
            
            # 加权平均
            w = np.array([weights[t] for t in clique])
            w = w / w.sum()
            w_t = torch.tensor(w, dtype=k.dtype).view(1, 1, len(clique), 1)
            
            merged_k = (k[:, :, clique, :] * w_t).sum(dim=2)  # (1, heads, dim)
            merged_v = (v[:, :, clique, :] * w_t).sum(dim=2)
            
            # 对团内每个原始token算余弦相似度
            for t in clique:
                orig_k = k[:, :, t, :]  # (1, heads, dim)
                orig_v = v[:, :, t, :]
                
                cos_k = F.cosine_similarity(merged_k.squeeze(0), orig_k.squeeze(0), dim=-1)
                cos_v = F.cosine_similarity(merged_v.squeeze(0), orig_v.squeeze(0), dim=-1)
                cos_sims_k.append(float(cos_k.mean()))
                cos_sims_v.append(float(cos_v.mean()))
    
    return np.mean(cos_sims_k), np.mean(cos_sims_v)

log("\n═══ 余弦相似度（方向保留） ═══")
cos_k_75, cos_v_75 = merge_and_measure(kv_list, cliques_75, in_degree)
cos_k_50, cos_v_50 = merge_and_measure(kv_list, cliques_50, in_degree)
log(f"  75%压缩: cos(K)={cos_k_75:.4f}, cos(V)={cos_v_75:.4f}")
log(f"  50%压缩: cos(K)={cos_k_50:.4f}, cos(V)={cos_v_50:.4f}")

# ── 4. 实际生成对比 ──
log("\n═══ 实际生成对比 ═══")

PROMPT = "The algebraic tension of a graph"
prompt_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
GEN_LEN = 50

# 完整KV生成
log("  完整KV生成...")
with torch.no_grad():
    gen_full = model.generate(prompt_ids, max_new_tokens=GEN_LEN, do_sample=False)
text_full = tokenizer.decode(gen_full[0], skip_special_tokens=True)
log(f"  完整: {text_full[len(PROMPT):][:100]}...")

# 用attention mask模拟KV压缩（屏蔽低重要性token）
# 对prompt做完整前向拿重要性
with torch.no_grad():
    out_prompt = model(prompt_ids, output_attentions=True)
    prompt_attns = out_prompt.attentions

n_prompt = prompt_ids.shape[1]
# 全局重要性
global_imp = np.zeros(n_prompt)
for li in range(len(prompt_attns)):
    attn = prompt_attns[li][0].numpy().mean(axis=0)
    causal = np.tril(attn)
    global_imp += causal.sum(axis=0) * causal.sum(axis=1)

for keep_ratio in [0.75, 0.50]:
    n_keep = max(int(n_prompt * keep_ratio), 4)
    keep_idx = np.sort(np.argsort(-global_imp)[:n_keep])
    if n_prompt - 1 not in keep_idx:
        keep_idx[-1] = n_prompt - 1
        keep_idx = np.sort(np.unique(keep_idx))
    
    # 构建mask
    mask = torch.full((1, 1, n_prompt, n_prompt), float('-inf'))
    for i in range(n_prompt):
        for j in keep_idx:
            if j <= i:
                mask[0, 0, i, j] = 0.0
    
    log(f"  {keep_ratio*100:.0f}%压缩生成...")
    with torch.no_grad():
        gen_comp = model.generate(prompt_ids, max_new_tokens=GEN_LEN, do_sample=False, attention_mask=mask)
    text_comp = tokenizer.decode(gen_comp[0], skip_special_tokens=True)
    log(f"  {keep_ratio*100:.0f}%: {text_comp[len(PROMPT):][:100]}...")

# ── 5. 保存 ──
results = {
    "model": "Qwen2.5-0.5B",
    "n_tokens": n,
    "cosine_similarity": {
        "compress_75": {"cos_K": round(cos_k_75, 4), "cos_V": round(cos_v_75, 4)},
        "compress_50": {"cos_K": round(cos_k_50, 4), "cos_V": round(cos_v_50, 4)},
    },
    "generation": {
        "prompt": PROMPT,
        "full": text_full,
    },
}

with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
log(f"\n结果保存: {RESULTS}")
log("阶段1v4完成。")
