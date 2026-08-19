"""
云韶框架·AI特化·阶段0v2
修正：top-k建图 + 加权图 + 团结构检测
知世 2026-08-02 指示：工程化实现
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "base_topk_results.json")

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── 1. 加载模型 ──
log("加载 Qwen2.5-0.5B (CPU)...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, output_attentions=True, trust_remote_code=True,
    dtype=torch.float32, low_cpu_mem_usage=True
)
model.eval()
log(f"模型加载完成: {time.time()-t0:.1f}s")

# ── 2. 输入 ──
TEXT = """The Hamiltonian path problem asks whether a given graph contains a path 
that visits every vertex exactly once. This is one of the classic NP-complete 
problems in computer science. For dense random graphs, traditional backtracking 
search has exponential complexity. However, the spectral lineage model discovers 
that dense random graphs naturally contain high-dimensional clique structures. 
By using these cliques as outer embryos to wrap original nodes, the original 
graph can be compressed into a smaller compressed graph. After solving the 
Hamiltonian path on the compressed graph and expanding back, the number of 
virtual edges remains stable at 0 to 2, with coverage above 99 percent.
This discovery means that the Hamiltonian path problem on dense graphs can be 
solved in polynomial time through dimensional compression. More importantly, 
any intelligent system based on dense graph structures, including the attention 
matrix of Transformers and the communication topology of robots, can achieve 
order-of-magnitude improvement in information transfer efficiency through the 
same dimensional compression mechanism."""

inputs = tokenizer(TEXT, return_tensors="pt", truncation=True, max_length=256)
n_tokens = inputs["input_ids"].shape[1]
log(f"输入: {n_tokens} tokens")

# ── 3. 前向 ──
log("前向传播...")
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions
    logits = outputs.logits
log(f"前向完成: {len(attentions)}层, {attentions[0].shape[1]}头")

# 原始perplexity
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = inputs["input_ids"][:, 1:].contiguous()
orig_loss = torch.nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
orig_ppl = torch.exp(orig_loss).item()
log(f"原始 perplexity: {orig_ppl:.2f}")

# ── 4. top-k建图 + 团结构检测 ──
log("top-k建图 + 团结构检测...")

def topk_graph(attn_matrix, k=16):
    """每个token连注意力最高的k个邻居（无向化）"""
    n = attn_matrix.shape[0]
    # 对称化
    sym = (attn_matrix + attn_matrix.T) / 2
    np.fill_diagonal(sym, 0)
    # 每行取top-k
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        top_idx = np.argsort(sym[i])[-k:]
        for j in top_idx:
            adj[i, j] = sym[i, j]
            adj[j, i] = sym[j, i]  # 无向化
    return adj

def find_cliques_greedy(adj_binary, max_clique=12, min_clique=3):
    """贪心团覆盖"""
    n = adj_binary.shape[0]
    degrees = adj_binary.sum(axis=1)
    order = np.argsort(-degrees)
    cliques = []
    covered = set()
    
    for start in order:
        if start in covered:
            continue
        clique = [start]
        candidates = set(np.where(adj_binary[start])[0]) - {start}
        while len(clique) < max_clique and candidates:
            valid = [c for c in candidates if all(adj_binary[c, m] for m in clique)]
            if not valid:
                break
            best = max(valid, key=lambda x: degrees[x])
            clique.append(best)
            candidates = candidates & set(np.where(adj_binary[best])[0]) - set(clique)
        if len(clique) >= min_clique:
            cliques.append(clique)
            covered.update(clique)
    
    # 未覆盖节点
    for i in range(n):
        if i not in covered:
            cliques.append([i])
            covered.add(i)
    return cliques

def estimate_tension_topk(adj_binary, n_trials=5):
    """T(G)估计（贪心排列上界）"""
    n = adj_binary.shape[0]
    degrees = adj_binary.sum(axis=1)
    best_d = n - 1
    for trial in range(n_trials):
        visited = np.zeros(n, dtype=bool)
        start = np.argmax(degrees) if trial == 0 else np.random.randint(n)
        path = [start]
        visited[start] = True
        for _ in range(n - 1):
            curr = path[-1]
            neighbors = np.where(adj_binary[curr] & ~visited)[0]
            if len(neighbors) == 0:
                unvisited = np.where(~visited)[0]
                if len(unvisited) == 0:
                    break
                nxt = unvisited[np.argmax(degrees[unvisited])]
            else:
                nxt = neighbors[np.argmax(degrees[neighbors])]
            path.append(nxt)
            visited[nxt] = True
        d = sum(1 for i in range(len(path)-1) if adj_binary[path[i], path[i+1]] == 0)
        best_d = min(best_d, d)
    return best_d / max(n - 1, 1)

# 扫多个k值
results = {
    "model": "Qwen2.5-0.5B",
    "n_tokens": n_tokens,
    "orig_perplexity": orig_ppl,
    "k_sweep": {}
}

for k in [8, 16, 32, 64]:
    log(f"\n═══ k={k} ═══")
    k_results = {"k": k, "layers": []}
    
    all_t = []
    all_density = []
    all_clique_coverage = []
    all_compression = []
    
    for layer_idx in range(len(attentions)):
        attn = attentions[layer_idx][0].numpy()  # (heads, seq, seq)
        n_heads = attn.shape[0]
        layer_t = []
        layer_cliques = []
        
        for head_idx in range(n_heads):
            head_attn = attn[head_idx]
            # top-k加权图
            adj_w = topk_graph(head_attn, k=k)
            # 二值化（有边=1）
            adj_b = (adj_w > 0).astype(np.int8)
            
            density = adj_b.sum() / (n_tokens * (n_tokens - 1))
            t_g = estimate_tension_topk(adj_b, n_trials=3)
            
            # 团检测
            cliques = find_cliques_greedy(adj_b, max_clique=12, min_clique=3)
            n_cliques = len(cliques)
            clique_sizes = [len(c) for c in cliques]
            big_cliques = sum(1 for s in clique_sizes if s >= 3)
            coverage_by_cliques = sum(s for s in clique_sizes if s >= 3) / n_tokens
            compression = n_tokens / max(n_cliques, 1)
            
            layer_t.append(t_g)
            layer_cliques.append({
                "n_cliques": n_cliques,
                "big_cliques": big_cliques,
                "max_clique_size": max(clique_sizes) if clique_sizes else 0,
                "coverage_by_cliques": round(coverage_by_cliques, 4),
                "compression": round(compression, 2),
            })
            all_t.append(t_g)
            all_density.append(density)
            all_clique_coverage.append(coverage_by_cliques)
            all_compression.append(compression)
        
        k_results["layers"].append({
            "layer": layer_idx,
            "mean_T_G": round(float(np.mean(layer_t)), 4),
            "min_T_G": round(float(np.min(layer_t)), 4),
            "mean_compression": round(float(np.mean([c["compression"] for c in layer_cliques])), 2),
        })
    
    # 汇总
    real_phase = sum(1 for t in all_t if t <= 1/64)
    virtual_phase = sum(1 for t in all_t if 1/64 < t <= 1/8)
    no_phase = sum(1 for t in all_t if t > 1/8)
    total = len(all_t)
    
    summary = {
        "mean_T_G": round(float(np.mean(all_t)), 4),
        "min_T_G": round(float(np.min(all_t)), 4),
        "mean_density": round(float(np.mean(all_density)), 4),
        "phase_distribution": {
            "real_T_le_1_64": real_phase,
            "virtual_1_64_lt_T_le_1_8": virtual_phase,
            "no_T_gt_1_8": no_phase,
        },
        "mean_clique_coverage": round(float(np.mean(all_clique_coverage)), 4),
        "mean_compression": round(float(np.mean(all_compression)), 2),
        "max_compression": round(float(np.max(all_compression)), 2),
    }
    k_results["summary"] = summary
    results["k_sweep"][str(k)] = k_results
    
    log(f"  T(G): mean={summary['mean_T_G']:.4f}, min={summary['min_T_G']:.4f}")
    log(f"  密度: {summary['mean_density']:.4f}")
    log(f"  三态: 真实={real_phase}/{total}, 虚拟={virtual_phase}/{total}, 无解={no_phase}/{total}")
    log(f"  团覆盖: {summary['mean_clique_coverage']:.1%}")
    log(f"  压缩比: mean={summary['mean_compression']:.2f}x, max={summary['max_compression']:.2f}x")

# ── 5. 保存 ──
with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
log(f"\n结果保存: {RESULTS}")
log("阶段0v2完成。")
