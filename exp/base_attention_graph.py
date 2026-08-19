"""
云韶框架·AI特化·阶段0验证
注意力矩阵 → 图 → 代数张力T(G)估计 → 谱系压缩 → perplexity变化
知世 2026-08-02 指示：工程化实现
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "base_results.json")

# 本地模型路径（网络被禁，用HF cache里的Qwen2.5-0.5B）
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── 1. 加载模型 ──
log(f"加载 Qwen2.5-0.5B (CPU) from {MODEL_PATH}...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, output_attentions=True, trust_remote_code=True,
    torch_dtype=torch.float32, low_cpu_mem_usage=True
)
model.eval()
log(f"模型加载完成: {time.time()-t0:.1f}s, 参数量: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

# ── 2. 准备输入 ──
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

# 限制序列长度以节省内存（0.5B模型24层14头，256 token够用）
MAX_SEQ = min(n_tokens, 256)
if n_tokens > MAX_SEQ:
    inputs = {k: v[:, :MAX_SEQ] for k, v in inputs.items()}
    n_tokens = MAX_SEQ
    log(f"截断到 {n_tokens} tokens（节省内存）")

# ── 3. 前向传播，提取注意力 ──
log("前向传播...")
t0 = time.time()
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)
    logits = outputs.logits
log(f"前向完成: {time.time()-t0:.1f}s, {len(attentions)}层, {attentions[0].shape[1]}头")

# ── 4. 原始perplexity ──
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = inputs["input_ids"][:, 1:].contiguous()
loss_fn = torch.nn.CrossEntropyLoss()
orig_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
orig_ppl = torch.exp(orig_loss).item()
log(f"原始 perplexity: {orig_ppl:.2f}")

# ── 5. 对每层每头建图 + 估计T(G) ──
log("建图 + 代数张力估计...")

def estimate_tension(attn_matrix, theta=0.1, n_trials=5):
    """
    估计代数张力 T(G) 的上界。
    T(G) = min_π D(π)/(n-1)，D(π)=排列中相邻节点在原图不相连的边数。
    用贪心排列（度最大优先）给上界。
    attn_matrix: (seq, seq) numpy array
    theta: 注意力权重阈值，>theta的算有边
    """
    n = attn_matrix.shape[0]
    # 对称化（无向图）
    adj = ((attn_matrix + attn_matrix.T) / 2 > theta).astype(np.int8)
    np.fill_diagonal(adj, 0)
    degrees = adj.sum(axis=1)
    
    best_d = n - 1  # 最坏情况
    for trial in range(n_trials):
        # 贪心：从度最大的节点开始，每步选未访问邻居中度最大的
        visited = np.zeros(n, dtype=bool)
        if trial == 0:
            start = np.argmax(degrees)
        else:
            start = np.random.randint(n)
        path = [start]
        visited[start] = True
        for _ in range(n - 1):
            curr = path[-1]
            neighbors = np.where(adj[curr] & ~visited)[0]
            if len(neighbors) == 0:
                # 无邻居，选未访问中度最大的（制造虚拟边）
                unvisited = np.where(~visited)[0]
                if len(unvisited) == 0:
                    break
                nxt = unvisited[np.argmax(degrees[unvisited])]
            else:
                nxt = neighbors[np.argmax(degrees[neighbors])]
            path.append(nxt)
            visited[nxt] = True
        # 计算D(π)
        d = sum(1 for i in range(len(path)-1) if adj[path[i], path[i+1]] == 0)
        best_d = min(best_d, d)
    
    t_g = best_d / max(n - 1, 1)
    return t_g, adj, degrees

results = {
    "model": "Qwen2.5-0.5B",
    "n_tokens": n_tokens,
    "orig_perplexity": orig_ppl,
    "layers": []
}

layer_data = []
for layer_idx in range(len(attentions)):
    attn = attentions[layer_idx][0].numpy()  # (heads, seq, seq)
    n_heads = attn.shape[0]
    layer_info = {"layer": layer_idx, "heads": []}
    
    for head_idx in range(n_heads):
        head_attn = attn[head_idx]  # (seq, seq)
        t_g, adj, degrees = estimate_tension(head_attn, theta=0.1, n_trials=3)
        edge_density = adj.sum() / (n_tokens * (n_tokens - 1)) if n_tokens > 1 else 0
        head_info = {
            "head": head_idx,
            "T_G": round(t_g, 6),
            "edge_density": round(float(edge_density), 6),
            "mean_degree": round(float(degrees.mean()), 2),
        }
        layer_info["heads"].append(head_info)
        layer_data.append((layer_idx, head_idx, t_g, edge_density, adj))
    
    results["layers"].append(layer_info)
    
    # 每层打印摘要
    t_vals = [h["T_G"] for h in layer_info["heads"]]
    log(f"  L{layer_idx:2d}: T(G) range [{min(t_vals):.4f}, {max(t_vals):.4f}], "
        f"mean={np.mean(t_vals):.4f}")

# ── 6. 三态分类 ──
log("三态分类...")
real_phase = sum(1 for _, _, t, _, _ in layer_data if t <= 1/64)
virtual_phase = sum(1 for _, _, t, _, _ in layer_data if 1/64 < t <= 1/8)
no_phase = sum(1 for _, _, t, _, _ in layer_data if t > 1/8)
total = len(layer_data)
log(f"  真实解相 (T≤1/64): {real_phase}/{total} ({100*real_phase/total:.1f}%)")
log(f"  虚拟解相 (1/64<T≤1/8): {virtual_phase}/{total} ({100*virtual_phase/total:.1f}%)")
log(f"  无解相 (T>1/8): {no_phase}/{total} ({100*no_phase/total:.1f}%)")

results["phase_distribution"] = {
    "real_phase_T_le_1_64": real_phase,
    "virtual_phase_1_64_lt_T_le_1_8": virtual_phase,
    "no_phase_T_gt_1_8": no_phase,
    "total": total
}

# ── 7. 谱系压缩（对T(G)最低的3个头） ──
log("谱系压缩（T(G)最低的3个头）...")
layer_data.sort(key=lambda x: x[2])  # 按T(G)排序
top3 = layer_data[:3]

def spectral_compress(adj, max_clique_size=12, min_clique_size=3):
    """
    谱系压缩：贪心团覆盖 → 压缩图 → 路径 → 展开
    简化版：找极大团，贪心覆盖，压缩图度数排序
    """
    n = adj.shape[0]
    # 贪心找团：从度最大的节点开始，逐步加入与当前团全连通的节点
    cliques = []
    covered = set()
    degrees = adj.sum(axis=1)
    order = np.argsort(-degrees)
    
    for start in order:
        if start in covered:
            continue
        # 构建团
        clique = [start]
        candidates = set(np.where(adj[start])[0]) - {start}
        while len(clique) < max_clique_size and candidates:
            # 找与当前团全连通的候选
            valid = []
            for c in candidates:
                if all(adj[c, m] for m in clique):
                    valid.append(c)
            if not valid:
                break
            # 选度最大的
            best = max(valid, key=lambda x: degrees[x])
            clique.append(best)
            candidates = candidates & set(np.where(adj[best])[0]) - set(clique)
        
        if len(clique) >= min_clique_size:
            cliques.append(clique)
            covered.update(clique)
    
    # 未覆盖节点单独成团
    for i in range(n):
        if i not in covered:
            cliques.append([i])
            covered.add(i)
    
    n_cliques = len(cliques)
    compression_ratio = n / max(n_cliques, 1)
    
    # 压缩图：两个团共享节点则有边
    node_to_clique = {}
    for ci, clique in enumerate(cliques):
        for node in clique:
            node_to_clique.setdefault(node, []).append(ci)
    
    comp_adj = np.zeros((n_cliques, n_cliques), dtype=np.int8)
    for node, clique_list in node_to_clique.items():
        for i in range(len(clique_list)):
            for j in range(i+1, len(clique_list)):
                comp_adj[clique_list[i], clique_list[j]] = 1
                comp_adj[clique_list[j], clique_list[i]] = 1
    
    comp_density = comp_adj.sum() / max(n_cliques * (n_cliques - 1), 1)
    
    # 虚拟边：压缩图路径中不相连的团对
    comp_degrees = comp_adj.sum(axis=1)
    comp_order = np.argsort(-comp_degrees)
    virtual_edges = 0
    for i in range(len(comp_order) - 1):
        if comp_adj[comp_order[i], comp_order[i+1]] == 0:
            virtual_edges += 1
    
    return {
        "n_cliques": n_cliques,
        "compression_ratio": round(compression_ratio, 2),
        "comp_density": round(float(comp_density), 4),
        "virtual_edges": virtual_edges,
        "coverage": len(covered) / n,
        "clique_sizes": [len(c) for c in cliques],
    }

compress_results = []
for layer_idx, head_idx, t_g, density, adj in top3:
    log(f"  L{layer_idx}H{head_idx}: T(G)={t_g:.4f}, density={density:.4f}")
    t0 = time.time()
    cr = spectral_compress(adj)
    elapsed = time.time() - t0
    cr["layer"] = layer_idx
    cr["head"] = head_idx
    cr["T_G"] = round(t_g, 6)
    cr["time_s"] = round(elapsed, 2)
    compress_results.append(cr)
    log(f"    团数={cr['n_cliques']}, 压缩比={cr['compression_ratio']}x, "
        f"虚拟边={cr['virtual_edges']}, 密度={cr['comp_density']}, "
        f"耗时={elapsed:.1f}s")

results["spectral_compression_top3"] = compress_results

# ── 8. 保存 ──
with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
log(f"结果保存: {RESULTS}")
log("阶段0完成。")
