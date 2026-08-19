"""
云韶框架·AI特化·阶段1
真实perplexity验证 + 团内token合并 + T(G)逐层诊断
v4证明了覆盖率，这里证明实际生成质量
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "exp1_results.json")

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── 1. 加载 ──
log("加载 Qwen2.5-0.5B (CPU)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True,
    dtype=torch.float32, low_cpu_mem_usage=True
)
model.eval()

# ── 2. 准备多段文本（测perplexity需要多段） ──
TEXTS = [
    """The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures.""",
    """Transformer models use self-attention mechanisms to process sequences of tokens. The attention matrix defines a weighted directed graph where tokens are nodes and attention weights are edge weights. This graph structure can be analyzed using algebraic topology and spectral methods to find optimal compression strategies.""",
    """In mixture of experts models, each token is routed to a subset of specialized expert networks. The routing decisions create a bipartite graph between tokens and experts. Analyzing this graph structure reveals load balancing issues and suggests optimal routing strategies that maximize expert utilization.""",
    """Quantum error correction uses surface codes arranged on a grid topology. The syndrome measurement creates a graph where errors form chains that must be matched and corrected. Finding the optimal matching path through this syndrome graph is equivalent to finding a minimum weight perfect matching.""",
    """The algebraic tension of a graph measures the minimum fraction of missing edges in any permutation of vertices. When tension is below one sixty-fourth, the graph certainly contains a Hamiltonian path. When tension exceeds one eighth, no polynomial method can construct such a path.""",
]

# tokenize所有文本
all_input_ids = []
for text in TEXTS:
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)["input_ids"]
    all_input_ids.append(ids)
log(f"准备 {len(TEXTS)} 段文本, 长度: {[ids.shape[1] for ids in all_input_ids]}")

# ── 3. 逐token生成式perplexity测量（带KV压缩） ──
def compute_ppl_with_kv_eviction(model, input_ids, keep_ratio, strategy="structural"):
    """
    逐token前向，每步压缩KV cache，测perplexity
    strategy: "structural" | "recent" | "random" | "none"
    """
    n = input_ids.shape[1]
    if n < 10:
        return None
    
    n_keep = max(int(n * keep_ratio), 8)
    
    # 先跑一次完整前向拿注意力（用于计算重要性）
    with torch.no_grad():
        out_full = model(input_ids, output_attentions=True)
        attentions = out_full.attentions
        logits_full = out_full.logits
    
    # 完整perplexity
    shift_logits = logits_full[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss_full = torch.nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    ppl_full = torch.exp(loss_full).item()
    
    if strategy == "none":
        return ppl_full, ppl_full, 1.0
    
    # 计算每层的token重要性
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    
    # 逐层KV压缩：对每层独立选保留token
    # 用注意力矩阵的列求和（入度）作为重要性
    layer_keep_indices = []
    for li in range(n_layers):
        attn = attentions[li][0].numpy()  # (heads, seq, seq)
        # 因果注意力列求和 = 每个token被关注的总量
        causal_attn = np.tril(attn.mean(axis=0))  # (seq, seq) 跨头平均
        col_sum = causal_attn.sum(axis=0)  # (seq,) 入度
        
        if strategy == "structural":
            # 结构重要性：入度×出度
            row_sum = causal_attn.sum(axis=1)  # 出度
            importance = col_sum * row_sum
            keep_idx = np.argsort(-importance)[:n_keep]
        elif strategy == "h2o":
            # H2O: 纯入度（heavy hitter）
            keep_idx = np.argsort(-col_sum)[:n_keep]
        elif strategy == "recent":
            # StreamingLLM: 前4 + 最后n_keep-4
            n_recent = n_keep - 4
            keep_idx = np.array(list(range(4)) + list(range(n - n_recent, n)))
        elif strategy == "random":
            np.random.seed(42 + li)
            keep_idx = np.random.choice(n, n_keep, replace=False)
        else:
            keep_idx = np.arange(n_keep)
        
        # 必须包含最后一个token（当前生成位置）
        if n - 1 not in keep_idx:
            keep_idx[-1] = n - 1
        keep_idx = np.sort(np.unique(keep_idx))
        layer_keep_indices.append(keep_idx)
    
    # 用压缩KV重跑前向
    # 方法：手动构建压缩后的attention mask
    # 简化：对每层，只保留keep_idx对应的KV，重算attention
    with torch.no_grad():
        # 逐层前向，手动压缩KV
        hidden = model.model.embed_tokens(input_ids)
        
        for li in range(n_layers):
            layer = model.model.layers[li]
            keep_idx = layer_keep_indices[li]
            keep_tensor = torch.tensor(keep_idx, dtype=torch.long)
            
            # 正常前向这一层
            # 用完整hidden过layer，但修改attention mask使得只attend到keep_idx
            # 构建attention mask: (1, 1, seq, seq)
            # 对于位置i，只能attend到keep_idx中≤i的位置
            attn_mask = torch.full((1, 1, n, n), float('-inf'))
            for pos in range(n):
                valid_keys = keep_idx[keep_idx <= pos]
                if len(valid_keys) > 0:
                    attn_mask[0, 0, pos, valid_keys] = 0.0
            
            # 过self-attention + FFN
            residual = hidden
            hidden_norm = layer.input_layernorm(hidden)
            
            # self attention with custom mask
            attn_output = layer.self_attn(
                hidden_norm,
                attention_mask=attn_mask,
                position_ids=torch.arange(n).unsqueeze(0),
            )
            if isinstance(attn_output, tuple):
                attn_out = attn_output[0]
            else:
                attn_out = attn_output
            hidden = residual + attn_out
            
            # FFN
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        
        # final norm + lm_head
        hidden = model.model.norm(hidden)
        logits_compressed = model.lm_head(hidden)
    
    shift_logits_c = logits_compressed[:, :-1, :].contiguous()
    loss_c = torch.nn.CrossEntropyLoss()(shift_logits_c.view(-1, shift_logits_c.size(-1)), shift_labels.view(-1))
    ppl_compressed = torch.exp(loss_c).item()
    
    ratio = ppl_compressed / ppl_full
    return ppl_full, ppl_compressed, ratio

# ── 4. 跑实验 ──
log("\n═══ 真实perplexity实验 ═══")
results = {
    "model": "Qwen2.5-0.5B",
    "n_texts": len(TEXTS),
    "experiments": []
}

KEEP_RATIOS = [1.0, 0.75, 0.50, 0.25]
STRATEGIES = ["structural", "h2o", "recent", "random"]

for keep_ratio in KEEP_RATIOS:
    log(f"\n── 保留 {keep_ratio*100:.0f}% ──")
    ratio_results = {}
    
    for strategy in STRATEGIES:
        ppls_full = []
        ppls_comp = []
        ratios = []
        
        for ti, input_ids in enumerate(all_input_ids):
            try:
                result = compute_ppl_with_kv_eviction(model, input_ids, keep_ratio, strategy)
                if result is None:
                    continue
                ppl_f, ppl_c, r = result
                ppls_full.append(ppl_f)
                ppls_comp.append(ppl_c)
                ratios.append(r)
            except Exception as e:
                log(f"  {strategy} text{ti} 错误: {e}")
                continue
        
        if ratios:
            mean_ratio = np.mean(ratios)
            mean_ppl_f = np.mean(ppls_full)
            mean_ppl_c = np.mean(ppls_comp)
            ratio_results[strategy] = {
                "mean_ppl_full": round(mean_ppl_f, 2),
                "mean_ppl_compressed": round(mean_ppl_c, 2),
                "mean_ratio": round(mean_ratio, 4),
                "ppl_increase_pct": round((mean_ratio - 1) * 100, 1),
            }
            log(f"  {strategy:12s}: PPL {mean_ppl_f:.2f}→{mean_ppl_c:.2f} (×{mean_ratio:.3f}, +{(mean_ratio-1)*100:.1f}%)")
    
    results["experiments"].append({
        "keep_ratio": keep_ratio,
        "strategies": ratio_results,
    })

# ── 5. T(G)逐层诊断 ──
log("\n═══ T(G)逐层诊断 ═══")
log("哪些层可以激进压缩，哪些必须保留？")

# 用第一段文本的注意力
with torch.no_grad():
    out_diag = model(all_input_ids[0], output_attentions=True)
    diag_attns = out_diag.attentions

n_tokens_diag = all_input_ids[0].shape[1]
K = 64

layer_diagnosis = []
for li in range(len(diag_attns)):
    attn = diag_attns[li][0].numpy()
    n_heads = attn.shape[0]
    
    head_tgs = []
    for hi in range(n_heads):
        head_attn = np.tril(attn[hi])
        sym = (head_attn + head_attn.T) / 2
        np.fill_diagonal(sym, 0)
        # top-k
        adj = np.zeros((n_tokens_diag, n_tokens_diag), dtype=np.int8)
        for i in range(n_tokens_diag):
            top_idx = np.argsort(sym[i])[-K:]
            for j in top_idx:
                adj[i, j] = 1
                adj[j, i] = 1
        
        # 快速T(G)估计
        degrees = adj.sum(axis=1)
        visited = np.zeros(n_tokens_diag, dtype=bool)
        start = np.argmax(degrees)
        path = [start]
        visited[start] = True
        for _ in range(n_tokens_diag - 1):
            curr = path[-1]
            neighbors = np.where(adj[curr] & ~visited)[0]
            if len(neighbors) == 0:
                unvisited = np.where(~visited)[0]
                if len(unvisited) == 0:
                    break
                nxt = unvisited[np.argmax(degrees[unvisited])]
            else:
                nxt = neighbors[np.argmax(degrees[neighbors])]
            path.append(nxt)
            visited[nxt] = True
        d = sum(1 for i in range(len(path)-1) if adj[path[i], path[i+1]] == 0)
        t_g = d / max(n_tokens_diag - 1, 1)
        head_tgs.append(t_g)
    
    mean_tg = np.mean(head_tgs)
    min_tg = np.min(head_tgs)
    
    # 诊断
    if mean_tg <= 1/64:
        phase = "真实解相（可激进压缩）"
    elif mean_tg <= 1/8:
        phase = "虚拟解相（适度压缩+桥接）"
    else:
        phase = "无解相（必须保留全部KV）"
    
    layer_diagnosis.append({
        "layer": li,
        "mean_T_G": round(float(mean_tg), 4),
        "min_T_G": round(float(min_tg), 4),
        "phase": phase,
        "recommendation": "compress_75%" if mean_tg <= 1/64 else ("compress_50%" if mean_tg <= 1/8 else "keep_all"),
    })
    log(f"  L{li:2d}: T(G)={mean_tg:.4f} → {phase}")

results["layer_diagnosis"] = layer_diagnosis

# 汇总
n_compress_aggressive = sum(1 for d in layer_diagnosis if d["recommendation"] == "compress_75%")
n_compress_moderate = sum(1 for d in layer_diagnosis if d["recommendation"] == "compress_50%")
n_keep_all = sum(1 for d in layer_diagnosis if d["recommendation"] == "keep_all")
log(f"\n诊断汇总: 激进压缩={n_compress_aggressive}层, 适度压缩={n_compress_moderate}层, 保留全部={n_keep_all}层")
log(f"理论KV节省: {(n_compress_aggressive*0.75 + n_compress_moderate*0.5) / len(layer_diagnosis) * 100:.0f}%")

# ── 6. 保存 ──
with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
log(f"\n结果保存: {RESULTS}")
log("阶段1完成。")
