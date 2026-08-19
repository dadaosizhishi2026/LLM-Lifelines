"""
云韶框架·AI特化·阶段1v2
真实perplexity验证（干净版）
方法：用4D attention mask屏蔽被丢弃token，不手动拆层
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

# ── 1. 加载（eager注意力） ──
log("加载 Qwen2.5-0.5B (CPU, eager attention)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True,
    dtype=torch.float32, low_cpu_mem_usage=True,
    attn_implementation="eager",
)
model.eval()
log("模型加载完成")

# ── 2. 文本 ──
TEXTS = [
    "The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures.",
    "Transformer models use self-attention mechanisms to process sequences of tokens. The attention matrix defines a weighted directed graph where tokens are nodes and attention weights are edge weights. This graph structure can be analyzed using algebraic topology and spectral methods to find optimal compression strategies.",
    "In mixture of experts models, each token is routed to a subset of specialized expert networks. The routing decisions create a bipartite graph between tokens and experts. Analyzing this graph structure reveals load balancing issues and suggests optimal routing strategies that maximize expert utilization.",
    "Quantum error correction uses surface codes arranged on a grid topology. The syndrome measurement creates a graph where errors form chains that must be matched and corrected. Finding the optimal matching path through this syndrome graph is equivalent to finding a minimum weight perfect matching.",
    "The algebraic tension of a graph measures the minimum fraction of missing edges in any permutation of vertices. When tension is below one sixty-fourth, the graph certainly contains a Hamiltonian path. When tension exceeds one eighth, no polynomial method can construct such a path.",
]

all_ids = [tokenizer(t, return_tensors="pt", truncation=True, max_length=256)["input_ids"] for t in TEXTS]
log(f"{len(TEXTS)} 段文本, 长度: {[ids.shape[1] for ids in all_ids]}")

# ── 3. 核心函数 ──
def get_full_attention(model, input_ids):
    """跑一次完整前向，拿注意力矩阵"""
    with torch.no_grad():
        out = model(input_ids, output_attentions=True)
    return out.attentions, out.logits

def compute_importance(attentions, n_tokens, k=64):
    """逐层逐头的token结构重要性（入度×出度）"""
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    importance = np.zeros((n_layers, n_heads, n_tokens))
    
    for li in range(n_layers):
        attn = attentions[li][0].numpy()
        for hi in range(n_heads):
            causal = np.tril(attn[hi])
            in_deg = causal.sum(axis=0)
            out_deg = causal.sum(axis=1)
            importance[li, hi] = in_deg * out_deg
    return importance

def build_eviction_mask(n_tokens, keep_indices, device):
    """
    构建4D causal attention mask，屏蔽被丢弃的token
    keep_indices: 保留的token位置（sorted array）
    返回: (1, 1, n, n) float mask，-inf=屏蔽，0=允许
    """
    mask = torch.full((1, 1, n_tokens, n_tokens), float('-inf'), device=device)
    keep_set = set(keep_indices.tolist())
    for i in range(n_tokens):
        # 位置i只能attend到keep_indices中≤i的位置（因果）
        for j in keep_indices:
            if j <= i:
                mask[0, 0, i, j] = 0.0
    return mask

def ppl_with_mask(model, input_ids, attn_mask):
    """用自定义attention mask跑前向，返回perplexity"""
    with torch.no_grad():
        out = model(input_ids, attention_mask=attn_mask)
        logits = out.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = torch.nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return torch.exp(loss).item()

def ppl_full(model, input_ids):
    """完整前向perplexity"""
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = torch.nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return torch.exp(loss).item()

# ── 4. 实验 ──
log("\n═══ 真实perplexity实验 ═══")
results = {"model": "Qwen2.5-0.5B", "n_texts": len(TEXTS), "experiments": []}

KEEP_RATIOS = [1.0, 0.75, 0.50, 0.25]
STRATEGIES = ["structural", "h2o", "recent", "random"]

for keep_ratio in KEEP_RATIOS:
    log(f"\n── 保留 {keep_ratio*100:.0f}% ──")
    ratio_results = {}
    
    for strategy in STRATEGIES:
        ppls_full = []
        ppls_comp = []
        
        for ti, input_ids in enumerate(all_ids):
            n = input_ids.shape[1]
            n_keep = max(int(n * keep_ratio), 8)
            
            # 完整PPL
            pf = ppl_full(model, input_ids)
            ppls_full.append(pf)
            
            if keep_ratio >= 1.0:
                ppls_comp.append(pf)
                continue
            
            # 拿注意力算重要性
            attentions, _ = get_full_attention(model, input_ids)
            importance = compute_importance(attentions, n)
            
            # 全局重要性（跨层跨头平均）
            global_imp = importance.mean(axis=(0, 1))
            
            if strategy == "structural":
                keep_idx = np.sort(np.argsort(-global_imp)[:n_keep])
            elif strategy == "h2o":
                # 纯入度
                in_deg = np.zeros(n)
                for li in range(len(attentions)):
                    attn = attentions[li][0].numpy().mean(axis=0)
                    in_deg += np.tril(attn).sum(axis=0)
                keep_idx = np.sort(np.argsort(-in_deg)[:n_keep])
            elif strategy == "recent":
                n_recent = n_keep - 4
                keep_idx = np.sort(np.array(list(range(4)) + list(range(n - n_recent, n))))
            elif strategy == "random":
                np.random.seed(42)
                keep_idx = np.sort(np.random.choice(n, n_keep, replace=False))
            
            # 确保最后一个token保留
            if n - 1 not in keep_idx:
                keep_idx[-1] = n - 1
                keep_idx = np.sort(np.unique(keep_idx))
            
            # 构建mask并测PPL
            mask = build_eviction_mask(n, keep_idx, input_ids.device)
            pc = ppl_with_mask(model, input_ids, mask)
            ppls_comp.append(pc)
        
        mean_f = np.mean(ppls_full)
        mean_c = np.mean(ppls_comp)
        ratio = mean_c / mean_f
        ratio_results[strategy] = {
            "mean_ppl_full": round(mean_f, 2),
            "mean_ppl_compressed": round(mean_c, 2),
            "ratio": round(ratio, 4),
            "increase_pct": round((ratio - 1) * 100, 1),
        }
        log(f"  {strategy:12s}: PPL {mean_f:.2f} → {mean_c:.2f} (×{ratio:.3f}, +{(ratio-1)*100:.1f}%)")
    
    results["experiments"].append({"keep_ratio": keep_ratio, "strategies": ratio_results})

# ── 5. T(G)逐层诊断 ──
log("\n═══ T(G)逐层诊断 ═══")
attentions_diag, _ = get_full_attention(model, all_ids[0])
n_diag = all_ids[0].shape[1]
K = min(64, n_diag - 1)

layer_diagnosis = []
for li in range(len(attentions_diag)):
    attn = attentions_diag[li][0].numpy()
    n_heads = attn.shape[0]
    head_tgs = []
    
    for hi in range(n_heads):
        causal = np.tril(attn[hi])
        sym = (causal + causal.T) / 2
        np.fill_diagonal(sym, 0)
        adj = np.zeros((n_diag, n_diag), dtype=np.int8)
        for i in range(n_diag):
            top_idx = np.argsort(sym[i])[-K:]
            for j in top_idx:
                adj[i, j] = 1
                adj[j, i] = 1
        
        degrees = adj.sum(axis=1)
        visited = np.zeros(n_diag, dtype=bool)
        start = np.argmax(degrees)
        path = [start]
        visited[start] = True
        for _ in range(n_diag - 1):
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
        head_tgs.append(d / max(n_diag - 1, 1))
    
    mean_tg = float(np.mean(head_tgs))
    if mean_tg <= 1/64:
        phase = "真实解相"
        rec = "compress_75%"
    elif mean_tg <= 1/8:
        phase = "虚拟解相"
        rec = "compress_50%"
    else:
        phase = "无解相"
        rec = "keep_all"
    
    layer_diagnosis.append({"layer": li, "mean_T_G": round(mean_tg, 4), "phase": phase, "recommendation": rec})
    log(f"  L{li:2d}: T(G)={mean_tg:.4f} → {phase} → {rec}")

results["layer_diagnosis"] = layer_diagnosis

n_agg = sum(1 for d in layer_diagnosis if d["recommendation"] == "compress_75%")
n_mod = sum(1 for d in layer_diagnosis if d["recommendation"] == "compress_50%")
n_keep = sum(1 for d in layer_diagnosis if d["recommendation"] == "keep_all")
log(f"\n汇总: 激进压缩={n_agg}层, 适度={n_mod}层, 保留={n_keep}层")
log(f"理论KV节省: {(n_agg*0.75 + n_mod*0.5 + n_keep*0.0) / len(layer_diagnosis) * 100:.0f}%")

# ── 6. 保存 ──
with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
log(f"\n结果保存: {RESULTS}")
log("阶段1v2完成。")
