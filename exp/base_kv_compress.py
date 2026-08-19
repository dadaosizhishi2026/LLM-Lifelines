"""
云韶框架·AI特化·阶段0v4
方向3：KV Cache压缩（用T(G)/团结构决定KV保留策略）
不动注意力计算，只决定"哪些token的KV该留"
对比：T(G)策略 vs 随机丢弃 vs 尾部保留（StreamingLLM式）
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "base_kv_compress_results.json")

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── 1. 加载 ──
log("加载 Qwen2.5-0.5B (CPU)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, output_attentions=True, trust_remote_code=True,
    dtype=torch.float32, low_cpu_mem_usage=True
)
model.eval()

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

# ── 2. 原始前向（带KV cache和注意力） ──
log("原始前向...")
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)
    attentions = outputs.attentions
    logits_orig = outputs.logits
    past_kv = outputs.past_key_values  # DynamicCache in transformers 5.x

shift_logits = logits_orig[:, :-1, :].contiguous()
shift_labels = inputs["input_ids"][:, 1:].contiguous()
orig_loss = torch.nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
orig_ppl = torch.exp(orig_loss).item()
log(f"原始 perplexity: {orig_ppl:.4f}")

# DynamicCache → 转成 list of (key, value) tuples
if hasattr(past_kv, 'key_cache'):
    past_kv_list = [(past_kv.key_cache[i], past_kv.value_cache[i]) for i in range(len(past_kv.key_cache))]
else:
    past_kv_list = list(past_kv)
log(f"KV cache: {len(past_kv_list)}层, key shape={past_kv_list[0][0].shape}")

# ── 3. 用注意力矩阵计算每个token的结构重要性 ──
log("\n计算token结构重要性...")

def compute_token_importance(attentions, n_tokens, k=64):
    """
    对每层每头：top-k建图 → 计算每个token的度（结构重要性）
    返回：每层每头的token重要性分数 (n_layers, n_heads, n_tokens)
    """
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    importance = np.zeros((n_layers, n_heads, n_tokens))
    
    for li in range(n_layers):
        attn = attentions[li][0].numpy()  # (heads, seq, seq)
        for hi in range(n_heads):
            head_attn = attn[hi]  # (seq, seq)
            # 因果注意力：只看下三角
            causal_attn = np.tril(head_attn)
            # 每个token被多少其他token关注（入度）= 结构重要性
            # 入度高 = 这个token是"hub" = 很多后续token需要它
            in_degree = causal_attn.sum(axis=0)  # 列求和 = 被关注总量
            # 出度也重要（这个token关注多少其他token）
            out_degree = causal_attn.sum(axis=1)  # 行求和 = 关注总量
            # 综合：入度×出度（hub = 既被关注又关注别人）
            importance[li, hi] = in_degree * out_degree
    
    return importance

importance = compute_token_importance(attentions, n_tokens)

# 跨层跨头聚合：每个token的全局重要性
global_importance = importance.mean(axis=(0, 1))  # (n_tokens,)
log(f"全局重要性: min={global_importance.min():.6f}, max={global_importance.max():.6f}")

# ── 4. 三种KV保留策略 ──
KEEP_RATIOS = [0.75, 0.50, 0.25]  # 保留75%/50%/25%的KV

# eval_kv_compression removed — using attention_coverage directly

def attention_coverage(attentions, keep_indices, n_tokens):
    """计算保留token覆盖了多少注意力质量（因果方向）"""
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    coverages = []
    
    for li in range(n_layers):
        attn = attentions[li][0].numpy()  # (heads, seq, seq)
        for hi in range(n_heads):
            head_attn = np.tril(attn[hi])  # 因果
            total = head_attn.sum()
            if total == 0:
                coverages.append(1.0)
                continue
            # 保留列（被保留token作为key被关注的量）
            kept_mass = head_attn[:, keep_indices[li]].sum()
            coverages.append(kept_mass / total)
    
    return np.mean(coverages), np.median(coverages), np.min(coverages)

results = {
    "model": "Qwen2.5-0.5B",
    "n_tokens": n_tokens,
    "orig_perplexity": orig_ppl,
    "strategies": {}
}

for keep_ratio in KEEP_RATIOS:
    n_keep = max(int(n_tokens * keep_ratio), 4)
    log(f"\n═══ 保留 {keep_ratio*100:.0f}% ({n_keep}/{n_tokens} tokens) ═══")
    
    strategy_results = {}
    
    # 策略A：T(G)结构重要性（保留全局重要性最高的token）
    top_indices_global = np.argsort(-global_importance)[:n_keep]
    top_indices_global = np.sort(top_indices_global)  # 保持顺序
    keep_A = [top_indices_global] * len(attentions)  # 所有层用同一组
    cov_A = attention_coverage(attentions, keep_A, n_tokens)
    strategy_results["A_structural_importance"] = {
        "mean_coverage": round(float(cov_A[0]), 4),
        "median_coverage": round(float(cov_A[1]), 4),
        "min_coverage": round(float(cov_A[2]), 4),
    }
    log(f"  A·结构重要性: 覆盖 mean={cov_A[0]:.4f}, median={cov_A[1]:.4f}, min={cov_A[2]:.4f}")
    
    # 策略B：逐层独立选（每层用自己的重要性排序）
    keep_B = []
    for li in range(len(attentions)):
        layer_importance = importance[li].mean(axis=0)  # (n_tokens,) 跨头平均
        top_idx = np.argsort(-layer_importance)[:n_keep]
        keep_B.append(np.sort(top_idx))
    cov_B = attention_coverage(attentions, keep_B, n_tokens)
    strategy_results["B_per_layer_importance"] = {
        "mean_coverage": round(float(cov_B[0]), 4),
        "median_coverage": round(float(cov_B[1]), 4),
        "min_coverage": round(float(cov_B[2]), 4),
    }
    log(f"  B·逐层重要性: 覆盖 mean={cov_B[0]:.4f}, median={cov_B[1]:.4f}, min={cov_B[2]:.4f}")
    
    # 策略C：StreamingLLM式（保留前4个 + 最后n_keep-4个）
    n_recent = n_keep - 4
    keep_C_idx = np.array(list(range(4)) + list(range(n_tokens - n_recent, n_tokens)))
    keep_C_idx = np.sort(np.unique(keep_C_idx))
    keep_C = [keep_C_idx] * len(attentions)
    cov_C = attention_coverage(attentions, keep_C, n_tokens)
    strategy_results["C_streaming_llm"] = {
        "mean_coverage": round(float(cov_C[0]), 4),
        "median_coverage": round(float(cov_C[1]), 4),
        "min_coverage": round(float(cov_C[2]), 4),
    }
    log(f"  C·StreamingLLM: 覆盖 mean={cov_C[0]:.4f}, median={cov_C[1]:.4f}, min={cov_C[2]:.4f}")
    
    # 策略D：随机（基线）
    np.random.seed(42)
    keep_D_idx = np.sort(np.random.choice(n_tokens, n_keep, replace=False))
    keep_D = [keep_D_idx] * len(attentions)
    cov_D = attention_coverage(attentions, keep_D, n_tokens)
    strategy_results["D_random"] = {
        "mean_coverage": round(float(cov_D[0]), 4),
        "median_coverage": round(float(cov_D[1]), 4),
        "min_coverage": round(float(cov_D[2]), 4),
    }
    log(f"  D·随机:       覆盖 mean={cov_D[0]:.4f}, median={cov_D[1]:.4f}, min={cov_D[2]:.4f}")
    
    # 策略E：H2O式（保留累积注意力最高的token，即"heavy hitter"）
    # 每层每头独立选累积注意力最高的token
    keep_E = []
    for li in range(len(attentions)):
        attn = attentions[li][0].numpy()
        # 累积注意力 = 每个token被所有后续token关注的总量（列求和）
        cum_attn = np.tril(attn.mean(axis=0)).sum(axis=0)  # (seq,)
        top_idx = np.argsort(-cum_attn)[:n_keep]
        keep_E.append(np.sort(top_idx))
    cov_E = attention_coverage(attentions, keep_E, n_tokens)
    strategy_results["E_h2o_heavy_hitter"] = {
        "mean_coverage": round(float(cov_E[0]), 4),
        "median_coverage": round(float(cov_E[1]), 4),
        "min_coverage": round(float(cov_E[2]), 4),
    }
    log(f"  E·H2O:        覆盖 mean={cov_E[0]:.4f}, median={cov_E[1]:.4f}, min={cov_E[2]:.4f}")
    
    # 对比
    log(f"  ── 对比（mean coverage）──")
    log(f"     A结构={cov_A[0]:.4f} | B逐层={cov_B[0]:.4f} | C流式={cov_C[0]:.4f} | D随机={cov_D[0]:.4f} | E·H2O={cov_E[0]:.4f}")
    
    results["strategies"][f"keep_{keep_ratio}"] = strategy_results

# ── 5. 保存 ──
with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
log(f"\n结果保存: {RESULTS}")
log("阶段0v4完成。")
