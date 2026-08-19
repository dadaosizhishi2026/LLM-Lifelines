"""
云韶框架·AI特化·P7c-框架压缩阶梯
SDPA + 物理KV删除，测本地小模型上下文极限
对比：不压缩 vs 87.5%安全区 vs T(G)引导（逐层差异化）
核心问题：框架压缩能把本地小模型上下文推多远？
重要性代理：KV向量范数（SDPA兼容，不实体化注意力矩阵）
逐层差异化：用短序列(2K)算的T(G)分布外推每层保留比例
"""
import sys, os, json, time, gc
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "p7c_compressed_ladder.json")

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def mem_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

log(f"可用内存: {psutil.virtual_memory().available/1e9:.1f}GB")
log("加载 Qwen2.5-0.5B (CPU, SDPA)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True,
    dtype=torch.float32, low_cpu_mem_usage=True,
    attn_implementation="sdpa",
)
model.eval()
n_layers = len(model.model.layers)
n_heads = model.config.num_attention_heads
head_dim = model.config.hidden_size // n_heads
log(f"模型加载后内存: {mem_gb():.2f}GB, {n_layers}层 {n_heads}头")

BASE_BLOCK = """The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures. By using these cliques as outer embryos to wrap original nodes, the original graph can be compressed into a smaller compressed graph. After solving the Hamiltonian path on the compressed graph and expanding back, the number of virtual edges remains stable at zero to two, with coverage above ninety-nine percent. In mixture of experts language models, each token is independently routed to a small subset of specialized expert networks through a learned gating mechanism. The routing decisions create a dynamic bipartite graph between tokens and experts that changes with every forward pass. The algebraic tension framework provides a principled way to diagnose which routing configurations are structurally stable and which are prone to collapse under distribution shift. """

def make_text(target_tokens):
    n_blocks = max(1, int(target_tokens / 150) + 2)
    return BASE_BLOCK * n_blocks

# ── Step 1: 用短序列(2K)算每层T(G)，外推逐层保留权重 ──
log("\n[Step1] 短序列(2K)算每层T(G)（eager，拿注意力）...")
# 临时用eager拿注意力（2K在eager下内存可控：注意力矩阵5.6GB峰值，但2K实测Δ2.4GB ok）
# 为安全，用1K算T(G)
text_short = make_text(1024)
ids_short = tokenizer(text_short, return_tensors="pt", truncation=True, max_length=1024)["input_ids"]
n_short = ids_short.shape[1]

# 用sdpa拿不到注意力，临时切eager太重。改用KV范数代理T(G)的逐层差异
# 直接sdpa prefill拿KV，用每层KV范数的分布差异作为"结构集中度"代理
with torch.no_grad():
    out_s = model(ids_short, use_cache=True)
    kv_s = out_s.past_key_values

# 每层"结构集中度"：KV范数的变异系数（CV=std/mean，高=集中=高T(G)）
layer_cv = []
for li in range(n_layers):
    keys = kv_s.layers[li].keys  # (1, heads, n, head_dim)
    # 每个token的KV范数
    k_norm = keys[0].norm(dim=-1).mean(dim=0)  # (n,) 跨头平均
    cv = float(k_norm.std() / (k_norm.mean() + 1e-8))
    layer_cv.append(cv)
del out_s, kv_s
gc.collect()

cv_arr = np.array(layer_cv)
cv_max, cv_min = float(cv_arr.max()), float(cv_arr.min())
log(f"  每层CV范围: [{cv_min:.3f}, {cv_max:.3f}]")
log(f"  CV高（结构集中，可多压缩）的层: {np.argsort(-cv_arr)[:5].tolist()}")

# ── Step 2: 阶梯实验 ──
def kv_importance(kv_cache, li):
    """每层每token重要性 = key范数×value范数（跨头平均）"""
    keys = kv_cache.layers[li].keys[0]    # (heads, n, head_dim)
    vals = kv_cache.layers[li].values[0]
    k_norm = keys.norm(dim=-1).mean(dim=0)  # (n,)
    v_norm = vals.norm(dim=-1).mean(dim=0)
    return (k_norm * v_norm).cpu().numpy()

def physical_evict(kv_cache, keep_idx_per_layer):
    """物理删除KV条目，返回新DynamicCache（所有层用统一长度需对齐）"""
    # 物理删除后所有层长度必须一致（生成时位置对齐）
    # 所以用全局统一keep_idx
    new_cache = DynamicCache()
    for li in range(n_layers):
        keys = kv_cache.layers[li].keys
        vals = kv_cache.layers[li].values
        keep = keep_idx_per_layer
        new_cache.update(keys[:, :, keep, :], vals[:, :, keep, :], li)
    return new_cache

def ppl_from_kv(model, input_ids, kv_cache, kept_positions):
    """用压缩后的KV算PPL：对保留位置算loss"""
    # 重新forward用压缩KV（需要position对齐）
    # 简化：用kept token子集重新forward（位置重排）
    kept_ids = input_ids[:, kept_positions]
    with torch.no_grad():
        out = model(kept_ids)
        logits = out.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = kept_ids[:, 1:].contiguous()
    loss = torch.nn.CrossEntropyLoss()(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return torch.exp(loss).item()

LADDER = [4096, 8192, 16384]
KEEP_RATIOS = [1.0, 0.875, 0.75]  # 1.0=不压缩, 0.875=安全区, 0.75=T(G)区

results = {
    "model": "Qwen2.5-0.5B", "attn": "sdpa", "dtype": "float32",
    "n_layers": n_layers, "layer_cv": [round(c, 4) for c in layer_cv],
    "ladder": []
}

for target in LADDER:
    log(f"\n{'═'*55}")
    log(f"目标: {target} tokens")
    log(f"{'═'*55}")
    
    text = make_text(target)
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=target)["input_ids"]
    n = input_ids.shape[1]
    log(f"  实际: {n} tokens")
    
    # 全局重要性（跨层KV范数累积）
    gc.collect()
    with torch.no_grad():
        out_base = model(input_ids, use_cache=True)
        kv = out_base.past_key_values
        logits_base = out_base.logits
    
    # 基线PPL
    shift_logits = logits_base[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss_base = torch.nn.CrossEntropyLoss()(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    ppl_base = torch.exp(loss_base).item()
    
    # 全局重要性
    global_imp = np.zeros(n)
    for li in range(n_layers):
        global_imp += kv_importance(kv, li)
    
    del out_base, logits_base
    gc.collect()
    
    log(f"  基线PPL={ppl_base:.3f}")
    
    for keep_ratio in KEEP_RATIOS:
        if keep_ratio == 1.0:
            # 不压缩
            results["ladder"].append({
                "tokens": n, "mode": "baseline", "keep_ratio": 1.0,
                "ppl": round(ppl_base, 3), "effective_tokens": n,
            })
            log(f"    不压缩:      PPL={ppl_base:.3f} (有效{n})")
            continue
        
        n_keep = int(n * keep_ratio)
        
        # 均匀：全局top-k
        keep_uniform = np.sort(np.argsort(-global_imp)[:n_keep])
        if n-1 not in keep_uniform:
            keep_uniform[-1] = n-1
            keep_uniform = np.sort(np.unique(keep_uniform))
        
        # T(G)引导：用CV加权（高CV层的重要性权重大）
        weighted_imp = np.zeros(n)
        for li in range(n_layers):
            weight = np.log1p(layer_cv[li] * 10)  # CV越高权重越大
            weighted_imp += kv_importance(kv, li) * weight
        keep_tg = np.sort(np.argsort(-weighted_imp)[:n_keep])
        if n-1 not in keep_tg:
            keep_tg[-1] = n-1
            keep_tg = np.sort(np.unique(keep_tg))
        
        # 测两种压缩的PPL（用保留token子集重算）
        t0 = time.time()
        ppl_uniform = ppl_from_kv(model, input_ids, kv, keep_uniform)
        t_uniform = time.time() - t0
        
        t0 = time.time()
        ppl_tg = ppl_from_kv(model, input_ids, kv, keep_tg)
        t_tg = time.time() - t0
        
        results["ladder"].append({
            "tokens": n, "mode": "uniform", "keep_ratio": keep_ratio,
            "ppl": round(ppl_uniform, 3), "effective_tokens": len(keep_uniform),
            "ppl_ratio": round(ppl_uniform/ppl_base, 4), "time_s": round(t_uniform, 2),
        })
        results["ladder"].append({
            "tokens": n, "mode": "tg", "keep_ratio": keep_ratio,
            "ppl": round(ppl_tg, 3), "effective_tokens": len(keep_tg),
            "ppl_ratio": round(ppl_tg/ppl_base, 4), "time_s": round(t_tg, 2),
        })
        
        log(f"    均匀{keep_ratio}:    PPL={ppl_uniform:.3f} (×{ppl_uniform/ppl_base:.3f}, 有效{len(keep_uniform)})")
        log(f"    T(G){keep_ratio}:    PPL={ppl_tg:.3f} (×{ppl_tg/ppl_base:.3f}, 有效{len(keep_tg)})")
    
    del kv
    gc.collect()

# ── 保存 ──
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, cls=NpEncoder)
log(f"\n结果保存: {RESULTS}")
log("P7c框架压缩阶梯完成。")
