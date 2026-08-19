"""
云韶框架·AI特化·P8-稀疏注意力（虚拟边工程实现）
核心：不materialize完整N×N注意力矩阵，逐query块只算"局部窗口+top-k枢纽"
目标：测本地小模型用稀疏注意力能到多长上下文
纯CPU，7.3GB硬上限
"""
import sys, os, time, gc, math
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn.functional as F
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def mem_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

MEM_LIMIT = 7.3

# ══ 稀疏注意力核心 ══
def sparse_prefill(q, k, v, window, n_global, scaling, q_block=256):
    """
    稀疏注意力prefill。永不materialize完整N×N。
    q: (H_q, N, D), k/v: (H_kv, N, D), GQA广播
    每个query attend: 局部窗口[i-window,i] + top-n_global全局枢纽(按key范数)
    返回: (H_q, N, D)
    """
    H_q, N, D = q.shape
    H_kv = k.shape[0]
    rep = H_q // H_kv
    device = q.device
    
    # 全局枢纽：key范数top-k（跨头平均）
    if n_global > 0:
        with torch.no_grad():
            knorm = k.float().norm(dim=-1).mean(dim=0)  # (N,)
            n_global = min(n_global, N)
            global_idx = torch.topk(knorm, n_global).indices.sort().values
    else:
        global_idx = torch.empty(0, dtype=torch.long, device=device)
    
    out = torch.zeros(H_q, N, D, device=device, dtype=q.dtype)
    
    for s in range(0, N, q_block):
        e = min(s + q_block, N)
        qb = q[:, s:e]  # (H_q, B, D)
        B = e - s
        
        # 每个query块的key集合：局部窗口 + 全局枢纽
        local_start = max(0, e - window)
        local_idx = torch.arange(local_start, e, device=device)
        if global_idx.numel() > 0:
            key_idx = torch.cat([local_idx, global_idx]).unique().sort().values
        else:
            key_idx = local_idx
        M = key_idx.numel()
        
        ks = k[:, key_idx]  # (H_kv, M, D)
        vs = v[:, key_idx]  # (H_kv, M, D)
        
        # 因果mask：query绝对位置s+bi只能attend key<=s+bi
        key_pos = key_idx.unsqueeze(0)  # (1, M)
        qpos = torch.arange(s, e, device=device).unsqueeze(1)  # (B, 1)
        causal = (key_pos <= qpos)  # (B, M)
        
        # GQA广播
        if rep > 1:
            ks = ks.repeat_interleave(rep, dim=0)
            vs = vs.repeat_interleave(rep, dim=0)
        
        # 分头手算注意力（小矩阵 B×M，不爆）
        # causal: (B, M) bool
        mask_add = torch.where(causal, torch.zeros((), device=device), torch.full((), float('-inf'), device=device))  # (B,M)
        for h in range(H_q):
            qh = qb[h]  # (B, D)
            kh = ks[h]  # (M, D)
            vh = vs[h]  # (M, D)
            scores = (qh @ kh.transpose(-1, -2)) * scaling  # (B, M)
            scores = scores + mask_add
            weights = F.softmax(scores, dim=-1)
            out[h, s:e] = weights @ vh  # (B, D)
    
    return out

# ══ 接入模型：monkey-patch Qwen2Attention.forward ══
def make_sparse_attn_forward(window, n_global, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM):
    """返回一个替换Qwen2Attention.forward的函数"""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    
    def sparse_forward(self, hidden_states, position_embeddings, attention_mask=None,
                       past_key_values=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(bsz, q_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(bsz, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(bsz, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # 稀疏注意力（单batch）
        scaling = HEAD_DIM ** -0.5
        out = sparse_prefill(q[0], k[0], v[0], window, n_global, scaling)
        attn_output = out.transpose(0, 1).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None
    
    return sparse_forward

# ══ 主流程 ══
log(f"可用内存: {psutil.virtual_memory().available/1e9:.1f}GB, 硬上限{MEM_LIMIT}GB")
log("加载 Qwen2.5-0.5B (CPU, eager)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True,
    dtype=torch.float32, low_cpu_mem_usage=True,
    attn_implementation="eager",
)
model.eval()
n_layers = len(model.model.layers)
n_heads = model.config.num_attention_heads
head_dim = model.config.hidden_size // n_heads
log(f"模型内存: {mem_gb():.2f}GB, {n_layers}层 {n_heads}头 head_dim{head_dim}")

BASE_BLOCK = """The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures. By using these cliques as outer embryos to wrap original nodes, the original graph can be compressed into a smaller compressed graph. After solving the Hamiltonian path on the compressed graph and expanding back, the number of virtual edges remains stable at zero to two, with coverage above ninety-nine percent. In mixture of experts language models, each token is independently routed to a small subset of specialized expert networks through a learned gating mechanism. The routing decisions create a dynamic bipartite graph between tokens and experts that changes with every forward pass. The algebraic tension framework provides a principled way to diagnose which routing configurations are structurally stable and which are prone to collapse under distribution shift. """

def make_ids(target):
    n_blocks = max(1, int(target / 150) + 2)
    text = BASE_BLOCK * n_blocks
    return tokenizer(text, return_tensors="pt", truncation=True, max_length=target)["input_ids"]

def ppl_of(logits, input_ids):
    sl = logits[:, :-1, :].contiguous()
    lab = input_ids[:, 1:].contiguous()
    loss = torch.nn.CrossEntropyLoss()(sl.view(-1, sl.size(-1)), lab.view(-1))
    return torch.exp(loss).item()

# patch所有层
orig_forwards = []
for li in range(n_layers):
    attn = model.model.layers[li].self_attn
    orig_forwards.append(attn.forward)

NUM_KV_HEADS = model.config.num_key_value_heads
def apply_sparse(window, n_global):
    fn = make_sparse_attn_forward(window, n_global, n_heads, NUM_KV_HEADS, head_dim)
    for li in range(n_layers):
        attn = model.model.layers[li].self_attn
        import types
        attn.forward = types.MethodType(fn, attn)

def restore():
    for li in range(n_layers):
        model.model.layers[li].self_attn.forward = orig_forwards[li]

# ══ 阶梯实验 ══
# (N, window, n_global, 稀疏度标签)
CONFIGS = [
    (4096,  410,  41,  "10%"),
    (8192,  820,  82,  "10%"),
    (8192,  410,  41,  "5%"),
    (16384, 820,  82,  "5%"),
    (16384, 164,  16,  "1%"),
    (32768, 328,  33,  "1%"),
    (32768, 164,  16,  "0.5%"),
    (65536, 328,  33,  "0.5%"),   # 超模型上限32K，RoPE外推
    (65536, 66,   7,   "0.1%"),
    (131072,131,  13,  "0.1%"),   # 超模型上限
]

results = {"model": "Qwen2.5-0.5B", "mem_limit_gb": MEM_LIMIT, "runs": []}

for N, window, n_global, sp_label in CONFIGS:
    log(f"\n{'═'*55}")
    log(f"N={N}, window={window}, global={n_global}, 稀疏度≈{sp_label}")
    log(f"{'═'*55}")
    
    if mem_gb() > MEM_LIMIT - 0.5:
        log(f"  ⛔ 内存已近上限({mem_gb():.2f}GB)，停止")
        break
    
    input_ids = make_ids(N)
    n = input_ids.shape[1]
    log(f"  实际tokens: {n}, 当前内存: {mem_gb():.2f}GB")
    
    apply_sparse(window, n_global)
    gc.collect()
    mem_before = mem_gb()
    
    try:
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits
        t_fwd = time.time() - t0
        
        ppl = ppl_of(logits, input_ids)
        mem_peak = mem_gb()
        tps = n / t_fwd
        
        results["runs"].append({
            "N": n, "window": window, "n_global": n_global, "sparsity": sp_label,
            "status": "ok", "ppl": round(ppl, 3),
            "time_s": round(t_fwd, 1), "tokens_per_sec": round(tps, 1),
            "mem_before_gb": round(mem_before, 2), "mem_peak_gb": round(mem_peak, 2),
            "mem_delta_gb": round(mem_peak - mem_before, 2),
            "beyond_model_max": n > 32768,
        })
        log(f"  ✅ PPL={ppl:.3f}, 速度={tps:.1f}t/s, 耗时={t_fwd:.1f}s")
        log(f"  内存: {mem_before:.2f}→{mem_peak:.2f}GB (Δ{mem_peak-mem_before:.2f})")
        if n > 32768:
            log(f"  ⚠️ 超模型max_position_embeddings(32768)，RoPE外推，PPL可能不可靠")
        
        del out, logits
        gc.collect()
    except Exception as e:
        log(f"  ❌ {type(e).__name__}: {str(e)[:120]}")
        results["runs"].append({
            "N": n, "window": window, "n_global": n_global, "sparsity": sp_label,
            "status": f"error:{type(e).__name__}", "error": str(e)[:200],
        })
        gc.collect()
        if "memory" in str(e).lower() or "alloc" in str(e).lower():
            log("  ⛔ OOM，停止更大规模")
            restore()
            break

restore()

import json
class NpEnc(json.JSONEncoder):
    def default(self, o):
        try:
            import numpy as np
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.integer,)): return int(o)
        except: pass
        return super().default(o)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p8_sparse_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, cls=NpEnc)
log(f"\n结果保存: p8_sparse_results.json")
log("P8稀疏注意力实验完成。")
