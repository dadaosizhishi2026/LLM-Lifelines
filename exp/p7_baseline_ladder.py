"""
云韶框架·AI特化·P7-基线阶梯
本地小模型上下文极限测试（不压缩基线）
阶梯：1K→2K→4K→8K→16K（或直到内存不够/PPL崩溃）
记录：PPL / 内存 / 速度 / KV Cache大小
约束：CPU only，内存7.3GB硬上限
"""
import sys, os, json, time, gc
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "p7_baseline_ladder.json")

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
)

MEM_LIMIT_GB = 7.3  # 硬上限

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def mem_gb():
    """当前进程内存占用(GB)"""
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / 1e9

def sys_mem_gb():
    """系统可用内存(GB)"""
    return psutil.virtual_memory().available / 1e9

log(f"系统总内存: {psutil.virtual_memory().total/1e9:.1f}GB, 可用: {sys_mem_gb():.1f}GB")
log(f"内存硬上限: {MEM_LIMIT_GB}GB")

log("加载 Qwen2.5-0.5B (CPU, eager)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True,
    dtype=torch.float32, low_cpu_mem_usage=True,
    attn_implementation="eager",
)
model.eval()
log(f"模型加载后内存: {mem_gb():.2f}GB")

# ── 构造可重复扩展的长文本 ──
# 用真实感文本块重复扩展到目标长度
BASE_BLOCK = """The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures. By using these cliques as outer embryos to wrap original nodes, the original graph can be compressed into a smaller compressed graph. After solving the Hamiltonian path on the compressed graph and expanding back, the number of virtual edges remains stable at zero to two, with coverage above ninety-nine percent. In mixture of experts language models, each token is independently routed to a small subset of specialized expert networks through a learned gating mechanism. The routing decisions create a dynamic bipartite graph between tokens and experts that changes with every forward pass. The algebraic tension framework provides a principled way to diagnose which routing configurations are structurally stable and which are prone to collapse under distribution shift. """

def make_text(target_tokens):
    """生成约target_tokens长度的文本"""
    # 估算：每块约150 tokens
    n_blocks = max(1, int(target_tokens / 150) + 2)
    text = BASE_BLOCK * n_blocks
    return text

# ── 阶梯实验 ──
LADDER = [1024, 2048, 4096, 8192, 16384, 32768]

results = {
    "model": "Qwen2.5-0.5B",
    "mem_limit_gb": MEM_LIMIT_GB,
    "model_mem_gb": round(mem_gb(), 2),
    "dtype": "float32",
    "attn": "eager",
    "ladder": []
}

for target in LADDER:
    log(f"\n{'═'*50}")
    log(f"目标: {target} tokens")
    log(f"{'═'*50}")
    
    text = make_text(target)
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=target)["input_ids"]
    n = input_ids.shape[1]
    log(f"  实际tokens: {n}")
    
    # 内存预估：KV Cache = 2 × n_layers × n_heads × n × head_dim × 4bytes
    n_layers = len(model.model.layers)
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    kv_mem_gb = 2 * n_layers * n_heads * n * head_dim * 4 / 1e9
    # 注意力矩阵（eager）= n_layers × n_heads × n × n × 4bytes（峰值）
    attn_mem_gb = n_layers * n_heads * n * n * 4 / 1e9
    log(f"  预估KV Cache: {kv_mem_gb:.2f}GB")
    log(f"  预估注意力矩阵峰值: {attn_mem_gb:.2f}GB")
    
    # 如果预估超过内存上限，跳过（避免OOM崩整个进程）
    if mem_gb() + kv_mem_gb + attn_mem_gb > MEM_LIMIT_GB * 1.3:
        log(f"  ⛔ 预估总内存超限（{mem_gb()+kv_mem_gb+attn_mem_gb:.1f}GB > {MEM_LIMIT_GB*1.3:.1f}GB），跳过")
        results["ladder"].append({
            "target_tokens": target,
            "actual_tokens": n,
            "status": "skipped_mem_estimate",
            "est_kv_gb": round(kv_mem_gb, 2),
            "est_attn_gb": round(attn_mem_gb, 2),
        })
        continue
    
    gc.collect()
    mem_before = mem_gb()
    
    try:
        # 测速 + PPL
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits
        t_forward = time.time() - t0
        
        # PPL
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = torch.nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        ppl = torch.exp(loss).item()
        
        mem_after = mem_gb()
        tokens_per_sec = n / t_forward
        
        entry = {
            "target_tokens": target,
            "actual_tokens": n,
            "status": "ok",
            "ppl": round(ppl, 3),
            "forward_time_s": round(t_forward, 2),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "mem_before_gb": round(mem_before, 2),
            "mem_after_gb": round(mem_after, 2),
            "mem_delta_gb": round(mem_after - mem_before, 2),
            "est_kv_gb": round(kv_mem_gb, 2),
        }
        results["ladder"].append(entry)
        
        log(f"  ✅ PPL={ppl:.3f}, 速度={tokens_per_sec:.1f}t/s, 耗时={t_forward:.1f}s")
        log(f"  内存: {mem_before:.2f}→{mem_after:.2f}GB (Δ{mem_after-mem_before:.2f})")
        
        # 释放
        del out, logits, shift_logits
        gc.collect()
        
    except Exception as e:
        log(f"  ❌ 失败: {type(e).__name__}: {str(e)[:100]}")
        results["ladder"].append({
            "target_tokens": target,
            "actual_tokens": n,
            "status": f"error: {type(e).__name__}",
            "error": str(e)[:200],
            "est_kv_gb": round(kv_mem_gb, 2),
        })
        gc.collect()
        # 如果是OOM，停止后续更大的
        if "memory" in str(e).lower() or "alloc" in str(e).lower():
            log(f"  ⛔ OOM，停止更大规模测试")
            break

# ── 保存 ──
with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
log(f"\n结果保存: {RESULTS}")
log("P7基线阶梯完成。")
