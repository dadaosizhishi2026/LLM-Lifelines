"""
云韶框架·AI特化·P7b-SDPA可行性
验证：SDPA attention在CPU上能否跑长上下文（O(n)内存，不实体化n×n矩阵）
不用DirectML，纯CPU SDPA
"""
import sys, os, time, gc
sys.stdout.reconfigure(encoding='utf-8')

import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

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
log(f"模型加载后内存: {mem_gb():.2f}GB")

BASE_BLOCK = """The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures. By using these cliques as outer embryos to wrap original nodes, the original graph can be compressed into a smaller compressed graph. After solving the Hamiltonian path on the compressed graph and expanding back, the number of virtual edges remains stable at zero to two, with coverage above ninety-nine percent. In mixture of experts language models, each token is independently routed to a small subset of specialized expert networks through a learned gating mechanism. The routing decisions create a dynamic bipartite graph between tokens and experts that changes with every forward pass. The algebraic tension framework provides a principled way to diagnose which routing configurations are structurally stable and which are prone to collapse under distribution shift. """

def make_text(target_tokens):
    n_blocks = max(1, int(target_tokens / 150) + 2)
    return BASE_BLOCK * n_blocks

# 测试SDPA能跑多长
for target in [4096, 8192, 16384, 32768]:
    log(f"\n── SDPA {target} tokens ──")
    text = make_text(target)
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=target)["input_ids"]
    n = input_ids.shape[1]
    log(f"  实际: {n} tokens, 当前内存: {mem_gb():.2f}GB")
    
    gc.collect()
    mem_before = mem_gb()
    try:
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits
        t_fwd = time.time() - t0
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = torch.nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        ppl = torch.exp(loss).item()
        
        mem_after = mem_gb()
        log(f"  ✅ PPL={ppl:.3f}, 速度={n/t_fwd:.1f}t/s, 耗时={t_fwd:.1f}s, 内存Δ={mem_after-mem_before:.2f}GB")
        
        del out, logits, shift_logits
        gc.collect()
    except Exception as e:
        log(f"  ❌ {type(e).__name__}: {str(e)[:120]}")
        gc.collect()
        if "memory" in str(e).lower() or "alloc" in str(e).lower():
            log("  ⛔ OOM停止")
            break

log("\nP7b SDPA可行性测试完成。")
