"""冒烟测试：只跑单个4K稀疏配置，确认patch能跑通"""
import sys, os, time, gc, types
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn.functional as F
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p8_sparse_attn import sparse_prefill, make_sparse_attn_forward, make_ids, ppl_of

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def mem(): return psutil.Process(os.getpid()).memory_info().rss/1e9

log("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
    dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="eager")
model.eval()
n_layers = len(model.model.layers)
n_heads = model.config.num_attention_heads
n_kv = model.config.num_key_value_heads
head_dim = model.config.hidden_size // n_heads
log(f"{n_layers}层 {n_heads}头 {n_kv}KV头 head_dim{head_dim}, 内存{mem():.2f}GB")

# patch
fn = make_sparse_attn_forward(410, 41, n_heads, n_kv, head_dim)
for li in range(n_layers):
    attn = model.model.layers[li].self_attn
    attn.forward = types.MethodType(fn, attn)

input_ids = make_ids(4096)
n = input_ids.shape[1]
log(f"输入 {n} tokens, 跑稀疏注意力...")
t0 = time.time()
try:
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits
    ppl = ppl_of(logits, input_ids)
    log(f"✅ 成功! PPL={ppl:.3f}, 耗时={time.time()-t0:.1f}s, 内存{mem():.2f}GB")
except Exception as e:
    import traceback
    log(f"❌ {type(e).__name__}: {e}")
    traceback.print_exc()
