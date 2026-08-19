import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
q = torch.randn(1, 14, 8, 64)
k = torch.randn(1, 2, 8, 64)
cos = torch.randn(1, 8, 64)
sin = torch.randn(1, 8, 64)
out = apply_rotary_pos_emb(q, k, cos, sin)
print("type:", type(out))
print("len:", len(out) if isinstance(out, (tuple, list)) else "not-seq")
if isinstance(out, (tuple, list)):
    for i, o in enumerate(out):
        print("  [%d] shape=%s" % (i, getattr(o, "shape", None)))
