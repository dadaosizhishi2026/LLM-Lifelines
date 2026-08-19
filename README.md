# 大模型的"命根子"：拆掉10%，它崩溃136倍

**We mapped which attention heads are load-bearing. Remove 10% of them — the model breaks 136×.**

大模型里成千上万个注意力头，哪些是承重墙？这篇论文给出了地图——并且用 Qwen2.5-0.5B 和 7B 实测验证：

拆掉识别出的"骨架头"，模型困惑度崩溃 **136倍到5000倍以上**；拆同样比例的冗余头，几乎没影响。当你压缩、剪枝、量化一个模型时，先知道哪堵墙不能拆。

## 这篇讲了什么 / What's inside

- 四种边结构：真实边（骨架）/ 虚拟边（长程捷径）/ 残缺边（盲区诊断）/ 弯曲边（突破死路）
- 稀疏注意力：结构判据全面胜过范数判据，困惑度最低降 **38.3%**
- 骨架识别尺度不变：0.5B 与 7B 行为规律一致
- KV缓存按结构集中度分配：75%保留比最优，降7.9%；反向分配反而升27.2%
- 完整数学公式（7.1–7.5）+ 可移植Python实现（8.1–8.5）

## Files · 文件

| Language | File |
|----------|------|
| 中文 | 哈密顿LLM注意力增强·专业重制版-对外学术-20260819.pdf |
| English | Long-Context-LLM-Attention-Hamiltonian-Virtual-Edges-EN-20260819.pdf |

*Academic exchange only. All thresholds are tunable parameters; full reproduction instructions included.*

*仅供学术交流。全部阈值为可调参数，附完整复现说明。*
