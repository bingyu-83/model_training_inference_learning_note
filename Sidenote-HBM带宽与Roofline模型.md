# Sidenote：HBM 带宽比算力更稀缺 × Roofline 模型深度展开

> **对应原文位置**：`Week1-2-学习总结.md` → 一、最大的认知转变 → "HBM 带宽比算力更稀缺" 段落
>
> 本文是该段落的 deep dive 注释，展开每一个需要理解清楚的知识点。

---

## 原文回顾

> HBM 带宽比算力更稀缺。H100 的 FP16 算力是 990 TFLOPS，但 HBM 带宽只有 3.35 TB/s。一个简单的矩阵乘法，理论上需要的带宽远超实际可用带宽，这就是为什么大多数 kernel 是 memory-bound 而不是 compute-bound。Roofline 模型把这个关系说得很清楚——拐点左边优化内存访问，拐点右边才轮到优化算法。

---

## Deep Dive 1：为什么说带宽"比算力更稀缺"

两个数字的量纲不同，直接比没有意义，需要换算到同一维度：

```
H100 峰值算力：990 TFLOPS = 990 × 10¹² FLOP/s
H100 HBM 带宽：3.35 TB/s  = 3.35 × 10¹² Byte/s
```

做一次 FP16 FMA（乘加）需要读 2 个操作数，各 2 bytes：

```
如果要让 Tensor Core 一直不闲着，需要的喂数速度：
  990 × 10¹² FLOP/s × 2 Byte/FLOP = 1,980 TB/s

实际 HBM 只能提供：3.35 TB/s

差距：1,980 / 3.35 ≈ 590 倍
```

**结论**：HBM 带宽只能喂饱 Tensor Core 算力的 0.17%。
绝大多数时间 Tensor Core 都在等数据，而不是在计算。

### 历代 GPU 的算力/带宽剪刀差（越来越大）

| GPU | FP16 算力 | HBM 带宽 | 拐点（FLOP/Byte） |
|-----|---------|---------|-----------------|
| A100 | 312 TFLOPS | 2.0 TB/s | ~156 |
| H100 | 990 TFLOPS | 3.35 TB/s | ~296 |
| H200 | 990 TFLOPS | 4.8 TB/s | ~206 |
| B200 | ~2250 TFLOPS | 8.0 TB/s | ~281 |

> **反直觉点**：H200 的拐点比 H100 低（206 vs 296），意味着更容易达到 compute-bound。H200 的重点升级是带宽，不是算力。B200 两者都大幅提升，但拐点依然在 ~280，memory-bound 问题并没有消失。

---

## Deep Dive 2：算术强度（Arithmetic Intensity）

**定义**：一个 kernel 每从内存读取 1 Byte 数据，能做多少次浮点运算。

```
算术强度 I = FLOP / Byte
```

这是判断一个 kernel 是 memory-bound 还是 compute-bound 的核心指标。

### 直觉理解

```
算术强度低（如 I = 1）：
  读 1 个数，做 1 次加法，写回去
  → 数据搬运量 >> 计算量
  → 带宽是瓶颈，Tensor Core 大量空转

算术强度高（如 I = 1000）：
  读 1 个数，用它参与 1000 次运算
  → 计算量 >> 数据搬运量
  → 算力是瓶颈，带宽绰绰有余
```

### 典型 LLM Kernel 的算术强度

| Kernel | 算术强度估算 | 原因 |
|--------|------------|------|
| Elementwise（加/乘/激活）| ~1 FLOP/Byte | 读一个数，做一次运算，写回 |
| Softmax | ~3-5 FLOP/Byte | 读一行，做 exp+sum+除法 |
| LayerNorm | ~5-10 FLOP/Byte | 读一行，做均值+方差+归一化 |
| KV Cache 读取（decode） | <1 FLOP/Byte | 读大量历史 KV，只做少量运算 |
| GEMV（decode 推理） | ~N/2 FLOP/Byte | N 通常 4096-8192，仍偏低 |
| Flash Attention（prefill）| ~50-200 FLOP/Byte | 分块复用数据，接近拐点 |
| 大批量 GEMM（训练）| >500 FLOP/Byte | 矩阵越大，复用越充分 |

> **关键洞察**：Decode 阶段每步只生成 1 个 token，GEMM 退化为 GEMV（矩阵×向量），算术强度急剧下降。这就是"Decode is memory-bandwidth-bound"的根本原因，也是为什么 Continuous Batching 和更大 batch size 能提升 GPU 利用率——增大 batch 让 GEMV 变回 GEMM。

---

## Deep Dive 3：Roofline 模型

### 核心思想

Roofline 是一个性能上界模型，给定硬件参数，任何 kernel 的性能都不能超过两条线：

```
1. 内存带宽线（斜线）：性能 ≤ 带宽 × 算术强度
   → 受 HBM 搬运速度限制

2. 峰值算力线（水平线）：性能 ≤ 峰值 TFLOPS
   → 受 Tensor Core 计算速度限制

实际性能 = min(带宽 × I, 峰值算力)
```

### 图形结构

```
性能 (TFLOPS)
   │
990├────────────────────────────── 峰值算力（水平天花板）
   │                      ╱‾‾‾‾‾‾‾
   │                   ╱  ← 拐点 ~296 FLOP/Byte
   │                ╱
   │             ╱   斜率 = HBM 带宽 3.35 TB/s
   │          ╱
   │       ╱
   └───────────────────────────────→ 算术强度 (FLOP/Byte)
          memory-bound  │  compute-bound
```

### 拐点（Ridge Point）计算

```
拐点算术强度 = 峰值算力 / 峰值带宽
H100: 990 TFLOPS / 3.35 TB/s ≈ 296 FLOP/Byte
```

含义：kernel 的算术强度 < 296 → memory-bound；> 296 → compute-bound。

### 用 Roofline 指导优化决策

```
kernel 在拐点左边（memory-bound）：
  → 提升算力没用（带宽已经是瓶颈）
  → 正确方向：减少 HBM 读写
    ✓ Kernel Fusion：把多个 kernel 合并，中间结果留在 SRAM
    ✓ Flash Attention：分块计算，避免写回完整 attention 矩阵
    ✓ 量化：FP8/INT4 数据更小，同等带宽能搬更多数

kernel 在拐点右边（compute-bound）：
  → 优化内存访问没用（算力已经是瓶颈）
  → 正确方向：提升 Tensor Core 利用率
    ✓ 更大 tile size，减少 warp 空转
    ✓ 使用低精度（FP8 算力是 FP16 的 2×）
    ✓ 减少算法冗余 FLOP（如 sparse attention）
```

---

## Deep Dive 4：为什么 Kernel Fusion 有效

直觉模型：每次 kernel 启动都需要从 HBM 读数据、计算完、写回 HBM。

```
未融合（3 个独立 kernel）：
  HBM → [Linear] → HBM → [GeLU] → HBM → [LayerNorm] → HBM
  4 次 HBM 读写，但有效计算量很小 → 算术强度极低

融合后（1 个 kernel）：
  HBM → [Linear + GeLU + LayerNorm] → HBM
  2 次 HBM 读写，中间结果在 SRAM（L1/shared memory）里传递
  → 算术强度 3× 提升，在 Roofline 上向右移动
```

**SRAM vs HBM 带宽对比**：

| 内存层级 | H100 带宽 | 延迟 |
|---------|---------|------|
| Register | ~80 TB/s | ~1 cycle |
| L1/Shared Memory（SRAM）| ~33 TB/s | ~20 cycles |
| L2 Cache | ~12 TB/s | ~200 cycles |
| HBM | ~3.35 TB/s | ~600 cycles |

Kernel Fusion 把数据访问从 HBM 推到 SRAM，带宽提升 10 倍，这就是 Flash Attention 快的根本原因。

---

## Deep Dive 5：Flash Attention 为什么把算术强度推高

标准 Attention 的问题：

```
标准实现：
  Q, K, V ∈ HBM
  Step 1: S = Q × Kᵀ        → 写 S (N×N) 到 HBM  ← 巨大的中间矩阵！
  Step 2: P = softmax(S)     → 读 S，写 P 到 HBM
  Step 3: O = P × V          → 读 P，写 O 到 HBM

N=4096 时，S 矩阵 = 4096 × 4096 × 2 bytes = 32 MB（每层每次！）
HBM 读写次数：O(N²)，算术强度极低
```

Flash Attention 的解法：

```
分块（Tiling）：把 Q, K, V 切成小块，每次只加载一块到 SRAM
  → 在 SRAM 里完成 softmax 的数值稳定计算（online softmax）
  → 永远不把 S 矩阵写回 HBM
  → HBM 读写降为 O(N)，算术强度大幅提升

代价：需要两次 pass（forward 不存 S，backward 重新计算）
收益：HBM 访问减少 5-20×，对 memory-bound 的 attention 提升显著
```

---

## 总结：一图串联所有概念

```
HBM 带宽（3.35 TB/s）远小于算力需要（~1980 TB/s）
          ↓
大多数 kernel 算术强度低（<10 FLOP/Byte）
          ↓
实际性能远低于 Roofline 的计算上界 → memory-bound
          ↓
优化方向：减少 HBM 读写（Kernel Fusion / Flash Attention / 量化）
          ↓
算术强度提升 → 在 Roofline 图上向右移动 → 更接近峰值算力
```

---

## 延伸阅读

- **Roofline 原始论文**：[Roofline: An Insightful Visual Performance Model (2008)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- **GPU MODE Lecture 4**：[Compute and Memory Basics](https://youtu.be/lTmYrKwjSOU) — 从 SM 架构讲到内存层次，直接覆盖本文内容
- **GPU MODE Lecture 12**：[Flash Attention](https://youtu.be/zEuwuCTEf_0) — Flash Attention 算法深度讲解
- **Flash Attention 原始论文**：[FlashAttention: Fast and Memory-Efficient Exact Attention (2022)](https://arxiv.org/abs/2205.14135)
