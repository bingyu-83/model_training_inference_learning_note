---
title: "Week1-2 Quick Start 手册"
tags: [GCP, AWS, GPU, ramp-up, quick-start]
category: GCP
status: active
date: 2026-07-19
type: 计划
cssclasses: [max]
---

# Week1-2 Quick Start 手册

---

## 对标 Week 1-2 Ramp-up 计划

| Week 1-2 原始条目 | 对应手册章节 | 库内辅助文档 |
|------------------|----------------|----------------|
| GPU MODE 频道 | 第一块、第三块、第四块 | [[MLOPS/GPU/GCP/Week1-2-GPU-NVIDIA/GPU-MODE视频推荐清单\|GPU-MODE视频推荐清单]] |
| Intra-node Connectivity / Fabric Manager / NVSwitch | 第二块 | [[MLOPS/GPU/01-GPU架构与硬件/8张独立H100显卡为何在服务器里能化身一张超级大显卡\|8张 H100 化身一张超级大显卡]] / [[MLOPS/GPU/01-GPU架构与硬件/没有基础也能看懂GPU之间如何通信\|GPU之间如何通信]] |
| Cluster / NVDomain / Global Fabric Manager / GPM | 第二块 | [[MLOPS/GPU/02-GPU通信与网络/写在理解Mooncake拓扑感知路由之前彻底理解NUMA和NUMA亲和性每一名AIInfra都应该了解的底层原理\|彻底理解 NUMA 与拓扑]] |
| NCCL / NCCL Learning | 第三块 | [[MLOPS/GPU/02-GPU通信与网络/一文看懂NCCL八种集合通信从 AllReduce到AllToAll\|一文看懂 NCCL 八种集合通信]] / [[MLOPS/GPU/02-GPU通信与网络/NCCL-Case-Complete\|NCCL Case]] |
| MultiGPU + NCCL from the authors | 第三块 | [[MLOPS/GPU/02-GPU通信与网络/为什么模型并行策略会决定网络压力\|并行策略与网络压力]] |
| Product Evolution：Ampere / Hopper / Blackwell | 第一块 | [[MLOPS/NVIDIATensorCore演进从Volta到Blackwell\|Tensor Core 演进]] / [[MLOPS/GPU/01-GPU架构与硬件/一文讲明白大模型显存占用只考虑单卡\|大模型显存占用]] |
| NCP Certification | 读完白皮书后备考 | [[MLOPS/GPU/GCP/Week1-2-GPU-NVIDIA/NCP认证报考指南\|NCP认证报考指南]] / [[MLOPS/GPU/04-NVIDIA认证/一文分清NVIDIA三大专家认证AIIAIOAIN拿捩AI基建时代职场红利\|NVIDIA 三大认证分清]] |

---

## 时间分配总览

```
第一块：GPU 硬件骨架
第二块：节点内通信
第三块：NCCL + 并行策略
──────────────── 休息 ────────────────
第四块：模型原理 + KV Cache
第五块：GCP 平台特有知识
第六块：运维排障速查
```

---

## 第一块：GPU 硬件骨架

**目标：能回答"H100 比 A100 快在哪"**

### 核心心智模型

GPU 的性能由三件事决定：**算多快（Tensor Core）、存多少（HBM）、搬多快（带宽）**

```
算力：Tensor Core
  A100: 312 TFLOPS (FP16)
  H100: 990 TFLOPS (FP16) → 3x
  B200: 2250 TFLOPS (FP16) → 7x

存储：HBM
  A100: 80GB HBM2e，2.0 TB/s
  H100: 80GB HBM3，3.35 TB/s
  B200: 192GB HBM3e，8 TB/s

节点内通信：NVLink
  A100: 600 GB/s
  H100: 900 GB/s
  B200: 1.8 TB/s
```

### 三个关键特性记住就够

| 特性                       | 哪代引入             | 一句话解释                |
| ------------------------ | ---------------- | -------------------- |
| MIG                      | Ampere (A100)    | 一张卡切成 7 个独立小卡，多租户推理用 |
| Transformer Engine + FP8 | Hopper (H100)    | 自动混合精度，训练吞吐 3x       |
| 双 Die + FP4              | Blackwell (B200) | 两个芯片封装成一个，算力再翻倍      | [[Blackwell-B200-双Die与FP4解析\|深读：B200 为何比 H200 更强]] |

### GCP 机型对应

| GPU | GCP 机型 | 节点间网络 |
|-----|---------|-----------|
| A100 | A2 | 标准 VPC |
| H100 | A3 / A3+ | TCPDirect / TCPXO |
| B200 | A4（即将） | RDMA/RoCE |

**📖 深读：** [[MLOPS/GPU/GCP/Week1-2-GPU-NVIDIA/GPU架构演进-Ampere-Hopper-Blackwell|GPU架构演进-Ampere-Hopper-Blackwell.md]]

**🔗 原始资料：**
- [Ampere 白皮书 PDF](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- [Hopper 白皮书 PDF](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf)
- [Blackwell 白皮书 PDF](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- [GPU MODE Lecture 4: Compute & Memory Basics](https://www.youtube.com/watch?v=lTmYrKwjSOU)（SM 架构 + 内存层次）
- [GPU MODE Lecture 8: CUDA Performance Checklist](https://www.youtube.com/watch?v=SGhfUhlowB4)（理解 GPU 为什么快/慢）

> 📌 **对标 Week 1-2：** Product Evolution — Ampere / Hopper / Blackwell
> GPU MODE: https://www.youtube.com/@GPUMODE/videos

**📚 库内辅助文档：**
- [[MLOPS/GPU/01-GPU架构与硬件/CUDA从入门到精通一篇讲透GPU并行编程性能优化与工程实践|CUDA从入门到精通一篇讲透GPU并行编程性能优化与工程实践]] — SM架构、并行编程、性能优化
- [[MLOPS/GPU/01-GPU架构与硬件/一文讲明白大模型显存占用只考虑单卡|一文讲明白大模型显存占用只考虑单卡]] — HBM、显存层次、Tensor Core
- [[MLOPS/NVIDIATensorCore演进从Volta到Blackwell|NVIDIA Tensor Core 演进：从 Volta 到 Blackwell]] — Tensor Core 各代演进深度解析

---

## 第二块：节点内通信

**目标：能解释"为什么 8 卡 H100 训练比 8 卡 A100 快不止 3 倍"**

### 三层结构

```
NCCL（软件）
    ↓
Fabric Manager（管理，守护进程）
    ↓
NVSwitch × 4 + NVLink × 18/GPU（硬件）
```

### 核心类比

- **NVLink** = GPU 之间的高速公路（H100: 900 GB/s，PCIe 只有 64 GB/s，差 14 倍）
- **NVSwitch** = 高速公路的立交桥（让任意两个 GPU 直连，不用绕路）
- **Fabric Manager** = 交通管理中心（初始化路由表，没它 NVSwitch 不工作）

### 最重要的一条排障知识

```
Fabric Manager 没启动 → NVSwitch 未初始化 → NCCL fallback 到 TCP
症状：nccl-tests busbw 只有几 GB/s（正常应该几百 GB/s）
修复：systemctl start nvidia-fabricmanager
```

### 拓扑查看

```bash
nvidia-smi topo -m
# NV18 = 通过 18 条 NVLink 全速连接（正常）
# SYS  = 通过 PCIe（慢，异常）
```

**📖 深读：** [[MLOPS/GPU/GCP/Week1-2-GPU-NVIDIA/NVLink-NVSwitch-FabricManager|NVLink-NVSwitch-FabricManager.md]]

**🔗 视频：**
- [MultiGPU + NCCL（作者讲解）](https://www.youtube.com/watch?v=2xMzQ1Z2Qe0) — 必看，NCCL 作者亲自讲多卡通信原理

> 📌 **对标 Week 1-2：** Intra-node Connectivity — Fabric Manager / NVSwitch / all-to-all communication
> Cluster — NVDomain / Global Fabric Manager / Google GPU Pod Manager (GPM)

**📚 库内辅助文档：**
- [[MLOPS/GPU/01-GPU架构与硬件/8张独立H100显卡为何在服务器里能化身一张超级大显卡|8张独立 H100 显卡为何在服务器里能化身一张超级大显卡]] — NVLink/NVSwitch 全互联拓扑详解
- [[MLOPS/GPU/01-GPU架构与硬件/没有基础也能看懂GPU之间如何通信|没有基础也能看懂GPU之间如何通信]] — GPU 节点内通信入门
- [[MLOPS/GPU/02-GPU通信与网络/写在理解Mooncake拓扑感知路由之前彻底理解NUMA和NUMA亲和性每一名AIInfra都应该了解的底层原理|彻底理解 NUMA 和 NUMA 亲和性]] — 节点内拓扑感知、与 NVDomain 相关

---

## 第三块：NCCL + 并行策略

**目标：能解释"为什么训练慢，瓶颈在哪"**

### NCCL 五大原语（只需记住三个）

| 原语 | 一句话 | 用在哪 |
|------|--------|--------|
| **AllReduce** | 大家各出一份，最后人人得到总和 | DDP 梯度同步 |
| **AllGather** | 大家各出一片，最后人人得到全量 | ZeRO/FSDP 前向收集参数 |
| **ReduceScatter** | 大家各出全量，最后每人只得一片归约结果 | ZeRO/FSDP 反向分散梯度 |
| AlltoAll | 每人给每人发一片 | MoE Expert Parallel |
| Broadcast | 一人发给所有人 | 初始化参数 |

> AllReduce = ReduceScatter + AllGather（这个等式很重要）

### 并行策略与通信的对应

```
DDP（数据并行）
  → AllReduce 梯度
  → 走节点间网络（IB/RoCE/TCPDirect）
  → 瓶颈：跨节点带宽

TP（张量并行）
  → AllReduce / AllGather + ReduceScatter
  → 必须走节点内 NVLink（延迟敏感）
  → 瓶颈：NVLink 带宽

PP（流水线并行）
  → Send/Recv 激活值
  → 走节点间网络
  → 瓶颈：bubble（流水线气泡）
```

### 排障口诀

```
训练慢 → 先跑 nccl-tests
  busbw 低 → 通信瓶颈
    节点内低 → 查 Fabric Manager / NVLink
    节点间低 → 查网卡 / MTU / 路由
  busbw 正常 → 计算瓶颈 → 用 Nsight 分析
```

**📖 深读：** [[MLOPS/GPU/GCP/Week1-2-GPU-NVIDIA/NCCL集合通信原理|NCCL集合通信原理.md]]

**🔗 视频：**
- [GPU MODE Lecture 9: Reductions](https://www.youtube.com/watch?v=09wntC6BT5o)（并行归约原理，AllReduce 的底层基础）
- [GPU MODE Lecture 13: Ring Attention](https://www.youtube.com/watch?v=ZrhaHFPMXcA)（Ring 通信机制，直接关联 NCCL Ring AllReduce）

> 📌 **对标 Week 1-2：** NCCL / NCCL Learning
> MultiGPU + NCCL from the authors: https://www.youtube.com/watch?v=2xMzQ1Z2Qe0

**📚 库内辅助文档：**
- [[MLOPS/GPU/02-GPU通信与网络/一文看懂NCCL八种集合通信从AllReduce到AllToAll|一文看懂 NCCL 八种集合通信从 AllReduce 到 AllToAll]] — 五大原语全解析
- [[MLOPS/GPU/02-GPU通信与网络/NCCL-Case-Complete|NCCL Case Complete]] — NCCL 实战案例
- [[MLOPS/GPU/02-GPU通信与网络/为什么模型并行策略会决定网络压力|为什么模型并行策略会决定网络压力]] — 并行策略与 NCCL 通信的关系
- [[MLOPS/GPU/02-GPU通信与网络/大模型训练和推理中的DPTPPPEP理解|大模型训练和推理中的 DP TP PP EP 理解]] — 并行策略与 NCCL 原语的对应关系

---

## 第四块：模型原理 + KV Cache

**目标：能解释客户的 OOM 和推理慢问题**

### Transformer 最小心智模型

```
输入 tokens
    ↓
每个 token → Q、K、V 三个向量
    ↓
Attention: 当前 token 的 Q × 所有历史 token 的 K → 注意力权重
         → 权重 × 所有历史 token 的 V → 输出
    ↓
输出下一个 token（自回归，一次一个）
```

### KV Cache：为什么存在

```
没有 KV Cache：
  生成第 100 个 token → 重新计算前 99 个 token 的 K/V → O(N²) 计算

有 KV Cache：
  把算过的 K/V 存起来 → 每步只算新 token → O(N) 计算
  代价：显存随序列长度线性增长
```

### 显存占用快速估算

```
训练显存 ≈ 模型参数 × 16（混合精度 + 优化器状态）
  7B 模型 ≈ 7B × 16 bytes ≈ 112 GB

推理显存 = 模型权重 + KV Cache
  模型权重：7B × 2 bytes (FP16) ≈ 14 GB
  KV Cache：2 × layers × num_kv_heads × head_dim × seq_len × batch × 2 bytes
```

### OOM 快速判断

```
OOM 了 → 问三个问题：
1. 是训练还是推理？
   训练 → 先看 batch size / gradient checkpointing
   推理 → 先看 KV Cache（seq_len × batch 是否太大）

2. 模型多大？用什么精度？
   FP16/BF16 → 参数量 × 2 bytes
   INT8 → 参数量 × 1 byte

3. 并行策略是什么？
   TP=8 → 每卡只存 1/8 模型权重
   没有 TP → 每卡存完整模型
```

### vLLM 的核心贡献

PagedAttention：把 KV Cache 从"提前分配一大块连续显存"改成"按需分配固定大小的 block"
- 显存利用率：20-40% → 96%+
- 原理类比：操作系统的虚拟内存分页

**🔗 视频：**
- [GPU MODE Lecture 12: Flash Attention](https://www.youtube.com/watch?v=l8pRSuU81PU)（Attention 优化核心，理解 LLM 训练/推理性能）

---

## 第五块：GCP 平台特有知识

**目标：能解释 GCP 和 AWS 的差异，能回答 GCP 特有的客户问题**

### AI Hypercomputer 三件套

| 组件 | 作用 | AWS 对应 |
|------|------|---------|
| gSC（GPU Supercomputer） | 计算+网络+存储一体化集群 | SageMaker HyperPod |
| DWS（Dynamic Workload Scheduler） | 按需调度 GPU，不用预留 | 无直接对应 |
| Topology-Aware Placement | 把通信密集的 GPU 放在同一 rack | EFA placement group |

### GCP 网络栈（最重要的差异点）

```
AWS：EFA（所有 GPU 实例统一用 EFA）

GCP：按机型不同，用不同协议
  A3  (H100) → TCPDirect  （内核旁路 TCP）
  A3+ (H100) → TCPXO      （TCP + GPU Direct）
  A4  (B200) → RDMA/RoCE  （原生 RDMA，最快）
```

客户问"为什么 A3+ 比 A3 快"→ 网络协议升级，TCPXO 支持 GPU Direct，减少 CPU 参与

### GKE 上跑 GPU 的关键概念

```bash
# GPU 资源声明
resources:
  limits:
    nvidia.com/gpu: 8  # 请求 8 个 GPU

# 查看 GPU 节点
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-h100-80gb

# 查看 NCCL 插件
kubectl get daemonset -n kube-system | grep nccl
```

### DWS vs Reservation 选择逻辑

```
短期突发任务（<1天）→ DWS（按需，贵但灵活）
长期稳定训练（>1月）→ Reservation（便宜但要提前预留）
中期项目（1周-1月）→ CUD（承诺使用折扣）
```

---

## 第六块：运维排障速查

**目标：遇到客户问题，知道从哪里开始查**

### Health Check 三层模型（记住这个框架）

```
Layer 3: 跑推理/训练压测 → 验证真实工作负载
Layer 2: 跑 nccl-tests   → 验证通信正常
Layer 1: 跑 DCGM 诊断    → 验证硬件健康
```

从下往上查，Layer 1 有问题先修 Layer 1。

### 最常见的五个场景

**场景 1：NCCL hang 住**
```bash
export NCCL_DEBUG=INFO
# 看日志找哪对 GPU 通信卡住
# 检查 nvidia-smi nvlink -e（NVLink 错误）
# 检查 systemctl status nvidia-fabricmanager
```

**场景 2：训练 OOM**
```
→ 减小 batch size
→ 开启 gradient checkpointing
→ 用 LoRA 代替全参微调
→ 增加 TP（张量并行）分摊显存
```

**场景 3：推理 OOM**
```
→ 减小 max_seq_len 或 max_batch_size
→ 用 INT8/INT4 量化
→ 用 vLLM 的 PagedAttention（比 HuggingFace 原生省很多）
```

**场景 4：XID 错误**
```
XID 31/48 → 硬件 ECC 错误 → 触发 BoH 流程，可能需要换卡
XID 79    → GPU 掉总线    → 检查 PCIe 连接，重启节点
XID 94/95 → NVLink 错误  → 检查 NVLink，可能需要换卡
```

**场景 5：GPU 利用率低**
```
→ 先看是 compute bound 还是 memory bound
→ nvidia-smi dmon -s u（实时利用率）
→ 利用率低 + 显存带宽高 → memory bound → 考虑量化/增大 batch
→ 利用率低 + 显存带宽低 → CPU/数据加载瓶颈 → 检查 DataLoader
```

### 常用命令速查

```bash
# GPU 状态
nvidia-smi                          # 基础状态
nvidia-smi dmon -s u                # 实时利用率
nvidia-smi topo -m                  # 拓扑
nvidia-smi nvlink -e                # NVLink 错误

# Fabric Manager
systemctl status nvidia-fabricmanager
journalctl -u nvidia-fabricmanager -f

# DCGM
dcgmi diag -r 1                     # 快速诊断
dcgmi diag -r 3                     # 完整诊断

# NCCL 测试
./all_reduce_perf -b 8 -e 256M -f 2 -g 8
# 看 busbw：H100 节点内应该 >600 GB/s

# GKE
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-h100-80gb
kubectl describe node <node-name> | grep -A5 Allocatable
```

---

## 所有外部链接汇总

### 白皮书

| 文档 | 链接 |
|------|------|
| Ampere 架构白皮书 | [PDF](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) |
| Hopper 架构白皮书 | [PDF](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf) |
| Blackwell 架构白皮书 | [PDF](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) |

### 必看视频

| 视频 | 链接 | 时长 | 对应章节 |
|------|------|------|----------|
| MultiGPU + NCCL（作者讲解） | [YouTube](https://www.youtube.com/watch?v=2xMzQ1Z2Qe0) | 第二块 |
| GPU MODE Lecture 4: Compute & Memory Basics | [YouTube](https://www.youtube.com/watch?v=lTmYrKwjSOU) | 第一块 |
| GPU MODE Lecture 8: CUDA Performance Checklist | [YouTube](https://www.youtube.com/watch?v=SGhfUhlowB4) | 第一块 |
| GPU MODE Lecture 9: Reductions | [YouTube](https://www.youtube.com/watch?v=09wntC6BT5o) | 第三块 |
| GPU MODE Lecture 12: Flash Attention | [YouTube](https://www.youtube.com/watch?v=l8pRSuU81PU) | 第四块 |
| GPU MODE Lecture 13: Ring Attention | [YouTube](https://www.youtube.com/watch?v=ZrhaHFPMXcA) | 第三块 |

### GPU MODE 资源

| 资源 | 链接 |
|------|------|
| YouTube 频道 | [youtube.com/@GPUMODE](https://www.youtube.com/@GPUMODE/videos) |
| GitHub 讲座代码 | [github.com/cuda-mode/lectures](https://github.com/gpu-mode/lectures) |
| 第三方高质量笔记 | [christianjmills.com](https://christianjmills.com/series/notes/cuda-mode-notes.html) |

### GCP 文档

| 文档 | 链接 |
|------|------|
| Introduction to Cloud TPU | [docs.cloud.google.com](https://docs.cloud.google.com/tpu/docs/intro-to-tpu) |
| TPU system architecture | [docs.cloud.google.com](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) |
| Cloud TPU Multislice Overview | [docs.cloud.google.com](https://docs.cloud.google.com/tpu/docs/multislice-introduction) |
| Troubleshoot Cloud TPU workflow | [docs.cloud.google.com](https://docs.cloud.google.com/tpu/docs/troubleshooting/troubleshooting) |
| Multi-Tier Checkpointing on GKE | [docs.cloud.google.com](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/training/multi-tier-checkpointing) |
| Training ResNet50 on TPU with PyTorch | [docs.cloud.google.com](https://docs.cloud.google.com/tpu/docs/tutorials/resnet-pytorch) |

### NVIDIA 认证

| 资源 | 链接 |
|------|------|
| NVIDIA 认证官网（NCP-AIN） | [nvidia.com/certification](https://www.nvidia.com/en-us/learn/certification/ai-networking-professional/) |

---

## Week1-2 结束后的状态检验

能回答以下问题，说明骨架建立成功：

- [ ] H100 比 A100 快在哪三个维度？
- [ ] Fabric Manager 没启动会发生什么？
- [ ] AllReduce 和 ReduceScatter 的区别？
- [ ] TP 并行为什么必须走 NVLink？
- [ ] KV Cache 的显存随什么增长？
- [ ] GCP A3 和 A3+ 的网络协议有什么不同？
- [ ] 客户说训练 OOM，你第一个问什么？
- [ ] XID 31 意味着什么？

---

## 深读顺序

| 优先级 | 文件 | 预计时间 |
|--------|------|---------|
| ⭐⭐⭐ | [[MLOPS/GPU/GCP/Week1-2-GPU-NVIDIA/NCCL集合通信原理|NCCL集合通信原理.md]] |
| ⭐⭐⭐ | [[MLOPS/GPU/GCP/Week1-2-GPU-NVIDIA/GPU架构演进-Ampere-Hopper-Blackwell|GPU架构演进-Ampere-Hopper-Blackwell.md]] |
| ⭐⭐⭐ | [[MLOPS/GPU/GCP/Week1-2-GPU-NVIDIA/NVLink-NVSwitch-FabricManager|NVLink-NVSwitch-FabricManager.md]] |
| ⭐⭐⭐ | [[MLOPS/GPU/GCP/Week3-4-AI-Hypercomputer/GCP-AI-Hypercomputer-Architecture|GCP-AI-Hypercomputer-Architecture.md]] |
| ⭐⭐ | [[MLOPS/GPU/GCP/Week1-2-GPU-NVIDIA/GPU-MODE视频推荐清单|GPU-MODE视频推荐清单.md]] → 按顺序看视频 |
| ⭐⭐ | [[MLOPS/GPU/GCP/Week7-8-Operations/GPU-Health-Check-DCGM-XID|GPU-Health-Check-DCGM-XID.md]] |
| ⭐ | [[MLOPS/GPU/GCP/00-GCP-Ramp-Up-Plan|00-GCP-Ramp-Up-Plan.md]] | 通读，按周执行 |
