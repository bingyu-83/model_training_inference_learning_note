---
title: "AWS GPU Infrastructure: HyperPod & Inference Deployment"
tags: [AWS, GPU, HyperPod, SageMaker, inference, training, EKS, KVCache, vLLM]
category: AWS
status: active
date: 2026-07-30
type: 学习总结
cssclasses: [max]
---

# AWS GPU Infrastructure: HyperPod & Inference Deployment

> Summary of key concepts around AWS SageMaker HyperPod, GPU cluster setup, and LLM inference deployment on AWS. All content is based on publicly available AWS documentation and open-source resources.

---

## 一、AWS GPU 机型全景

### 训练主力机型

| 机型 | GPU | 显存 | 节点间网络 | 适用场景 |
|------|-----|------|-----------|---------|
| p5.48xlarge | 8×H100 | 80GB×8 | 3.2 Tbps EFA | 主流 LLM 训练基线 |
| p5e.48xlarge | 8×H200 | 141GB×8 | EFA | 长上下文 / MoE / 显存吃紧 |
| p5en.48xlarge | 8×H200 | 141GB×8 | 更高带宽 EFA | 数百卡以上分布式训练 |
| p6-b200 | Blackwell B200 | 192GB×8 | — | 下一代主力 |
| trn2.48xlarge | 16×Trainium2 | — | — | 成本敏感训练，TCO 低 30-40% |

### 推理主力机型

| 机型 | GPU | 显存 | 适用场景 |
|------|-----|------|---------|
| g5 | 1-8×A10G | 24GB | 主力推理 |
| g6 | 1-8×L4 | 24GB | 新一代高效推理 |
| g6e | 1-8×L40S | 48GB | 大模型推理 / 轻量训练 |
| inf2 | 1-12×Inferentia2 | — | 主力推理加速器，低延迟 |

### 机型选型三维度

```
① 显存（HBM）：70B 全参训练 → H200 141GB 更稳
② 互联带宽（EFA）：数百卡 all-reduce → p5en 不能让 EFA 成瓶颈
③ 算力精度（FP8）：H100/H200/B200 原生 FP8，算力翻倍
```

---

## 二、SageMaker HyperPod 核心概念

### 什么是 HyperPod

HyperPod = AWS 管理"可靠性层"的持久化 GPU 集群，在标准 EC2/EFA 基础设施之上加了三件事：

1. **节点自愈（Auto-resume）** — GPU 故障自动检测、自动替换节点、从最近 checkpoint 恢复训练
2. **生命周期自动化** — 集群部署、升级、回收脚本化
3. **集群级 SLA** — 大规模训练可用性保障

### HyperPod vs 其他方案

| 方案 | 节点自愈 | 运维负担 | 适用场景 |
|------|---------|---------|---------|
| HyperPod Slurm | ✅ 原生 | 中低 | HPC 背景团队，纯训练 |
| HyperPod EKS | ✅ 原生 | 中 | 云原生团队，训推一体 |
| EC2 自建 | 需自研 | 高 | 最大灵活性，已有 K8s 团队 |
| SageMaker Training Job | N/A | 零 | 短期任务（≤28 天），无需持久集群 |

### 什么时候不用 HyperPod

- **<24h 小集群**：持久化集群成本浪费，用 Training Job 更合适
- **只做推理**：自愈/大拓扑对推理无价值，用 SageMaker Inference
- **模型 <1B 单卡**：无需多机通信，torchrun 足够

### 节点自愈机制

HyperPod 持续对每个节点做 health check，检查项包括：
- GPU 健康（nvidia-smi、XID error、DCGM 异常）
- EFA 状态（设备是否 Up、链路是否正常）
- NVIDIA Driver（kernel module 加载、版本匹配）
- 节点响应性（SSM agent 心跳）

`NodeRecovery: Automatic` 时，检测到异常 → 自动标记 faulty → 停止调度 → 自动拉起替换节点 → 训练从最近 checkpoint 自动 resume。

**为什么 GPU 机器特别需要自愈：**
- GPU 机器故障率显著高于 CPU（HBM bit flip、NVLink error、XID 79 等）
- 1024 卡训练若 MTBF 10 天/卡 → 实际约 15 分钟就会挂一次
- 没有自愈机制 = 大规模训练不可能稳定跑下去

---

## 三、GPU 集群存储分层

```
速度快 ←──────────────────────────────────→ 速度慢
容量小 ←──────────────────────────────────→ 容量大

GPU HBM (80-192GB, 2-8 TB/s)
    ↕ 激活值、权重分片、KV Cache
节点 DDR5 内存 (2TB @ p5, ~500 GB/s)
    ↕ CPU offload (ZeRO-3)、数据 prefetch
Instance Store NVMe (30TB @ p5, ~200 GB/s)
    ↕ 临时 scratch、ephemeral checkpoint（实例释放即丢）
FSx for Lustre (1PB+, 1-4 GB/s per TiB)
    ↕ 数据集、持久 checkpoint、多节点共享
Amazon S3 (无限容量)
    ↕ 冷数据归档、DRA 同步 FSx
```

**Checkpoint 策略：** HBM → NVMe（快写）→ 异步到 FSx/S3（持久）

**常见错误：** checkpoint 直写 FSx → 抢训练带宽；NVMe 当持久存储 → 实例替换数据丢

---

## 四、EFA / SRD：AWS 的云原生 RDMA

### 技术栈层次

```
应用层：NCCL
    ↓ aws-ofi-nccl plugin
协议层：SRD（Scalable Reliable Datagram）
    ↓ EFA 实现
驱动层：EFA（OS-bypass 网卡，提供 libfabric 接口）
    ↓ 硬件
硬件层：ENA + EFA 专用 SRAM
```

### SRD 三个关键创新

1. **Multi-path Packet Spraying** — 同一连接的包洒到多条物理路径，充分利用所有网络路径
2. **Out-of-order Delivery** — 不保证包顺序，避免 TCP 的 Head-of-Line blocking
3. **Adaptive Congestion Control** — 基于延迟而非 packet loss 判断拥塞

### p5.48xlarge 的 3.2 Tbps 组成

8 张 H100 × 400 Gbps EFA = 3.2 Tbps，每张 GPU 独立一条 400G 通道，不是聚合带宽。

### EFA 排障步骤

```bash
# 1. 确认 EFA 设备存在
fi_info -p efa

# 2. 确认 NCCL 走了 EFA（不是 TCP）
export NCCL_DEBUG=INFO
# 看日志：NET/OFI = 正常；NET/Socket = 异常（走了 TCP）

# 3. 验证带宽
./all_reduce_perf -b 8 -e 1G -f 2 -g 8
# p5 节点内应该接近 3.2 Tbps

# 4. 安全组检查（必须自引用，双向放行）
# 5. 确认实例在同一 AZ（EFA 不支持跨 AZ）
```

---

## 五、容量预留三种模式

| 模式 | 期限 | 成本 | 适用场景 |
|------|------|------|---------|
| Capacity Block for ML | 1-14 天固定 block | 折扣价，一次性买断 | 短期大规模训练高峰 |
| Flexible Training Plan (FTP) | 时间窗内灵活调度 | 比 ODCR 低 20-30% | SageMaker 训练主线 |
| ODCR | 无固定期限，手动释放 | On-demand 价位，最贵 | 长期稳定需求，关键业务 |

**类比：**
- Capacity Block = 机票（买断一段时间）
- FTP = 流量套餐（先买额度，时间窗内灵活用）
- ODCR = 包月车位（锁一块资源一直备着）

**⚠ 常见坑：** ODCR 未释放会持续计费，训练停止后必须主动释放。

---

## 六、LLM 推理：Prefill vs Decode 双瓶颈

### 两阶段特征

| 阶段 | 瓶颈 | 决定指标 | 优化方向 |
|------|------|---------|---------|
| Prefill | 算力（Compute-bound） | TTFT（首字延迟） | FlashAttention、前缀缓存、FP8 |
| Decode | 内存带宽（Memory-bound） | TPOT（单字时间） | PagedAttention、GQA、投机解码 |

**核心原因：** Decode 每生成一个 token，都要把整个模型权重从 HBM 搬一遍，受限于 HBM 带宽而非算力。

### Prefill 优化三板斧

1. **FlashAttention** — Q×K×V 在 SRAM 一条龙算完，不落 HBM，降低 TTFT 30-50%
2. **前缀缓存（Prefix Caching）** — 共享 System Prompt 的 KV Cache，RAG/chatbot 场景降低 TTFT 50-90%
3. **FP8 量化** — H100/H200 原生 FP8 算力是 FP16 的 2×，降低 TTFT 30-50%

### Decode 优化三板斧

1. **PagedAttention** — 借鉴 OS 虚拟内存分页，消灭 KV Cache 显存碎片，利用率从 40% → 95%
2. **GQA/MQA** — 多个 Query 共享 KV，KV Cache 体积缩小 4-8×（Llama-3/Qwen/Mistral 都是 GQA）
3. **投机解码（Speculative Decoding）** — 小模型猜 + 大模型批改，生成速度提升 1.5-3×，完全无损

### KV Cache 显存估算

```
单 Token KV Cache = 2 × layers × num_kv_heads × head_dim × dtype_bytes

Llama-3 70B (GQA-8):
  单 token = 2 × 80 × 8 × 128 × 2 = ~0.5 MB
  10 万字长文 = ~50 GB

p5en.48xlarge (1,128 GB 显存):
  权重 140GB → KV Cache 池 ~980GB
  10 万字单用户 50GB → 约 20 并发用户
```

### Continuous Batching vs Static Batching

**Static Batching（传统）：** 凑齐一批一起跑，等所有人跑完才放下一批 → GPU 利用率 30-50%

**Continuous Batching（现代）：** 每个 decode step 重新组 batch，已完成立刻退出 + 新请求立刻加入 → GPU 利用率 70-95%

vLLM 比传统 Triton 快 5-10× 的核心原因：Continuous Batching + PagedAttention。

---

## 七、推理部署方案选型

| 方案 | 适用场景 | 优势 | 劣势 |
|------|---------|------|------|
| SageMaker Endpoint | 快速上线、运维能力有限 | 全托管、开箱即用、内建监控 | 定制化空间有限 |
| EKS + vLLM | 需要灵活性、社区生态 | 部署简单、社区活跃、迭代快 | 性能略低于 TRT-LLM |
| EKS + TensorRT-LLM | 追求极致性能、有 K8s 团队 | 性能最优、高度可定制 | 运维复杂度高 |

### 推理稳定性三要素

1. **模型预热** — 部署后自动发送 warm-up 请求，确认模型已加载到 GPU 显存，避免首批请求 P99 抖动
2. **优雅退出** — 配置 preStop hook + terminationGracePeriodSeconds，Pod 下线前完成处理中的请求
3. **健康检查** — Liveness + Readiness Probe，异常节点自动摘除流量

---

## 八、大规模训练容错四层

| 容错层面 | 解决的问题 |
|---------|-----------|
| 节点故障检测 | 单节点 GPU 故障快速发现（DCGM、XID error） |
| 自动恢复 | 故障节点自动替换，不影响整体训练 |
| 训练自动 Resume | 从最近 Checkpoint 恢复，不从头开始 |
| 硬件替换 | GPU 硬件故障换机器 |

### 节点故障处理流程

```
① 确认最近 Checkpoint 时间 → 评估数据损失范围
② 排查故障节点 → dmesg / nvidia-smi / 日志三步定位
③ 触发节点替换
④ 验证新节点 → 再跑一次 NCCL smoke test
⑤ Resume 训练 → 从最近 Checkpoint 恢复
```

---

## 九、训练 vs 推理关注点对比

| 维度 | 大规模训练 | 推理服务 |
|------|-----------|---------|
| 核心指标 | GPU 利用率 / 训练吞吐量 / Loss 曲线 | P99 延迟 / 吞吐量 / Failure rate |
| 稳定性重点 | 容错自动恢复 / Checkpoint Resume | 模型预热 / 优雅退出 / 健康检查 |
| 扩缩容 | 通常固定规模 | 根据业务流量动态扩缩 |
| 瓶颈定位 | NCCL busbw / GPU util / 数据加载 | TTFT vs TPOT / KV Cache 显存 |

---

## 十、常用排障命令

```bash
# GPU 健康
nvidia-smi
nvidia-smi dmon -s u          # 实时 GPU 利用率
dcgmi diag -r 1               # DCGM 快速诊断

# EFA / NCCL
fi_info -p efa                # 确认 EFA 设备
export NCCL_DEBUG=INFO        # 开启 NCCL 调试
./all_reduce_perf -b 8 -e 256M -f 2 -g 8  # NCCL 带宽测试

# 存储
fio --rw=read --bs=1M --numjobs=16 --directory=/fsx  # FSx 吞吐测试
iostat -x 2                   # I/O wait 监控（目标 < 5%）

# 容量预留
aws ec2 describe-capacity-reservations --filters Name=state,Values=active
aws sagemaker list-training-plans

# HyperPod 集群
aws sagemaker describe-cluster --cluster-name <name>
aws sagemaker list-cluster-nodes --cluster-name <name>
```
