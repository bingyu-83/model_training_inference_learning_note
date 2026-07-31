---
title: "Week 1-2 学习总结：GPU 架构与通信"
tags: [GPU, NVIDIA, NCCL, NVLink, architecture, ramp-up, AWS, GCP]
category: GCP
status: active
date: 2026-07-19
type: 学习总结
cssclasses: [max]
---

# Week 1-2 学习总结：GPU 架构与通信

> 这不是知识点的罗列，而是学完之后真正改变了哪些认知、哪些地方踩过坑、哪些东西值得反复回味。

---

## 一、最大的认知转变

学之前我对 GPU 的理解停留在"显卡，跑模型用的"。学完之后意识到，**GPU 系统本质上是一个分布式内存系统，通信才是瓶颈，不是计算**。

这个转变体现在几个具体的地方：

**HBM 带宽比算力更稀缺。** H100 的 FP16 算力是 990 TFLOPS，但 HBM 带宽只有 3.35 TB/s。一个简单的矩阵乘法，理论上需要的带宽远超实际可用带宽，这就是为什么大多数 kernel 是 memory-bound 而不是 compute-bound。Roofline 模型把这个关系说得很清楚——拐点左边优化内存访问，拐点右边才轮到优化算法。

**NVLink 和 PCIe 的差距不是量变是质变。** H100 的 NVLink 4 是 900 GB/s，PCIe Gen5 是 64 GB/s，差了 14 倍。这不只是数字，它意味着 TP（张量并行）必须在节点内做，跨节点做 TP 在工程上几乎不可行，因为延迟和带宽都撑不住。这个认知直接影响了我对并行策略选择的理解。

**Fabric Manager 是一个容易被忽视的单点故障。** 它是用户空间的守护进程，不是内核驱动。`nvidia-smi` 能正常跑不代表 NVSwitch 已经初始化。见过一个典型场景：节点重启后 Fabric Manager 没有自动启动，`nvidia-smi` 显示 8 张卡全部正常，但 NCCL 跑出来的 busbw 只有 3 GB/s，因为通信全部 fallback 到 TCP 了。这种问题如果不知道 Fabric Manager 的存在，排查方向会完全跑偏。

---

## 二、GPU 架构：三代演进的本质逻辑

Ampere → Hopper → Blackwell 的演进，表面上是规格表的数字变化，背后有一条清晰的逻辑线：**每一代都在解决上一代的主要瓶颈**。

**Ampere 解决的是多租户和精度问题。** MIG 让一张 A100 可以切成 7 个独立实例，这在推理场景下意义很大——不是每个请求都需要整张卡，切分之后资源利用率可以大幅提升。2:4 结构稀疏是另一个思路，通过剪枝换吞吐，但实际落地比较有限，因为保精度的同时做结构化剪枝在工程上很难。

**Hopper 解决的是训练精度和吞吐的矛盾。** Transformer Engine 的核心是动态精度选择：前向用 FP8，如果检测到精度损失就退回 FP16，每层独立决策。这让 FP8 训练变得实用，而不是一个需要大量手工调参的黑魔法。相比 A100，H100 的训练吞吐提升约 3x，但功耗也从 400W 涨到 700W，这个代价在数据中心侧是真实的成本。

**Blackwell 解决的是单 die 的物理极限。** 继续缩小制程已经很难，B200 的答案是把两个 die 封装成一个，通过 10 TB/s 的 NV-HBI 互联，对软件层完全透明。这个设计让 SM 数量翻倍，FP4 的引入则是在推理场景下继续压榨吞吐——4 bit 量化在精度可接受的前提下，理论吞吐再翻倍。

一个值得记住的数字关系：从 A100 到 H100，算力 3x，HBM 带宽 1.7x，NVLink 带宽 1.5x。算力增长远快于带宽增长，这个剪刀差意味着 memory-bound 的问题在 H100 上比 A100 更严重，不是更轻松。

---

## 三、NVLink / NVSwitch / Fabric Manager：三层不能混淆

这三个概念经常被混在一起说，但它们在不同层次上工作，混淆了就没法排障。

**NVLink 是物理链路**，SerDes 接口，点对点，H100 每张卡有 18 条，每条 50 GB/s，合计 900 GB/s。它解决的是两张 GPU 之间怎么传数据的问题。

**NVSwitch 是交叉开关芯片**，解决的是 8 张卡怎么全互联的问题。没有 NVSwitch，8 张卡只能做有限的点对点连接，GPU0 要和 GPU7 通信需要多跳，带宽被共享稀释。有了 NVSwitch，任意两张卡之间都是直连，非阻塞，全速。一台 DGX H100 有 4 个 NVSwitch 3 芯片，每张 GPU 的 18 条 NVLink 分布连接到这 4 个 Switch 上。

**Fabric Manager 是管理面**，用户空间守护进程，负责初始化 NVSwitch 的路由表、管理 NVDomain 分区、监控 NVLink 健康。它是 NVSwitch 能工作的前提。没有它，NVSwitch 硬件在那里，但路由表是空的，GPU 之间无法通信。

排障的时候这个层次很重要：

```
nvidia-smi 正常 → 驱动和 GPU 硬件没问题
nvidia-smi topo -m 显示 NV18 → NVLink 物理连接正常
systemctl status nvidia-fabricmanager → active → NVSwitch 路由表已初始化
nccl-tests busbw > 600 GB/s → 通信路径正确走了 NVLink
```

四步缺一不可，每一步排除一个层次的问题。

---

## 四、NCCL：不只是"通信库"

NCCL 的定位经常被低估。它不只是封装了几个通信原语的库，它做的事情是**在给定拓扑下自动选择最优传输路径和算法**。

五大原语里最重要的关系是：**AllReduce = ReduceScatter + AllGather**。这个等式不是数学游戏，它是 ZeRO / FSDP 的设计基础。ZeRO 之所以能分片存储优化器状态和梯度，就是因为可以用 ReduceScatter 做梯度归约（每张卡只保留自己负责的那一片），需要完整参数时再用 AllGather 拼回来。理解了这个等式，ZeRO 的三个 stage 就不再是黑盒。

Ring AllReduce 和 Tree AllReduce 的选择逻辑也值得记住：Ring 适合大消息（带宽优先，每个链路负载均衡），Tree 适合小消息（延迟优先，O(log N) 跳数）。NCCL 默认根据消息大小自动切换，阈值大约在 256KB。在实际调优中，如果发现小 batch 训练通信延迟高，可以试试强制 `NCCL_ALGO=Tree`。

并行策略和通信原语的对应关系是另一个关键认知：

- DP 用 AllReduce，走节点间网络，带宽敏感
- TP 用 AllReduce / AllGather + ReduceScatter，必须走节点内 NVLink，延迟敏感
- PP 用 Send/Recv，走节点间网络，延迟敏感

这个对应关系直接决定了并行策略的选择边界：**TP 的 degree 不能超过单节点的 GPU 数量**，因为跨节点的 NVLink 不存在，走网络的 TP 通信开销会把收益全部吃掉。

**推理里的 NCCL 和训练不一样。** 训练时 NCCL 最典型的位置是反向传播后的梯度 AllReduce，是后台任务。推理时 NCCL 在前向关键路径上——Tensor Parallel 的 AllReduce/AllGather 不完成，下一层就不能继续；MoE 的 token dispatch 不完成，专家就没法开始算。这意味着推理里的 NCCL 延迟直接影响 TTFT 和每 token latency，不是训练那种"慢一点无所谓"的后台同步。

MoE 的通信形态尤其值得单独记：每个 token 被路由到不同专家，专家可能分布在不同 GPU 上，通信模式是 token dispatch（发到专家所在 GPU）+ token combine（专家算完聚合回来），本质是 All-to-All。这和 dense 模型的 AllReduce 完全不同，而且通信量随路由结果动态变化，热点专家会造成不均衡。

---

## 五、GCP 特有的东西：网络栈的差异

从 AWS 迁移过来最需要重新建立认知的是网络栈。AWS 用 EFA，所有 GPU 实例统一一套。GCP 按机型不同用不同协议，这不是历史遗留问题，而是有意为之的演进路径：

- A3 (H100) → TCPDirect：内核旁路 TCP，绕过内核协议栈，降低延迟
- A3+ (H100) → TCPXO：在 TCPDirect 基础上加入 GPU Direct，数据可以直接从网卡写入 GPU 显存，不经过 CPU
- A4 (B200) → RDMA/RoCE：原生 RDMA，性能最接近 InfiniBand

这个演进的方向是减少 CPU 参与。每一代都在把数据路径上的 CPU 拷贝去掉一层。理解这个方向，就能解释为什么 A3+ 比 A3 快，以及为什么 A4 的网络性能会有质的提升。

GCP 的 NCCL 插件机制也值得注意。不同机型需要安装不同的 NCCL 插件（`gcp-nccl-tcpdirect-plugin` / `gcp-nccl-tcpxo-plugin`），在 GKE 上通过 DaemonSet 自动安装。如果插件版本不对或者没装，NCCL 会 fallback 到标准 TCP，性能会差很多，但不会报错，这是一个容易踩的坑。

---

## 六、Hands-on 中真正有用的命令

不是命令大全，是那些真正在排障中用到的：

```bash
# 第一步：确认拓扑
nvidia-smi topo -m
# 看 NV18 还是 SYS，NV18 才是正常的节点内全互联

# 第二步：确认 Fabric Manager
systemctl status nvidia-fabricmanager
# 必须是 active，不是 active 就是问题根源

# 第三步：跑 NCCL 基准
./all_reduce_perf -b 8 -e 256M -f 2 -g 8
# H100 节点内 busbw 应该 > 600 GB/s
# 如果只有几 GB/s，看 NCCL_DEBUG=INFO 的 transport 字段

# 第四步：定位 transport
export NCCL_DEBUG=INFO
# 看日志里的 transport：
# NVL → 走 NVLink（正常）
# NET/Socket → 走 TCP（异常，Fabric Manager 问题或网络配置问题）
# NET/IB → 走 InfiniBand（跨节点正常）

# 第五步：检查 NVLink 错误
nvidia-smi nvlink -e
# 有错误计数说明硬件有问题，需要进一步排查 XID
```

`nvidia-smi topo -m` 的输出是最快的健康检查。一眼扫过去，全是 NV18 说明节点内通信正常，出现 SYS 说明某对 GPU 之间走了 PCIe，需要查原因。

---

## 七、KV Cache 与推理框架：两条优化路线

理解 KV Cache 之后，vLLM 和 SGLang 的差异就变得很清晰——它们解决的是两个不同层次的问题。

**PagedAttention（vLLM）解决的是单请求内的显存碎片。** 传统做法按 max_seq_len 预分配连续显存，实际用不满就浪费。PagedAttention 借鉴操作系统分页，把 KV Cache 切成固定大小的 block（16 或 32 个 token），按需分配，显存利用率从 60% 提升到 96%+。请求结束后 block 立即释放，没有跨请求的状态保留。

**RadixAttention（SGLang）解决的是跨请求的重复计算。** 多个请求共享相同前缀（system prompt、few-shot 示例、RAG 检索结果）时，PagedAttention 每次都重新计算这些前缀的 KV。RadixAttention 用前缀树（Radix Tree）把 KV Cache 持久化，新请求来了先做最长前缀匹配，命中的部分直接复用，只计算新增部分。前缀重叠 90% 以上时，TTFT 可以快 4-6 倍。

选型逻辑很简单：**请求之间有大量共享前缀（Agent、RAG、多轮对话）→ SGLang；请求完全独立（批量翻译、内容审核）→ vLLM。** 两者不是竞争关系，解决的是不同瓶颈。

**Kimi K3 的 MoE 架构也值得记一下。** 2.8T 参数、每次激活 104B，用的是 KDA（Kimi Delta Attention）+ Gated MLA 混合注意力，3:1 比例（69 KDA + 24 Gated MLA）。KDA 是线性注意力，把历史压缩进固定大小的状态，避免 KV Cache 随序列长度线性增长；MLA 定期回到完整上下文做精确检索。这个设计在 1M token 上下文下 KV Cache 占用减少 75%，解码吞吐是标准 attention 的 6 倍。MoE 部分 896 个专家每次激活 18 个，配合 Latent MoE 把专家 FLOPs 减少约一半。

---

## 八、还没搞清楚的地方

诚实地记录下来，避免假装理解：

- **NVSwitch 的 SHARP 协议**：H100 的 NVSwitch 3 支持在交换机内完成 AllReduce（in-network computing），理论上可以减少 GPU 的通信负担。但实际上 NCCL 什么时候会用 SHARP，触发条件是什么，还没有深入看过。

- **Fabric Manager 的 NVDomain 分区细节**：知道可以把 8 张卡分成多个 NVDomain 做多租户隔离，但具体的配置方式和 GCP 上 GPU Pod Manager 怎么管理这个，还是黑盒。

- **NCCL 的 LL128 协议**：`NCCL_PROTO` 有 Simple / LL / LL128 三种，LL128 据说在某些场景下延迟更低，但具体的适用场景和原理没有深入研究过。

- **B200 的双 die NV-HBI 互联**：对软件透明，但在 NUMA 拓扑上是否有影响？两个 die 之间的内存访问延迟是否不对称？这个问题在做性能调优时可能会遇到。

---

## 九、一句话总结每个核心概念

| 概念 | 一句话本质 |
|------|-----------|
| SM | GPU 的基本计算单元，每个 SM 独立调度，共享 L2 |
| Tensor Core | 专门做矩阵乘法的硬件，每代支持更低精度 |
| HBM | 堆叠式高带宽内存，容量小但带宽是 DDR 的 10 倍以上 |
| NVLink | GPU 间点对点高速链路，比 PCIe 快 14 倍 |
| NVSwitch | 让节点内所有 GPU 全互联的交叉开关芯片 |
| Fabric Manager | 初始化 NVSwitch 路由表的守护进程，没它 NVSwitch 不工作 |
| NVDomain | Fabric Manager 管理的 GPU 逻辑通信域，用于多租户隔离 |
| AllReduce | 所有 GPU 各出一份，最后人人得到总和，DDP 梯度同步用 |
| ReduceScatter | 归约后每人只拿一片，ZeRO 反向传播用 |
| AllGather | 每人出一片，最后人人得到全量，ZeRO 前向传播用 |
| Ring AllReduce | 环形拓扑做 AllReduce，带宽利用率高，适合大消息 |
| Roofline | 判断 kernel 是 memory-bound 还是 compute-bound 的模型 |
| Kernel Fusion | 把多个 kernel 合并，减少 HBM 读写次数 |
| TCPDirect / TCPXO | GCP 的节点间网络协议，逐代减少 CPU 参与 |
| PagedAttention | vLLM 的 KV Cache 分页管理，消除显存碎片，利用率 96%+ |
| RadixAttention | SGLang 的跨请求 KV Cache 复用，前缀树索引，共享前缀场景快 4-6x |
| MoE Token Dispatch | MoE 推理中 token 路由到专家的 All-to-All 通信，动态不均衡 |
