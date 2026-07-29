# Model Training & Inference Learning Notes

This repository documents a systematic deep-dive into the theory and practice of large model training and inference infrastructure.

The notes cover three areas:

- **Learning preparation** — curated study materials, reading guides, and structured ramp-up plans drawn from publicly available resources including NVIDIA whitepapers, GPU MODE lectures, GCP/AWS documentation, and open-source community content
- **Knowledge summaries** — concept breakdowns written from an engineering perspective, focusing on mental models, design rationale, and the "why" behind architectural decisions rather than surface-level definitions
- **Hands-on walkthroughs** — practical steps for setting up environments, running benchmarks, diagnosing performance issues, and validating GPU cluster health

Topics include GPU architecture (Ampere / Hopper / Blackwell), intra-node communication (NVLink / NVSwitch / Fabric Manager), collective communication (NCCL), distributed training parallelism strategies (DP / TP / PP / EP), LLM inference optimization (KV Cache / PagedAttention / PD disaggregation), and cloud AI infrastructure (GCP AI Hypercomputer / GKE / TCPDirect).

All referenced materials are sourced from publicly available resources.
