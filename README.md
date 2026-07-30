# Model Training & Inference Learning Notes

This repository documents a systematic deep-dive into the theory and practice of large model training and inference infrastructure.

The notes cover three areas:

- **Learning preparation** — curated study materials, reading guides, and structured ramp-up plans drawn from publicly available resources including NVIDIA whitepapers, GPU MODE lectures, GCP/AWS documentation, and open-source community content
- **Knowledge summaries** — concept breakdowns written from an engineering perspective, focusing on mental models, design rationale, and the "why" behind architectural decisions rather than surface-level definitions
- **Hands-on walkthroughs** — practical steps for setting up environments, running benchmarks, diagnosing performance issues, and validating GPU cluster health

Topics include GPU architecture (Ampere / Hopper / Blackwell), intra-node communication (NVLink / NVSwitch / Fabric Manager), collective communication (NCCL), distributed training parallelism strategies (DP / TP / PP / EP), LLM inference optimization (KV Cache / PagedAttention / PD disaggregation), and cloud AI infrastructure (GCP AI Hypercomputer / GKE / TCPDirect).

All referenced materials are sourced from publicly available resources.

## How This Repo Gets Updated

The main branch contains the stable, reviewed version of all notes.

Additions and expansions are developed in separate branches — typically named `sidenote/<topic>` — and merged into main via pull requests on an irregular basis. Each PR usually covers one of the following:

- **Sidenotes**: deep-dive annotations linked from a specific paragraph in an existing note, expanding on a concept that warrants more detail than the summary allows
- **New topic files**: standalone notes on a new subject added to the learning plan
- **Corrections and updates**: fixes to factual errors or outdated information in existing notes

If you're reading a note and see a `📎 深度展开` reference, the linked sidenote file was introduced through one of these PRs and has already been merged into main.
