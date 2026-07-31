# Model Training & Inference Learning Notes

This repository documents a systematic deep-dive into the theory and practice of large model training and inference infrastructure.

The notes cover three areas:

- **Learning preparation** — curated study materials, reading guides, and structured ramp-up plans drawn from publicly available resources including NVIDIA whitepapers, GPU MODE lectures, GCP/AWS documentation, and open-source community content
- **Knowledge summaries** — concept breakdowns written from an engineering perspective, focusing on mental models, design rationale, and the "why" behind architectural decisions rather than surface-level definitions
- **Hands-on walkthroughs** — practical steps for setting up environments, running benchmarks, diagnosing performance issues, and validating GPU cluster health

Topics include GPU architecture (Ampere / Hopper / Blackwell), intra-node communication (NVLink / NVSwitch / Fabric Manager), collective communication (NCCL), distributed training parallelism strategies (DP / TP / PP / EP), LLM inference optimization (KV Cache / PagedAttention / PD disaggregation), and cloud AI infrastructure (GCP AI Hypercomputer / GKE / TCPDirect).

All referenced materials are sourced from publicly available resources.

## Files in This Repo

### Week-x-y QuickStart手册.md

A structured reference guide organized around the six topic blocks of the Week 1-2 ramp-up plan. The goal is to give you **everything you need to get oriented quickly** — core mental models, key numbers, command references, external links, and pointers to deeper reading — all in one place.

Structure: each block has a target outcome, a minimal mental model, the most important commands or concepts, and links to source materials and supplementary notes. Designed to be scanned, not read linearly.

Use this file when you are actively working through Week 1-2 and need a map of what to read, watch, and run — or when you need to quickly look something up without digging through multiple documents.

### Week-x-y 学习总结.md

A retrospective written after completing Week 1-2 of the ramp-up plan. The goal is to capture **what actually changed in understanding** — not a list of facts, but the mental model shifts, the "aha" moments, and the places where things are still unclear.

Structure: cognitive shifts first, then architecture logic, then hands-on commands, then honest gaps. Written from an engineering perspective, assuming the reader already has some background and wants to understand the *why* behind each concept rather than just the *what*.

Use this file when you want to review what the learning phase produced — what stuck, what the key insights were, and what still needs follow-up.

---

The main branch contains the stable, reviewed version of all notes.

Additions and expansions are developed in separate branches — typically named `sidenote/<topic>` — and merged into main via pull requests on an irregular basis. Each PR usually covers one of the following:

- **Sidenotes**: deep-dive annotations linked from a specific paragraph in an existing note, expanding on a concept that warrants more detail than the summary allows
- **New topic files**: standalone notes on a new subject added to the learning plan
- **Corrections and updates**: fixes to factual errors or outdated information in existing notes

If you're reading a note and see a `📎 深度展开` reference, the linked sidenote file was introduced through one of these PRs and has already been merged into main.
