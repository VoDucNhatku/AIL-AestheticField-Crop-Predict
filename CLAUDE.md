# AIL-Premium Project Context

## Papers analyzed
- **001** — Photography Perspective Composition (PPC), Yao et al., NeurIPS 2025. Video-based perspective transformation using I2V models (CogVideoX, Hunyuan, WAN2.1). PQA model (Qwen2-VL-2B). Code likely private (vivo).
- **008** — 3D Aesthetic Field, Tang et al., arXiv Feb 2026. Distills 2D aesthetic knowledge into feedforward 3D Gaussian Splatting. Two-stage search: coarse sampling + gradient ascent. Code likely not yet public.
- **034** — UNIC (baseline), Liu et al., arXiv Sep 2023. Feature extrapolation for unbounded image composition. Conditional DETR + FEM (6 transformer blocks). **Code + data PUBLIC** on GitHub.

## Current strategy: Gaussian AFF for Q1
- Replace UNIC's FEM with Gaussian-splatting-inspired feature extrapolation
- Add aesthetic feature distillation (from VEN) into Gaussian field
- Add depth-enhanced Gaussian parameterization (Depth Anything Small)
- Dual-mode: Full (training) + Lightweight (inference)
- Target: IEEE TIP or IEEE IoT Journal (Q1)

## Why UNIC is the foundation
- Only paper with public code + pretrained models + dataset
- 001 and 008 are too new, code likely incomplete or private
- Strategy: "old wine in new bottle" — UNIC architecture + 2026 ideas (3DGS, aesthetic field)

## Discarded proposals
- Drone/IoT tracking: UNIC too slow for real-time, would need separate optimization paper
- Federated Learning: no privacy issue in image composition, novelty ~0
- Multimodal (audio/radar): too broad, loses focus

## Resources
- Papers folder: `D:\AIL\AIL-Premium\papers\`
- Notes folder: `D:\AIL\AIL-Premium\notes\`
- UNIC GitHub: liuxiaoyu1104/UNIC
