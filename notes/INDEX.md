# INDEX — notes/

Thư mục này chứa toàn bộ artifacts phân tích cho 4 papers trong dự án.
Naming convention: `<id>-<worker>.md` cho per-paper · `analysis-<slug>.md` cho cross-paper.
Notebooks: `notebooks/<id>-<mode>.ipynb` cho paper-coder artifacts.

---

## Per-paper artifacts

| Paper ID | Worker | Đường dẫn | Ghi chú |
|----------|--------|-----------|---------|
| 001 | paper-overview | [001-overview.md](001-overview.md) | Photography Perspective Composition (PPC) — paradigm mới vượt cropping 2D, đề xuất camera movement 3D |
| 001 | pipeline-extract | [001-pipeline.md](001-pipeline.md) | 3 pipeline: Dataset Generation, PPC Inference (I2V+RLHF), PQA Model (Qwen2-VL-2B) |
| 001 | vi-translate | [001-vi.md](001-vi.md) | Bản dịch tiếng Việt toàn văn — Abstract, Introduction, Related Work, Methodology §3.1–3.4, Experiments, Conclusion, Appendix A + Glossary |
| 003 | paper-overview | [003-overview.md](003-overview.md) | Geometric Viewpoint Learning — 6DoF hyper-ray + HRE, học phân phối viewpoint con người từ point cloud indoor |
| 003 | pipeline-extract | [003-pipeline.md](003-pipeline.md) | 2-stage hierarchical: Location Branch (S² optic-ray) → Viewpoint Branch (S³ hyper-ray + View Cropping) |
| 004 | paper-overview | [004-overview.md](004-overview.md) | Pairwise aesthetic comparator — unified approach: score regression + binary + personalized aesthetics |
| 004 | pipeline-extract | [004-pipeline.md](004-pipeline.md) | Siamese ResNet-50 → pairwise comparison matrix → eigenvalue decomposition → aesthetic score |
| 008 | paper-overview | [008-overview.md](008-overview.md) | 3D Aesthetic Field (Gaussian Splatting) gợi ý camera pose đẹp nhất từ sparse captures — SOTA Viewpoint Suggestion |
| 008 | pipeline-extract | [008-pipeline.md](008-pipeline.md) | 2-stage: Distillation (feedforward 3DGS + aesthetic head) → Two-stage Search (coarse + gradient ascent) |
| 034 | paper-overview | [034-overview.md](034-overview.md) | UNIC — framework kết hợp khuyến nghị góc nhìn + bố cục ảnh không giới hạn biên, dùng feature extrapolation |
| 034 | pipeline-extract | [034-pipeline.md](034-pipeline.md) | Pipeline UNIC: cDETR backbone → FEM ngoại suy đặc trưng → hồi quy hộp không giới hạn → multi-step adjustment |
| 034 | vi-translate | [034-vi.md](034-vi.md) | Bản dịch tiếng Việt toàn văn paper 034 |
| 034 | paper-summary | [034-summary.md](034-summary.md) | Tóm tắt đầy đủ theo mục: problem, FEM, dataset, kết quả GAICD/FLMS, ablation 3 chiều |

---

## Notebooks (paper-coder)

| Paper ID | Mode | Đường dẫn | Ghi chú |
|----------|------|-----------|---------|
| 001 | run-results | [../notebooks/001-run-results.ipynb](../notebooks/001-run-results.ipynb) | Load PPC LoRA (HF: LujianYao/PPC) + Wan2.1-I2V-14B + VideoReward → tái lập Table 2; cần A100/H100, ~32 GB disk |

## Modal apps (paper-modal)

| Paper ID | Mode | Đường dẫn | Ghi chú |
|----------|------|-----------|---------|
| 034 | run-results | [../modal_apps/034-run-results/modal_app.py](../modal_apps/034-run-results/modal_app.py) | UNIC evaluation trên GAICD test set; T4 GPU, ~$0.59/run; cần upload checkpoint + dataset thủ công vào Modal Volume |
| 034 | reproduce | [../modal_apps/034-reproduce/modal_app.py](../modal_apps/034-reproduce/modal_app.py) | UNIC full training 50 epochs trên GAICD; 2×T4 (rẻ nhất), batch=4/GPU, ~$11.80/run (~10h); streaming stdout; patches CUDA_VISIBLE_DEVICES |

---

## Cross-paper analysis

| Scope | Loại | Đường dẫn | Ghi chú |
|-------|------|-----------|---------|
| all (001, 003, 004, 008) | reading-triage | [reading-triage-viewpoint-suggestion.md](reading-triage-viewpoint-suggestion.md) | Triage 4 papers theo chủ đề Viewpoint Suggestion — 008 & 003 ưu tiên cao |
| all (001, 003, 004, 008) | analysis | [analysis-complexity-reproducibility.md](analysis-complexity-reproducibility.md) | Code availability + độ phức tạp lý thuyết/code + đánh giá khả năng tự reimplementation cho 4 papers |
| 034 | research-gap | [research-gap-unic-034.md](research-gap-unic-034.md) | 8 hạn chế, 6 gaps xuyên suốt, 5 white spaces, 8 cơ hội — tập trung vào depth, real-time, personalization, language |
| 001, 008, 034 | compare-papers | [compare-001-008-034.md](compare-001-008-034.md) | So sánh 3 papers theo 9 chiều; 3 chiều phân kỳ chính; pipeline hybrid; 3 đề xuất cải thiện/paper |

---

## Nhật ký kiểm tra

| Loại | Đường dẫn | Ghi chú |
|------|-----------|---------|
| audit | [audit-log.md](audit-log.md) | Toàn bộ tác vụ AI đã thực hiện — worker × paper × file đầu ra × trạng thái |
