---
paper_id: "001"
title: "Photography Perspective Composition: Towards Aesthetic Perspective Recommendation"
source_filename: "001_Photography_Perspective_Composition_Towards_Aesthetic_Perspective_Recommendation.pdf"
worker: "paper-overview"
date: "2026-06-18"
venue: "NeurIPS 2025"
authors: "Lujian Yao, Siming Zheng, Xinbin Yuan, Zhuoxuan Cai, Pu Wu, Jinwei Chen, Bo Li, Peng-Tao Jiang"
affiliation: "vivo Mobile Communication Co., Ltd"
---

# Overview: Photography Perspective Composition — Towards Aesthetic Perspective Recommendation

## 1. Bài toán đang giải quyết (Problem Statement)

Bài báo đề xuất một paradigm mới gọi là **photography perspective composition (PPC)** — bố cục ảnh dựa trên phối cảnh (perspective). Khác với các phương pháp bố cục ảnh truyền thống chỉ hoạt động trong không gian 2D (image cropping — cắt ảnh), PPC thực hiện **3D recomposition** bằng cách di chuyển góc nhìn của máy ảnh trong không gian thực, qua đó thay đổi quan hệ không gian giữa các chủ thể trong khung hình.

**Câu hỏi nghiên cứu cụ thể:** Cho một ảnh chụp từ góc nhìn (perspective) chưa tốt về mặt thẩm mỹ, làm thế nào để tự động gợi ý và tạo ra video dẫn dắt người dùng di chuyển đến góc nhìn thẩm mỹ hơn?

**Phạm vi bài toán:**
- Đầu vào: một ảnh tĩnh chụp từ góc nhìn chưa tối ưu (suboptimal perspective)
- Đầu ra: một video transformation dẫn từ góc nhìn hiện tại sang góc nhìn thẩm mỹ hơn (aesthetically enhanced perspective)
- Không yêu cầu: prompt text, camera trajectory rõ ràng, hay thông tin bổ sung từ người dùng

## 2. Tại sao bài toán này khó / tại sao cần giải quyết?

**Giới hạn của phương pháp hiện tại (image cropping):**
- Cropping chỉ tái bố cục trong mặt phẳng 2D — không thay đổi quan hệ không gian thực giữa các chủ thể
- Khi cảnh có sự sắp xếp chủ thể hỗn loạn (chaotic spatial arrangement), cropping không tạo ra kết quả thỏa mãn

**Ba thách thức chính khi triển khai PPC:**

1. **Thu thập dữ liệu:** Không có dataset nào hiện có cho bài toán PPC; dữ liệu cropping 2D không chứa thông tin transformation phối cảnh.

2. **Thiết kế hệ thống gợi ý:** Thẩm mỹ bố cục thường có quan hệ partial order (so sánh tương đối) hơn là total order (xếp hạng tuyệt đối) — cần cơ chế so sánh cặp (pairwise comparison) thay vì đánh giá đơn lẻ.

3. **Đánh giá chất lượng:** Chưa có metric hay tiêu chí rõ ràng để đánh giá chất lượng phối cảnh sau transformation, cần thiết lập từ đầu dựa trên human performance.

## 3. Đóng góp chính (Contributions)

Bài báo trình bày ba đóng góp chính:

**Đóng góp 1 — Framework xây dựng dataset PPC tự động:**
Một pipeline tự động tạo dữ liệu training PPC từ các bộ ảnh chuyên nghiệp (professional photography datasets). Ý tưởng cốt lõi: tái tạo 3D từ ảnh đẹp → tạo video chuyển động từ góc nhìn tốt sang góc nhìn kém → reverse video để có dữ liệu "từ kém sang tốt". Nguồn dữ liệu gồm GAICD, SACD, FLMS, FCDB, và Unsplash. Hệ thống lọc dữ liệu tự động dùng PQA model với thang điểm 5 bậc (A–E).

**Đóng góp 2 — Hệ thống tạo video PPC:**
Một approach image-to-video (I2V) tạo ra camera movement sequence dẫn dắt người dùng từ góc nhìn kém sang góc nhìn đẹp hơn, kết hợp với RLHF (Reinforcement Learning from Human Feedback) qua DPO (Direct Preference Optimization) để align với human preferences. Hệ thống sử dụng guidance box giúp người dùng biết cần di chuyển máy ảnh theo hướng nào.

**Đóng góp 3 — Mô hình đánh giá chất lượng phối cảnh (PQA — Perspective Quality Assessment):**
Một VLM-based reward model đánh giá chất lượng transformation video qua ba chiều: visual quality (VQ — chất lượng hình ảnh), motion quality (MQ — chất lượng chuyển động), và composition aesthetic (CA — thẩm mỹ bố cục). Mô hình dùng Qwen2-VL-2B làm backbone, huấn luyện hai giai đoạn: unpair-wise (phân biệt chất lượng video) rồi pair-wise (so sánh thẩm mỹ bố cục).

## 4. Phương pháp tổng quát (High-Level Methodology)

PPC lấy đầu vào là một ảnh góc nhìn chưa tối ưu, dùng mô hình I2V (image-to-video) để sinh video camera movement từ góc kém sang góc tốt hơn, sau đó dùng RLHF/DPO với reward model PQA để tinh chỉnh chất lượng video theo sở thích người dùng. Toàn bộ pipeline không cần prompt text hay camera trajectory thủ công từ người dùng.

## 5. Kết quả nổi bật (Experiments & Results)

**Datasets được dùng:**
- Dữ liệu training: từ GAICD, SACD, FLMS, FCDB, Unsplash (bộ ảnh chuyên nghiệp)
- PQA stage 1: ~5K perspective transformation videos (1.5K high-quality, 3.5K low-quality) → 15K unpaired dataset
- PQA stage 2: paired videos từ CogVideoX 1.5, WAN 2.1, và ground truth

**Backbone models được thử nghiệm:** CogVideoX 1.5 5B, HunYuan I2V, Wan2.1 14B

**Metrics chính:**
- Perspective accuracy: CMM (Camera Motion Matching), FVD (Fréchet Video Distance), PSNR, SSIM, LPIPS
- Video quality: I2V Subject/Background, Subject Consistency, Background Consistency, Motion Smoothness, Dynamic Degree, Aesthetic Quality, Imaging Quality
- Human performance score: VQ, MQ, CA

**Kết quả định lượng (Table 2):**
| Model | CMM ↑ | FVD ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | VQ ↑ | MQ ↑ | CA ↑ |
|---|---|---|---|---|---|---|---|---|
| CogVideoX 1.5 5B | 0.5501 | 303 | 8.2380 | 0.2611 | 0.7969 | 0.7073 | 0.7311 | **0.7196** |
| HunYuan I2V | 0.4928 | 264 | **9.4017** | **0.3537** | 0.7915 | **0.7216** | **0.7496** | 0.7070 |
| Wan2.1 14B | **0.5989** | 345 | 9.3668 | 0.3265 | **0.7808** | 0.7195 | 0.7454 | 0.7072 |

**Hiệu quả của RLHF (Table 4):** RLHF cải thiện CMM từ 0.4928 → 0.5014, FVD từ 264.7672 → 270.2212 (nhẹ), VQ từ 0.7216 → 0.7477, MQ từ 0.7496 → 0.7774, CA từ 0.7070 → 0.7342.

**Kết quả định tính:** Mô hình hoạt động tốt trên đa dạng kịch bản: single-subject (nhân vật, động vật), multi-subject (nhiều chủ thể), landscape photography (phong cảnh ngang), và UAV-like scenarios (góc nhìn trên cao).

**Hạn chế (Limitations):**
1. Bị giới hạn bởi độ dài video của các I2V models hiện tại
2. Chất lượng training data phụ thuộc khả năng 3D reconstruction; có thể có artifacts (distortion, fixedness, blur)
3. Data scaling instability: hiệu năng giảm khi dataset quá lớn (model bị unstable với dữ liệu đa dạng)

## 6. Đánh giá: Có nên đọc kỹ không? Phù hợp cho ai?

**Nên đọc kỹ nếu:**
- Nghiên cứu về viewpoint suggestion / perspective recommendation trong photography
- Quan tâm đến image-to-video generation cho ứng dụng thực tế
- Nghiên cứu về aesthetic quality assessment, đặc biệt là reward model training với RLHF
- Muốn hiểu cách xây dựng dataset tự động từ ảnh chuyên nghiệp (automated data generation pipeline)

**Mức độ đọc gợi ý:** Deep-read — bài báo là *paper đầu tiên* đặt ra paradigm PPC, có nhiều kỹ thuật mới về data generation, PQA model, và RLHF alignment đáng học hỏi trực tiếp cho nhóm nghiên cứu viewpoint suggestion.

**Lưu ý:** Bài được published tại NeurIPS 2025, có project page tại https://vivocameraresearch.github.io/ppc.

---

## Thuật ngữ (Glossary)

| English | Tiếng Việt | Giải thích ngắn |
|---|---|---|
| Photography Perspective Composition (PPC) | Bố cục ảnh dựa trên phối cảnh | Paradigm mới: di chuyển góc máy ảnh trong 3D để cải thiện bố cục, thay vì cắt ảnh 2D |
| Perspective Transformation | Biến đổi phối cảnh | Thay đổi góc nhìn và vị trí máy ảnh trong không gian 3D |
| Image Cropping | Cắt ảnh | Phương pháp tái bố cục truyền thống, chỉ hoạt động trong mặt phẳng 2D |
| 3D Recomposition | Tái bố cục 3D | Quá trình cải thiện bố cục bằng cách thay đổi vị trí vật lý của máy ảnh |
| Perspective Quality Assessment (PQA) | Đánh giá chất lượng phối cảnh | Mô hình VLM đánh giá chất lượng video transformation theo VQ, MQ, CA |
| Image-to-Video (I2V) | Sinh video từ ảnh | Task tạo video camera movement từ một ảnh đầu vào |
| RLHF (Reinforcement Learning from Human Feedback) | Học tăng cường từ phản hồi con người | Kỹ thuật tinh chỉnh model dựa trên sở thích người dùng |
| DPO (Direct Preference Optimization) | Tối ưu hóa sở thích trực tiếp | Biến thể của RLHF, tối ưu trực tiếp từ dữ liệu cặp win/lose |
| Visual Quality (VQ) | Chất lượng hình ảnh | Một trong ba chiều đánh giá của PQA: độ sắc nét, màu sắc, v.v. |
| Motion Quality (MQ) | Chất lượng chuyển động | Một trong ba chiều đánh giá: độ mượt mà, nhất quán của camera movement |
| Composition Aesthetic (CA) | Thẩm mỹ bố cục | Một trong ba chiều đánh giá: cải thiện bố cục nghệ thuật qua video |
| Partial Order | Thứ tự bộ phận | Quan hệ so sánh không hoàn toàn — A tốt hơn B trong ngữ cảnh này nhưng không phải lúc nào cũng vậy |
| BTT Loss (Bradley-Terry with Ties) | Hàm mất mát Bradley-Terry có hoà | Hàm loss cho pairwise comparison, mở rộng Bradley-Terry framework để xử lý trường hợp hoà |
| Guidance Box | Hộp hướng dẫn | Bounding box trên ảnh chỉ hướng di chuyển máy ảnh cho người dùng |
| Suboptimal Perspective | Góc nhìn chưa tối ưu | Góc chụp hiện tại chưa đạt thẩm mỹ tốt |
| Automated Data Generation Pipeline | Pipeline tạo dữ liệu tự động | Quy trình tự động thu thập và tổng hợp dữ liệu training không cần annotation thủ công |
| ViewCrafter | ViewCrafter | Mô hình 3D scene reconstruction được dùng trong pipeline tạo dữ liệu PPC |
| VLM (Vision-Language Model) | Mô hình ngôn ngữ thị giác | Mô hình kết hợp hiểu ảnh và ngôn ngữ, dùng làm backbone cho PQA |
