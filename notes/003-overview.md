---
paper_id: "003"
title: "Geometric Viewpoint Learning with Hyper-Rays and Harmonics Encoding"
source_filename: "003_Geometric_Viewpoint_Learning_with_Hyper-Rays_and_Harmonics_Encoding.pdf"
worker: "paper-overview"
date: "2026-06-18"
venue: "ICCV 2023"
authors: "Zhixiang Min, Juan Carlos Dibene, Enrique Dunn"
affiliation: "Stevens Institute of Technology"
---

# Overview: Geometric Viewpoint Learning with Hyper-Rays and Harmonics Encoding

## 1. Bài toán đang giải quyết (Problem Statement)

Bài báo giải quyết bài toán **viewpoint learning** (học mẫu góc nhìn): cho trước một môi trường trong nhà dạng 3D point cloud, hệ thống phải học và dự đoán các vị trí và hướng máy ảnh **6DoF** (6 bậc tự do) phù hợp — tức là những góc nhìn mà con người thực sự hay chụp ảnh tại đó — và tổng quát hóa mẫu này sang các môi trường mới chưa thấy trong quá trình huấn luyện.

Bài toán được hình thức hóa là: học phân phối có điều kiện $`D(\mathbf{T} \mid \Pi)`$, trong đó $`\mathbf{T} \in SE(3)`$ là tư thế máy ảnh 6DoF và $`\Pi`$ là mô hình 3D của cảnh. Tại suy luận (inference), với một cảnh $`\Pi'`$ mới, hệ thống phải phân biệt góc nhìn "tốt" (inlier — khớp phân phối $`D`$) với góc nhìn ngẫu nhiên (outlier — lấy mẫu đồng đều từ $`U(\mathbf{T}')`$).

---

## 2. Tại sao bài toán này khó / tại sao cần giải quyết?

Bài báo chỉ ra hai hạn chế lớn của các phương pháp hiện có:

**Hạn chế 1 — Giảm về bài toán phân tích ảnh (image analysis):**
Các phương pháp trước đây (Adrian et al., Kyle et al.) biểu diễn viewpoint thông qua ảnh render: để đánh giá một góc nhìn, phải render cảnh từ góc đó rồi phân tích pixel ảnh. Điều này vừa **tốn kém tính toán** (render nhiều ảnh), vừa mất đi ngữ cảnh hình học và không gian vốn có của viewpoint 6DoF. Pixel ảnh che khuất thông tin hình học thực sự liên quan đến tư thế máy ảnh.

**Hạn chế 2 — Bài toán DoF thiếu hụt (DoF deficiency) của viewing rays:**
Viewing ray truyền thống là đường nửa thẳng 5DoF trong không gian 3D. Nhưng một tư thế máy ảnh 6DoF có thêm một bậc tự do — góc roll. Do đó, **một optic-ray có thể tương ứng với nhiều tư thế máy ảnh khác nhau** (khác nhau về roll), gây ra sự mập mờ khi dùng optic-rays làm biểu diễn. Để giải quyết sự mập mờ, các phương pháp cũ phải phân tích tập hợp (bundle) nhiều rays cùng nhau, tốn kém và phức tạp.

---

## 3. Đóng góp chính (Contributions)

Bài báo liệt kê bốn đóng góp kỹ thuật:

1. **Biểu diễn viewpoint mới bằng encoded viewing rays:** Thay vì phân tích pixel ảnh, biểu diễn viewpoint qua **tập hợp viewing rays** (pencil of viewing rays) được mã hóa thành embedding hình học và ngữ nghĩa, trực tiếp từ point cloud — không cần render ảnh.

2. **Harmonics Ray Encoders (HREs):** Bộ mã hóa ray mới kết hợp point cloud learning với encoding hình cầu (spherical harmonics), tạo ra **trường đặc trưng hình cầu** (spherical feature fields) mã hóa thông tin môi trường phụ thuộc vào hướng nhìn (view-dependent).

3. **Hyper-Rays — biểu diễn ray 6DoF bijective:** Nâng chiều viewing ray lên 6D (từ 5DoF optic-ray lên 6D hyper-ray trong $`\mathbb{L}^6 = \{\hat{\mathbf{o}} \in \mathbb{R}^3, \mathbf{q} \in \mathbb{S}^3\}`$) bằng cách bổ sung quaternion $`\mathbf{q}`$ để mã hóa roll-axis. Mỗi hyper-ray tương ứng **duy nhất** với một tư thế SE(3) — loại bỏ hoàn toàn sự mập mờ.

4. **Inference workflow phân tầng hiệu quả:** Tách rời (decouple) dự đoán vị trí (location) và hướng nhìn (orientation) thành hai giai đoạn — location branch dùng optic-rays, viewpoint branch dùng hyper-rays — cho phép lấy mẫu và phân tích 6DoF viewpoint dày đặc trong ~10 giây trên GPU thông thường.

---

## 4. Phương pháp tổng quát

Hệ thống gồm ba thành phần chính nối tiếp nhau:

**Bước 1 — Map Processing Branch:** Đầu vào là 3D point cloud. PointNet xử lý để tạo ra **per-point HRE coefficients** — các hệ số định nghĩa trường đặc trưng hình cầu cho mỗi điểm trong point cloud.

**Bước 2 — Location Branch (optic-rays):** Với mỗi vị trí ứng viên $`\mathbf{t}`$, tổng hợp tất cả optic-rays từ $`\mathbf{t}`$ đến các điểm point cloud, mã hóa qua HRE (length feature field + directional feature field trên $`\mathbb{S}^2`$), aggregate qua spherical Voronoi pooling → ra điểm vị trí $`S_{loc}(\mathbf{t}) \in [0, 1]`$.

**Bước 3 — Viewpoint Branch (hyper-rays):** Với các vị trí $`\mathbf{t}`$ đã qua ngưỡng lọc, với mỗi tư thế $`(\mathbf{R}, \mathbf{t})`$, tổng hợp hyper-rays tương ứng, mã hóa qua HRE (directional feature field trên $`\mathbb{S}^3`$), aggregate qua Voronoi 3-sphere pooling + view cropping theo FoV → ra điểm viewpoint $`S_{view}(\mathbf{R}, \mathbf{t})`$. Điểm cuối $`S_{final}(\mathbf{R}, \mathbf{t}) = S_{loc}(\mathbf{t}) \times S_{view}(\mathbf{R}, \mathbf{t})`$.

Toàn bộ pipeline chỉ nhận point cloud làm đầu vào, không render ảnh trong quá trình inference.

---

## 5. Kết quả nổi bật

**Dataset:** ScanNet — 1513 scans của 707 cảnh trong nhà khác nhau, ảnh RGB-D. Viewpoints ground truth do con người chụp, được hướng dẫn bởi phần mềm chỉ báo độ thu hút (featurefulness indicator).

**Metrics:** Precision, Recall, AP (Average Precision), FID (Fréchet Inception Distance) — đo mức độ khớp giữa viewpoints được đề xuất và viewpoints GT, ở ngưỡng dung sai vị trí <0.5m và góc nhìn <30°.

**Kết quả định lượng chính (Table 1 trong bài):**

| Method | Loc. AP | View AP | View+GT Loc. AP | FID |
|---|---|---|---|---|
| Adrian et al. | 53.87 | 10.70 (GT H.R.P.) | 27.15 → 10.70 | 230.88 |
| Kyle et al. | 71.89 | — | 22.99 | 187.63 |
| **Ours** | **84.67** | **55.08** | **72.41** | **123.78** |

Phương pháp đạt kết quả tốt nhất trên tất cả các metrics, đặc biệt nổi bật ở View AP (55.08 so với các baseline không đạt được con số đáng kể) — chứng tỏ khả năng dự đoán hướng nhìn 6DoF đầy đủ, điều mà các phương pháp cũ cơ bản không làm được.

**Ablation study (Table 2):** Tắt $`\mathbb{S}^3`$ direction encoding (tức là degradate về optic-rays) làm View AP giảm mạnh từ 52.79 xuống 29.58 — xác nhận tầm quan trọng thiết yếu của hyper-rays trong dự đoán orientation.

**Tốc độ inference:** ~10 giây để lấy mẫu toàn bộ 6DoF viewpoints trong một cảnh ScanNet với sampling grid 0.2m và 4096 orientations trên GTX1080Ti — nhanh hơn đáng kể so với các phương pháp dựa trên render.

---

## 6. So sánh sơ bộ với bài 008 và bài 001

### So sánh với bài 008 (3D Aesthetic Field)

| Khía cạnh | Bài 003 (Hyper-Rays) | Bài 008 (3D Aesthetic Field) |
|---|---|---|
| **Mục tiêu** | Học mẫu viewpoint con người (viewpoint pattern learning) | Gợi ý viewpoint thẩm mỹ từ sparse observations |
| **Input** | Dense 3D point cloud (ScanNet-style) | Sparse RGB images (2–6 ảnh) |
| **Biểu diễn cảnh** | Point cloud + PointNet | 3D Gaussian Splatting |
| **Tiêu chí đánh giá viewpoint** | Phân phối viewpoint của con người (data-driven) | Điểm thẩm mỹ từ teacher model (aesthetic score) |
| **Hàm mục tiêu** | Phân biệt inlier/outlier viewpoint | Tối đa hóa aesthetic score |
| **Cách encode viewpoint** | Hyper-rays (6D) + HRE | Gaussian Splatting features + rendering |
| **Tốc độ** | ~10s/scene trên commodity GPU | Bài báo không nêu rõ (not stated in the paper) |
| **Venue** | ICCV 2023 | arXiv 2026 |

**Điểm tương đồng:** Cả hai đều xử lý viewpoint trong không gian 6DoF SE(3) đầy đủ và đều làm việc với cảnh 3D thực tế. Cả hai đều tránh việc phân tích pixel ảnh render trực tiếp trong bước quan trọng nhất.

**Điểm khác biệt cốt lõi:** Bài 003 tập trung vào **hình học (geometry)** — mã hóa mối quan hệ không gian giữa camera và cảnh một cách chặt chẽ về mặt toán học. Bài 008 tập trung vào **thẩm mỹ (aesthetics)** — tích hợp tri thức aesthetic từ mô hình 2D vào biểu diễn 3D. Bài 003 học "con người hay chụp ở đâu"; bài 008 học "góc nào trông đẹp nhất".

### So sánh với bài 001 (PPC — Photography Perspective Composition)

| Khía cạnh | Bài 003 (Hyper-Rays) | Bài 001 (PPC) |
|---|---|---|
| **Output** | Danh sách viewpoints được đề xuất (dạng heatmap + poses) | Video dẫn dắt người dùng di chuyển đến góc nhìn tốt hơn |
| **Input** | 3D point cloud của môi trường | Một ảnh tĩnh từ góc nhìn chưa tối ưu |
| **Không gian tìm kiếm** | Toàn bộ 6DoF không gian cảnh | Vùng lân cận quanh góc nhìn hiện tại |
| **Backbone** | PointNet + HRE (geometry-based) | I2V model + VLM-based PQA (appearance-based) |
| **Tính tổng quát** | Tổng quát hóa sang cảnh mới hoàn toàn | Cần dataset đặc thù và fine-tuning |
| **Venue** | ICCV 2023 | NeurIPS 2025 |

**Điểm tương đồng:** Cả hai đều hướng đến bài toán thực tế là giúp người dùng tìm góc chụp tốt hơn trong môi trường 3D.

**Điểm khác biệt cốt lõi:** Bài 003 là bài toán **dự đoán/ranking viewpoints** trên toàn bộ cảnh (không gian tìm kiếm rộng, cần point cloud), trong khi bài 001 là bài toán **điều hướng người dùng** từ góc nhìn hiện tại sang góc tốt hơn (không gian tìm kiếm hẹp, chỉ cần một ảnh). Bài 003 thiên về geometric reasoning; bài 001 thiên về appearance và user guidance.

---

## 7. Có nên đọc kỹ không? Phù hợp cho ai?

**Mức độ đọc gợi ý: Nên đọc kỹ** — đây là bài nền tảng quan trọng cho hướng nghiên cứu viewpoint learning trong nhóm.

**Lý do nên đọc kỹ:**
- Là **framework deep learning đầu tiên cho viewpoint modality** — đặt nền móng lý thuyết (hyper-ray, HRE) mà các bài sau (kể cả 008) có thể kế thừa.
- Giải pháp toán học cho bài toán **6DoF bijective representation** (hyper-ray) rất thanh lịch và có thể áp dụng rộng rãi.
- Cách **tích hợp point cloud learning với spherical harmonics encoding** là đóng góp kỹ thuật độc lập, đáng học.
- Kết quả trên ScanNet cung cấp baseline tham khảo cho các so sánh tương lai.

**Phù hợp đặc biệt cho:**
- Ai đang nghiên cứu biểu diễn viewpoint trong không gian 3D (SE(3) camera pose).
- Ai muốn hiểu tại sao 5DoF optic-ray không đủ cho 6DoF viewpoint modeling.
- Ai đang thiết kế pipeline viewpoint suggestion cần geometry-aware encoding.
- Nghiên cứu sinh viết phần Related Work về viewpoint learning — bài này là milestone quan trọng.

**Hạn chế cần lưu ý (từ Conclusions của bài):**
- Phương pháp mô hình hóa tính chất **hình học** của viewpoint preferences, **không phải tính chất thẩm mỹ bề ngoài** (appearance aesthetics) — đây là khoảng cách lớn với bài 008.
- Yêu cầu dense 3D point cloud đầu vào — hạn chế khả năng dùng với dữ liệu ảnh thưa.
- Chưa tích hợp semantic understanding tường minh (chỉ có điểm ngữ nghĩa gián tiếp qua màu sắc trong point cloud).

---

## Thuật ngữ (Glossary)

| English | Tiếng Việt | Giải thích ngắn |
|---|---|---|
| Viewpoint learning | Học mẫu góc nhìn | Học phân phối các tư thế máy ảnh phổ biến/tốt từ dữ liệu quan sát của con người |
| 6DoF (Six Degrees of Freedom) | 6 bậc tự do | Tư thế đầy đủ của máy ảnh: 3 bậc vị trí (x,y,z) + 3 bậc góc (yaw, pitch, roll) |
| SE(3) | Nhóm chuyển động đặc biệt Euclid | Nhóm toán học mô tả các phép quay và tịnh tiến trong không gian 3D |
| Optic-ray | Tia quang học | Đường nửa thẳng 5DoF trong không gian 3D: origin + direction trên S² |
| Hyper-ray | Hyper-ray | Biểu diễn ray 6DoF: origin + quaternion direction trên S³, bijective với SE(3) |
| Harmonics Ray Encoder (HRE) | Bộ mã hóa ray hòa âm | Module mã hóa viewing ray thành embedding bằng cách kết hợp spherical harmonics với point cloud learning |
| Spherical harmonics | Hòa âm hình cầu | Hàm cơ sở trực giao trên mặt cầu, dùng để biểu diễn tín hiệu phụ thuộc hướng |
| Pencil of viewing rays | Tập tia nhìn | Tập hợp tất cả viewing rays phát ra từ một viewpoint đến các điểm trong cảnh |
| Point cloud | Đám mây điểm | Biểu diễn cảnh 3D dưới dạng tập hợp các điểm có tọa độ (x,y,z), màu và pháp tuyến |
| PointNet | PointNet | Kiến trúc mạng neural xử lý trực tiếp point cloud (Qi et al.) |
| Spherical Voronoi pooling | Gộp Voronoi hình cầu | Phân vùng bề mặt cầu thành các ô Voronoi để tổng hợp features theo hướng đều nhau |
| View cropping | Cắt theo tầm nhìn | Cơ chế trích xuất feature liên quan đến FoV cụ thể từ Voronoi 3-sphere feature |
| Location branch | Nhánh vị trí | Giai đoạn 1 của inference: đánh giá điểm vị trí bằng optic-rays |
| Viewpoint branch | Nhánh góc nhìn | Giai đoạn 2 của inference: đánh giá điểm orientation bằng hyper-rays |
| Inlier viewpoint | Góc nhìn inlier | Viewpoint khớp với phân phối học được từ dữ liệu (viewpoint tốt) |
| Outlier viewpoint | Góc nhìn outlier | Viewpoint ngẫu nhiên không khớp phân phối (viewpoint không tốt) |
| DoF deficiency | Thiếu hụt bậc tự do | Vấn đề 5DoF optic-ray không đủ để mô tả duy nhất tư thế 6DoF |
| Roll-axis | Trục roll | Bậc tự do thứ 6 của camera — góc xoay quanh trục quang học |
| Quaternion | Số tứ nguyên | Biểu diễn toán học cho phép quay 3D, tránh gimbal lock; $`\mathbf{q} \in \mathbb{S}^3`$ |
| FoV (Field of View) | Trường nhìn | Góc quan sát của camera theo chiều ngang và dọc |
| ScanNet | ScanNet | Dataset cảnh trong nhà với 1513 scans RGB-D từ 707 cảnh khác nhau |
| AP (Average Precision) | Độ chính xác trung bình | Metric tổng hợp precision-recall curve |
| FID (Fréchet Inception Distance) | Khoảng cách Fréchet Inception | Đo sự khác biệt phân phối giữa ảnh render từ viewpoints đề xuất và GT |
| Non-maximum suppression (NMS) | Chặn không-cực-đại | Kỹ thuật lọc sau inference để giữ các viewpoint cục bộ tốt nhất, loại bỏ trùng lặp |
| View-dependent encoding | Mã hóa phụ thuộc góc nhìn | Đặc trưng thay đổi theo hướng nhìn, không cố định như global features |
| Binary cross entropy (BCE) | Entropy chéo nhị phân | Hàm mất mát dùng cho bài toán phân loại nhị phân (inlier/outlier) |
