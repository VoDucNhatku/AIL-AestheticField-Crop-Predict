# Overview: Bài báo 004

| Trường       | Nội dung                                                                 |
|--------------|--------------------------------------------------------------------------|
| **Paper ID** | 004                                                                      |
| **Tiêu đề**  | Image Aesthetic Assessment Based on Pairwise Comparison – A Unified Approach to Score Regression, Binary Classification, and Personalization |
| **Nguồn**    | `004_Image_Aesthetic_Assessment_Based_on_Pairwise_Comparison.pdf`       |
| **Worker**   | paper-overview                                                           |
| **Ngày**     | 2026-06-18                                                               |
| **Venue**    | ICCV (International Conference on Computer Vision)                      |
| **Tác giả**  | Jun-Tae Lee, Chang-Su Kim – Korea University                            |

---

## 1. Bài toán đang giải quyết

Bài báo giải quyết bài toán **đánh giá thẩm mỹ ảnh (image aesthetic assessment)** theo ba nhánh đồng thời:

1. **Score regression (hồi quy điểm thẩm mỹ):** dự đoán một con số liên tục (ví dụ: 1–10) thể hiện chất lượng thẩm mỹ tổng quát của ảnh.
2. **Binary aesthetic classification (phân loại nhị phân thẩm mỹ):** phân ảnh thành hai lớp "chất lượng cao" (high quality) hoặc "chất lượng thấp" (low quality).
3. **Personalized aesthetics (thẩm mỹ cá nhân hóa):** điều chỉnh điểm thẩm mỹ để phản ánh sở thích riêng của từng người dùng.

Trước bài báo này, ba nhánh trên thường được xử lý bởi các mô hình riêng rẽ; bài báo đề xuất **một thuật toán thống nhất (unified approach)** xử lý cả ba.

---

## 2. Tại sao khó / cần giải quyết

- **Tính chủ quan và mơ hồ của thẩm mỹ:** không có quy tắc tuyệt đối; con người dùng nhiều tiêu chí như rule of thirds, visual balance, bố cục.
- **Score regression khó hơn binary classification:** dự đoán một điểm liên tục đòi hỏi mô hình nắm được thứ tự ưu tiên tinh tế hơn, nhưng điểm số lại quan trọng trong ứng dụng thực tế (sắp xếp, truy xuất ảnh).
- **Annotation individual images không đáng tin cậy:** chú thích một ảnh đơn lẻ rất khó nhất quán; so sánh cặp (pairwise comparison) dễ và chính xác hơn cho con người.
- **Personalization tốn kém:** huấn luyện mô hình riêng cho từng người cần quá nhiều dữ liệu gán nhãn; cần giải pháp few-shot.
- **Phương pháp CNN trước đây** (RDCNN, DMA-Net, Reg-Net, MNA-Net, PAC-Net, A-Lamp) chỉ xử lý được một nhánh, hoặc cần thông tin bổ sung ngoài ảnh (attribute, saliency detection, scene categorization).

---

## 3. Đóng góp chính (Contributions)

1. **Unified algorithm đầu tiên** giải quyết đồng thời ba bài toán: score regression, binary classification, personalized aesthetics — chỉ từ dữ liệu ảnh thô, không cần annotation phụ.
2. **Aesthetic comparator (bộ so sánh thẩm mỹ):** mạng Siamese kết hợp feature extractor song song và một ternary classifier (bộ phân loại ba lớp: superior / similar / inferior), ước lượng tỉ lệ điểm thẩm mỹ giữa hai ảnh.
3. **Pairwise comparison matrix + eigenvalue decomposition:** từ kết quả so sánh giữa nhiều reference images và ảnh đầu vào, xây dựng ma trận so sánh cặp, rồi lấy principal eigenvector (vector riêng chính) để hồi quy điểm số — áp dụng phương pháp scaling của Saaty [35].
4. **State-of-the-art** trên cả ba bài toán, đặc biệt vượt A-Lamp [29] tới **9.0% accuracy** trong binary classification.

---

## 4. Phương pháp tổng quát

**Bước 1 — Aesthetic Comparator:** Huấn luyện mạng Siamese (ResNet-50 làm backbone, thêm 4 local residual blocks res5-1 đến res5-4 song song với res5 gốc để khai thác đặc trưng cục bộ) kết hợp với ternary classifier để ước lượng tỉ lệ điểm thẩm mỹ giữa hai ảnh (quantize thành 3 lớp: superior/similar/inferior dùng ngưỡng $`\gamma`$ và $`\theta`$ tối ưu theo Lloyd algorithm).

**Bước 2 — Pairwise comparison matrix + Eigenvalue decomposition:** Dùng $`R`$ reference images (có điểm đã biết) để xây dựng ma trận so sánh cặp $`\mathbf{A}`$ (kích thước $`(R+1)\times(R+1)`$), sau đó giải bài toán trị riêng $`\mathbf{A}\mathbf{u} = \lambda\mathbf{u}`$ lấy principal eigenvector, nhân với hệ số scale $`\kappa^*`$ tối ưu hóa theo least squares để ra điểm thẩm mỹ.

**Biến thể theo nhiệm vụ:**
- *Score regression:* reference images phân bố đều trên toàn thang điểm ($`R = 110`$ trên AVA).
- *Binary classification:* reference images tập trung ở vùng điểm median ($`R = 30`$).
- *Personalized:* kết hợp generic reference images ($`R_g`$) và personal reference images ($`R_p`$) trong cùng một ma trận $`\mathbf{A}`$ mở rộng.

---

## 5. Kết quả nổi bật

### Datasets
| Dataset       | Mục đích                        | Quy mô                                         |
|---------------|---------------------------------|------------------------------------------------|
| **AVA** [32]  | Binary classification + generic regression | ~250,000 ảnh; 235,599 train / 19,930 test; điểm 1–10 (trung bình ~200 annotators) |
| **AADB** [18] | Generic score regression        | 10,000 ảnh; 8,500 train / 500 val / 1,000 test; 11 attribute scores |
| **FLICKER-AES** [34] | Personalized regression  | 40,000 ảnh; 35,263 train / 4,737 test; 210 workers |

### Score Regression (Spearman's $`\rho`$ / MASD)

| Phương pháp | AVA $`\rho`$↑ | AVA MASD↓ | AADB $`\rho`$↑ | AADB MASD↓ |
|-------------|--------|-----------|---------|------------|
| Reg-Net [18] | 0.558 | 0.0582 | 0.678 | 0.1268 |
| PAC-Net [17] | 0.871 | — | 0.837 | — |
| **Proposed** | **0.918** | **0.0229** | **0.879** | **0.1141** |

### Binary Classification (Accuracy %, AVA)

| Phương pháp | Accuracy (%) |
|-------------|-------------|
| A-Lamp [29] | 82.5 |
| **Proposed** | **91.5** |

(Tăng +9.0% so với state-of-the-art trước đó; không dùng external information như attribute/saliency.)

### Personalized Regression (Spearman's $`\rho`$, FLICKER-AES)

| Phương pháp | Generic | $`+R_p=10`$ | $`+R_p=100`$ |
|-------------|---------|--------|---------|
| PAM [34]    | 0.514   | +0.006 | +0.039  |
| **Proposed** | **0.668** | **+0.040** | **+0.044** |

(Generic $`\rho`$ cao hơn PAM đáng kể; personalization cải thiện thêm +0.040 chỉ với 10 personal reference images.)

---

## 6. Đánh giá vai trò trong context Viewpoint Suggestion

**Nên đọc kỹ không?** → **Có, đọc kỹ phần 3 (Proposed Algorithm) và phần 4.2 (Score Regression).**

**Vai trò trong bức tranh lớn:**

Bài 004 cung cấp một **aesthetic scoring module (module cho điểm thẩm mỹ)** hoàn chỉnh và mạnh mẽ, có thể được tích hợp trực tiếp vào pipeline của các hệ thống viewpoint suggestion (như bài 001, 003, 008) theo các cách sau:

| Cách tích hợp | Mô tả |
|---------------|-------|
| **Reward/scoring signal** | Dùng aesthetic comparator làm hàm đánh giá (scoring function) để xếp hạng các viewpoint candidate. |
| **Pairwise training signal** | Cặp (viewpoint A tốt hơn viewpoint B) có thể huấn luyện comparator — hợp với cách bài 001/003 dùng preference labels. |
| **Personalization** | Hệ thống viewpoint suggestion có thể cá nhân hóa gợi ý dựa trên sở thích người dùng qua few personal reference images. |
| **Score regression làm ground truth** | Thay vì chỉ dùng binary label (tốt/không tốt), bài 004 cung cấp continuous score → gradient phong phú hơn cho huấn luyện. |

**Điểm khác biệt quan trọng:** bài 004 đánh giá **ảnh 2D đã chụp**, không trực tiếp đề xuất viewpoint 3D trong không gian. Tuy nhiên, khi kết hợp với bài 001/003/008 (học aesthetic field 3D), module scoring của bài 004 có thể đóng vai trò **discriminator** hoặc **quality verifier** cho ảnh render từ viewpoint candidate.

---

## Thuật ngữ (Glossary)

| English | Tiếng Việt | Giải thích ngắn |
|---------|-----------|-----------------|
| Aesthetic assessment | Đánh giá thẩm mỹ | Quá trình tự động đánh giá chất lượng thẩm mỹ của ảnh |
| Score regression | Hồi quy điểm thẩm mỹ | Dự đoán điểm số liên tục thể hiện mức độ đẹp của ảnh |
| Binary classification | Phân loại nhị phân | Phân ảnh thành high quality / low quality |
| Personalized aesthetics | Thẩm mỹ cá nhân hóa | Điều chỉnh điểm thẩm mỹ theo sở thích riêng của người dùng |
| Pairwise comparison | So sánh cặp | So sánh hai ảnh để xác định ảnh nào đẹp hơn |
| Aesthetic comparator | Bộ so sánh thẩm mỹ | Mạng Siamese ước lượng tỉ lệ điểm thẩm mỹ giữa hai ảnh |
| Siamese network | Mạng Siamese | Kiến trúc mạng đôi với trọng số chia sẻ, xử lý hai đầu vào song song |
| Ternary classifier | Bộ phân loại ba lớp | Phân loại quan hệ hai ảnh thành superior / similar / inferior |
| Pairwise comparison matrix | Ma trận so sánh cặp | Ma trận A chứa tỉ lệ điểm thẩm mỹ giữa các cặp ảnh |
| Eigenvalue decomposition | Phân tích trị riêng | Phân tích ma trận để lấy principal eigenvector làm vector ưu tiên |
| Principal eigenvector | Vector riêng chính | Eigenvector ứng với trị riêng lớn nhất; dùng để suy ra điểm thẩm mỹ |
| Reference image | Ảnh tham chiếu | Ảnh có điểm đã biết, dùng để hiệu chuẩn điểm của ảnh đầu vào |
| Spearman's coefficient (ρ) | Hệ số tương quan Spearman | Đo độ tương quan giữa hai vector thứ hạng (rank); giá trị gần 1 là tốt |
| MASD | Sai lệch tuyệt đối trung bình điểm | Mean of Absolute Score Differences — đo sai số trực tiếp giữa điểm dự đoán và ground truth |
| AVA dataset | Bộ dữ liệu AVA | Large-scale aesthetic assessment dataset ~250k ảnh, điểm 1–10 từ ~200 người |
| AADB dataset | Bộ dữ liệu AADB | Aesthetics and Attributes Database — 10k ảnh, có attribute scores |
| FLICKER-AES dataset | Bộ dữ liệu FLICKER-AES | Dataset cho personalized aesthetics — 40k ảnh, 210 workers |
| Lloyd algorithm | Thuật toán Lloyd | Thuật toán lặp để tối ưu hóa ngưỡng lượng tử hóa (quantization thresholds) |
| Viewpoint suggestion | Gợi ý góc chụp | Bài toán tự động đề xuất vị trí/góc chụp tốt nhất cho người chụp ảnh |
| Aesthetic field | Trường thẩm mỹ | Biểu diễn 3D của chất lượng thẩm mỹ trong không gian cảnh vật |
