# Tóm tắt — 034 · Beyond Image Borders: Learning Feature Extrapolation for Unbounded Image Composition
> Nguồn: 034_Beyond_Image_Borders_Learning_Feature_Extrapolation_for_Unbounded_Image_Composit.pdf · Worker: paper-summary · Ngày: 2026-06-26

---

## TL;DR (3 câu)

Bài báo đề xuất UNIC (Unbounded Image Composition), một khung làm việc (framework) kết hợp khuyến nghị góc nhìn máy ảnh (camera view recommendation) không giới hạn biên ảnh với bố cục ảnh (image composition). Thay vì mở rộng ảnh bằng sinh ảnh (image out-painting) rồi cắt, UNIC ngoại suy đặc trưng (feature extrapolation) trong không gian đặc trưng (feature space) để dự đoán chuyển động máy ảnh và hộp cắt xén (bounding box) mà không tạo ra pixel giả. Kết quả thực nghiệm trên GAICD và FLMS cho thấy UNIC vượt trội các phương pháp cắt xén ảnh (image cropping) hiện tại ở cả hai bài toán: cắt xén trong biên và cắt xén vượt ngoài biên ảnh.

---

## Vấn đề (Problem)

Các phương pháp bố cục ảnh (image composition) hiện tại chỉ cắt xén trên ảnh đã chụp, bị giới hạn bởi biên ảnh ban đầu. Khi vùng bố cục tốt nhất nằm ngoài biên ảnh, kết quả bắt buộc là thứ tối ưu. Phương pháp mở rộng ảnh bằng out-painting (Zhong et al. [43]) cho phép dự đoán vượt biên nhưng vùng sinh có thể chứa artifact và được đưa vào kết quả cắt xén cuối cùng, làm giảm chất lượng thực.

---

## Ý tưởng cốt lõi (Core idea)

UNIC hợp nhất hai đầu ra — khuyến nghị góc nhìn máy ảnh $`\mathbf{v}_{pred}`$` và hộp bố cục `$`\mathbf{c}_{pred}`$ — thành một bài toán hồi quy hộp không giới hạn (unbounded box regression):

$$
\mathbf{c}_{pred} = f(\mathbf{I}_{init})
$$

*(phương trình 4 trong paper)*

Vì $`\mathbf{c}_{pred}`$` chia sẻ tâm và chiều dài/rộng với `$`\mathbf{v}_{pred}`$` theo tỉ lệ camera cố định (`$`4:3`$` hoặc `$`3:4`$`), `$`\mathbf{v}_{pred}`$` được suy ra trực tiếp từ `$`\mathbf{c}_{pred}`$. Đây là điểm đơn giản hóa then chốt: bài toán kép được rút gọn về một bài toán hồi quy hộp duy nhất.

Để cải thiện độ chính xác khi hộp dự đoán vượt biên ảnh, UNIC thêm một mô-đun ngoại suy đặc trưng (Feature Extrapolation Module — FEM) vào kiến trúc cDETR. FEM dự đoán các token đặc trưng $`\mathcal{Z}_{pad}`$` của vùng ngoài biên từ các token hiện có `$`\mathcal{Z}_{vis}`$, không sinh pixel.

---

## Tóm tắt theo mục (Section-by-section)

### Abstract

Bài báo đề xuất UNIC, khung làm việc kết hợp khuyến nghị góc nhìn và bố cục ảnh không giới hạn biên. UNIC đưa ra chuỗi thao tác máy ảnh (zoom in/out, dịch chuyển) và hộp cắt xén đồng thời, đảm bảo ảnh kết quả là ảnh thật (không chứa vùng sinh tổng hợp). Code, dataset, và mô hình được công bố tại GitHub.

### 1. Introduction

Phần giới thiệu phân tích hạn chế của các phương pháp cắt xén trong biên (in-bounds cropping): kết quả tối ưu bắt buộc khi vùng tốt nhất nằm ngoài biên. Phương pháp out-painting (Zhong et al.) mở rộng biên nhưng tạo ra artifact. UNIC giải quyết bằng cách: (1) dự đoán chuyển động máy ảnh (camera movement) thực, không sinh pixel; (2) ngoại suy trong không gian đặc trưng thay vì không gian ảnh; (3) hỗ trợ điều chỉnh nhiều bước (multi-step adjustment).

### 2. Related Work

Tổng quan hai nhánh phương pháp bố cục ảnh: đánh giá neo (anchor evaluation) — sinh ứng viên rồi xếp hạng, và hồi quy tọa độ (coordinate regression) — dự đoán trực tiếp hộp cắt xén. Phần 2.2 điểm qua các phương pháp out-painting (GAN-based, transformer-based) để làm nền so sánh với FEM của UNIC.

### 3. Method

**3.1 Định nghĩa bài toán.** Cho ảnh khởi tạo $`\mathbf{I}_{init}`$` với góc nhìn `$`\mathbf{v}_{init}`$`, mô hình dự đoán đồng thời `$`\mathbf{v}_{pred}`$` (góc nhìn mới) và `$`\mathbf{c}_{pred}`$` (hộp cắt xén trong `$`\mathbf{v}_{pred}`$`). Hai ràng buộc hình học giữa `$`\mathbf{v}_{pred}`$` và `$`\mathbf{c}_{pred}`$:

- Chia sẻ tâm: $`(x^{\mathbf{v}}_{pred}, y^{\mathbf{v}}_{pred}) = (x^{\mathbf{c}}_{pred}, y^{\mathbf{c}}_{pred})`$ *(phương trình 2)*

- Chia sẻ chiều dài hoặc chiều rộng tùy tỉ lệ:

$$
\begin{cases}
w^{\mathbf{v}}_{pred} = w^{\mathbf{c}}_{pred}, & w^{\mathbf{c}}_{pred}/h^{\mathbf{c}}_{pred} \geq w^{\mathbf{v}}_{pred}/h^{\mathbf{v}}_{pred} \\
h^{\mathbf{v}}_{pred} = h^{\mathbf{c}}_{pred}, & w^{\mathbf{c}}_{pred}/h^{\mathbf{c}}_{pred} \leq w^{\mathbf{v}}_{pred}/h^{\mathbf{v}}_{pred}
\end{cases}
$$

*(phương trình 3)*

**3.2 Mô hình hồi quy không giới hạn.** UNIC xây dựng trên conditional DETR (cDETR) [27] theo Jia et al. [16]: backbone CNN trích xuất đặc trưng $`\mathbf{h}_{init}`$`, transformer encoder tạo `$`\mathcal{Z}_{vis}`$`, transformer decoder cùng hai đầu `$`H_{box}`$` và `$`H_{conf}`$` dự đoán `$`n`$ ứng viên hộp và điểm tin cậy. Hàm mất mát huấn luyện:

$$
\mathcal{L}_\mathrm{comp} = \mathcal{L}_\mathrm{reg}(\mathbf{c}_{pred}, \mathbf{c}) + \lambda_\mathrm{IoU}\mathcal{L}_\mathrm{IoU}(\mathbf{c}_{pred}, \mathbf{c}) + \lambda_\mathrm{focal}\mathcal{L}_\mathrm{focal}(\mathbf{p}_{pred}, \mathbf{p})
$$

*(phương trình 5)*

Khớp nhãn dùng bipartite matching. Chiến lược nhãn mềm (soft label): quality guidance (dùng trước để ổn định) → self-distillation (dùng sau để cải thiện).

**3.3 Mô-đun ngoại suy đặc trưng (FEM).** FEM gồm 6 transformer block, nhận $`\mathcal{Z}_{vis}`$` qua cross-attention để dự đoán `$`\mathcal{Z}_{pad}`$` (token ngoài biên). Giám sát FEM: trích xuất ảnh toàn cảnh `$`\mathbf{I}`$` (lớn hơn `$`\mathbf{I}_{init}`$`) qua CNN/encoder với EMA để lấy `$`\mathcal{Z}_{out}`$ (token thực ngoài biên), sau đó tối thiểu hóa:

$$
\mathcal{L}_{\mathrm{extra}} = \text{smooth-}\ell_1(\mathcal{Z}_{pad}, sg(\mathcal{Z}_{out}))
$$

*(phương trình 6, $`sg(\cdot)`$ là stop gradient)*

Hàm mất mát tổng:

$$
\mathcal{L} = \mathcal{L}_\mathrm{comp} + \mathcal{L}_\mathrm{extra}
$$

*(phương trình 7)*

**3.4 Dataset bố cục ảnh không giới hạn.** Không có dataset công khai cho bài toán này. Tác giả chuyển đổi GAICD [38] và CPC [33] bằng cách lấy mẫu ngẫu nhiên $`\mathbf{v}_{init}`$` từ ảnh toàn cảnh, với ba ràng buộc: (a) kích thước `$`\mathbf{I}_{init} \geq 0.7`$` lần ảnh gốc; (b) `$`\mathrm{IoU}(\mathbf{v}_{init}, \mathbf{v}) \geq 0.7`$`; (c) tỉ lệ `$`4:3`$` hoặc `$`3:4`$.

### 4. Experiments

**4.1 Chi tiết thực nghiệm.** GAICD: 2,636 ảnh train / 200 val / 500 test. CPC: 10,800 ảnh, đánh giá trên FLMS [9]. Backbone CNN khởi tạo từ ImageNet pre-trained. Transformer encoder/decoder 6 lớp. AdamW, learning rate $`10^{-5}`$` (CNN) và `$`10^{-4}`$` (transformer), giảm `$`\times 1/10`$ tại epoch 30, tổng 50 epoch.

**4.2 Kết quả bố cục ảnh không giới hạn.** Bảng 1: UNIC đạt $`Acc_{1/5}(\epsilon=0.90) = 23.2`$`, `$`Acc_{1/10}(\epsilon=0.85) = 72.8`$ trên GAICD; IoU = 0.801 (Disp = 0.048) trên GAICD, IoU = 0.828 (Disp = 0.040) trên FLMS — tất cả tốt nhất so với các phương pháp baseline.

**4.3 Kết quả cắt xén ảnh.** UNIC (không FEM) vẫn đạt $`Acc_{1/5}(\epsilon=0.90) = 74.7`$`, `$`Acc_{1/10}(\epsilon=0.85) = 95.5`$ trên GAICD, vượt Jia et al. [16] là phương pháp mạnh nhất; tương đương trên FLMS.

**4.4 Ablation study.** Ba thực nghiệm ablation:
- *Chiến lược ngoại suy*: ngoại suy đặc trưng (+22.6% $`Acc_{1/5}(\epsilon=0.85)`$ so với không ngoại suy; image-level SRN giảm hiệu suất).
- *Hàm mất mát FEM*: smooth-$`\ell_1`$ tốt nhất tổng thể.
- *Điều chỉnh nhiều bước*: IoU tăng từ 0.721 (step=1) → 0.852 (step=3) trên ảnh lớn từ GAICD.

### 5. Limitation and Future Work

UNIC gặp khó khăn khi người/vật thể bất ngờ xuất hiện ở vùng ngoài biên và ảnh hưởng thẩm mỹ (có thể khắc phục bằng multi-step). Thao tác máy ảnh hiện tại giới hạn ở dịch chuyển và zoom, chưa hỗ trợ thay đổi hướng (camera pose). Thông tin chiều sâu (depth) được gợi ý là hướng mở rộng.

### 6. Conclusion

UNIC mang lại framework thống nhất cho bố cục ảnh không giới hạn biên, sử dụng ngoại suy đặc trưng thay cho sinh ảnh, và xây dựng hai dataset mới từ GAICD/CPC. Kết quả vượt trội các phương pháp SOTA ở cả hai bài toán.

---

## Kết quả chính (Key results)

| Metric | Dataset | UNIC (Ours) | Phương pháp tốt nhất trước đó |
|--------|---------|-------------|-------------------------------|
| $`Acc_{1/5}`$` (`$`\epsilon=0.90`$) | GAICD | **23.2** | 22.3 (Zhong et al.) |
| $`Acc_{1/5}`$` (`$`\epsilon=0.85`$) | GAICD | **59.0** | 53.5 (Zhong et al.) |
| $`Acc_{1/10}`$` (`$`\epsilon=0.90`$) | GAICD | **32.7** | 28.7 (Zhong et al.) |
| $`Acc_{1/10}`$` (`$`\epsilon=0.85`$) | GAICD | **72.8** | 67.2 (Zhong et al.) |
| IoU ↑ | GAICD | **0.801** | 0.795 (Zhong et al.) |
| Disp ↓ | GAICD | **0.048** | 0.050 (Zhong et al.) |
| IoU ↑ | FLMS | **0.828** | 0.818 (Zhong et al.) |
| Disp ↓ | FLMS | **0.040** | 0.042 (Zhong et al.) |
| $`Acc_{1/5}`$` (`$`\epsilon=0.90`$) (cropping) | GAICD | **74.7** | 72.0 (Jia et al.) |
| $`Acc_{1/10}`$` (`$`\epsilon=0.85`$) (cropping) | GAICD | **95.5** | 92.6 (Jia et al.) |

---

## Đóng góp (Contributions)

1. **Framework UNIC**: đề xuất bài toán mới — khuyến nghị góc nhìn và bố cục ảnh không giới hạn biên kết hợp, đảm bảo ảnh kết quả là thật (real pixels) nhờ dựa vào chuyển động máy ảnh thực.
2. **Mô-đun FEM + hàm mất mát ngoại suy**: tích hợp ngoại suy đặc trưng (feature extrapolation) dựa trên masked image modeling vào cDETR, giám sát bằng $`\mathcal{L}_{extra}`$ với EMA để ổn định huấn luyện.
3. **Hai dataset mới**: xây dựng bộ dữ liệu bố cục ảnh không giới hạn biên từ GAICD và CPC bằng quy trình lấy mẫu có ràng buộc.

---

## Hạn chế tác giả tự nêu (Author-stated limitations)

- Vật thể bất ngờ xuất hiện ở vùng ngoài biên (unseen objects) gây ảnh hưởng thẩm mỹ; multi-step adjustment giúp nhưng chưa hoàn toàn giải quyết.
- Thao tác máy ảnh giới hạn ở dịch chuyển 2D và zoom; không hỗ trợ thay đổi hướng máy ảnh (camera pose adjustment).
- Chưa khai thác thông tin chiều sâu (depth information) cho khuyến nghị tốt hơn.

---

## Thuật ngữ (Glossary)

| English | Tiếng Việt | Giải thích ngắn |
|---------|------------|-----------------|
| Image composition | Bố cục ảnh | Tìm vùng cắt xén thẩm mỹ nhất trong một cảnh |
| Image cropping | Cắt xén ảnh | Chọn vùng con trong biên ảnh đã chụp |
| Unbounded image composition | Bố cục ảnh không giới hạn biên | Bố cục ảnh có thể vượt ra ngoài biên ảnh ban đầu |
| Camera view recommendation | Khuyến nghị góc nhìn máy ảnh | Đề xuất thao tác di chuyển/zoom để đạt góc nhìn tốt hơn |
| Feature extrapolation | Ngoại suy đặc trưng | Dự đoán đặc trưng cho vùng ngoài biên ảnh trong không gian đặc trưng |
| Image out-painting | Sinh ảnh ngoài biên | Tạo pixel mới cho vùng ngoài biên ảnh gốc bằng mô hình sinh |
| Feature Extrapolation Module (FEM) | Mô-đun ngoại suy đặc trưng | Thành phần UNIC dự đoán token đặc trưng ngoài biên |
| Bounding box | Hộp giới hạn / Hộp cắt xén | Tọa độ $`[x, y, w, h]`$ xác định vùng bố cục |
| Anchor evaluation | Đánh giá neo | Sinh ứng viên cắt xén rồi xếp hạng |
| Coordinate regression | Hồi quy tọa độ | Dự đoán trực tiếp tọa độ hộp cắt xén |
| conditional DETR (cDETR) | cDETR | Kiến trúc detection transformer cải tiến dùng trong UNIC |
| Exponential Moving Average (EMA) | Trung bình động hàm mũ | Kỹ thuật ổn định tham số encoder/CNN để tạo nhãn giám sát cho FEM |
| Bipartite matching | Khớp hai phía | Thuật toán ghép cặp dự đoán-nhãn trong DETR |
| Multi-step adjustment | Điều chỉnh nhiều bước | Lặp lại dự đoán UNIC để tinh chỉnh góc nhìn qua nhiều bước |
| GAICD | GAICD | Grid Anchor-based Image Cropping Dataset — dataset cắt xén ảnh với nhãn exhaustive |
| FLMS | FLMS | Dataset kiểm tra bố cục ảnh của Fang et al. (2014) |
| CPC | CPC | Composition-Preserving Cropping dataset — dataset với nhãn thưa (sparse) |
| Smooth-$`\ell_1`$` loss | Hàm mất mát smooth-`$`\ell_1`$` | Hàm mất mát hồi quy kết hợp `$`L_1`$` và `$`L_2`$, bền với ngoại lệ |
| Stop gradient ($`sg`$) | Dừng gradient | Toán tử ngăn gradient lan truyền ngược qua đối số |
| Quality guidance | Hướng dẫn chất lượng | Chiến lược nhãn mềm dùng điểm thẩm mỹ làm trọng số |
| Self-distillation | Tự chưng cất | Chiến lược nhãn mềm dùng dự đoán của mô hình EMA làm nhãn |
