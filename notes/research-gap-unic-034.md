# Khoảng trống nghiên cứu — unic-034
> Phạm vi: paper 034 (UNIC — Beyond Image Borders) · Worker: research-gap · Ngày: 2026-06-26

---

## Hạn chế theo từng bài (Per-paper limitations)

### 034 — UNIC (Beyond Image Borders)

**Hạn chế tác giả tự nêu (Section 5):**

1. **Vật thể bất ngờ ngoài biên (unexpected out-of-border objects):** Khi người hoặc vật thể không mong muốn xuất hiện ở vùng ngoài biên ảnh ban đầu, chúng có thể lọt vào khung được khuyến nghị và làm giảm thẩm mỹ. Multi-step adjustment giúp một phần nhưng không giải quyết triệt để.

2. **Thao tác máy ảnh bị giới hạn (limited camera operations):** UNIC chỉ hỗ trợ ba thao tác cơ bản — dịch chuyển ngang/dọc (pan) và zoom in/out. Không có thao tác thay đổi hướng máy ảnh (camera pose / tilt / rotation).

3. **Thiếu thông tin chiều sâu (no depth information):** Mô hình xử lý ảnh 2D thuần túy, không tận dụng cảm biến chiều sâu hay ước lượng độ sâu đơn ảnh (monocular depth estimation) để cải thiện khuyến nghị.

**Hạn chế quan sát thêm qua phân tích:**

4. **Bài toán dataset nhân tạo (synthetic dataset construction):** Dataset unbounded được xây dựng từ GAICD/CPC bằng cách lấy mẫu $\mathbf{v}_{init}$ từ ảnh toàn cảnh theo ràng buộc tỉ lệ $\alpha = 0.7$ và $\beta = 0.7$. Cách tạo dữ liệu này mô phỏng không hoàn hảo tình huống thực tế: người dùng thực không nhất thiết luôn giữ chủ thể trong khung hay dùng tỉ lệ 4:3.

5. **FEM chỉ hoạt động tốt khi vùng ngoài biên có tính nhất quán ngữ nghĩa cao:** Mô hình giả định rằng đặc trưng ngoài biên có thể được ngoại suy từ đặc trưng trong biên qua cross-attention. Điều này không đúng khi cảnh thay đổi đột ngột (abrupt scene change) tại biên ảnh.

6. **Hội tụ đa bước chưa được phân tích sâu:** Thực nghiệm multi-step chỉ thực hiện trên 83 ảnh lớn từ GAICD, tách biệt với bảng so sánh chính, và chỉ chạy tối đa 3 bước — điều kiện hội tụ thực tế chưa được xác định chính thức.

7. **Phụ thuộc vào chất lượng backbone CNN (backbone dependency):** Toàn bộ luồng trích xuất $\mathcal{Z}_{vis}$ phụ thuộc vào ImageNet-pretrained CNN. Với các cảnh ảnh đặc thù (ảnh y tế, ảnh thiên văn, ảnh siêu cận cảnh), FEM có thể ngoại suy sai do distribution shift.

8. **Thiếu đánh giá người dùng (no user study):** Mọi đánh giá đều dựa trên metric tự động (IoU, Disp, Acc) với ground truth từ dataset. Không có user study để xác nhận rằng khuyến nghị của UNIC thực sự hữu ích cho người dùng thực tế khi chụp ảnh.

---

## Khoảng trống xuyên suốt (Cross-cutting gaps)

*(Những điểm mà toàn bộ lĩnh vực image composition / viewpoint recommendation còn bỏ ngỏ, có thể thấy rõ qua paper 034)*

### G1 — Thiếu đánh giá thực (real-world evaluation gap)

Tất cả các baseline được so sánh trong paper (VFN, GAIC, CGS, CACNet, Jia et al., Zhong et al.) đều đánh giá trên dataset ảnh tĩnh với nhãn ground truth từ chuyên gia. Không có thực nghiệm nào đo lường tác động thực tế khi người dùng chụp ảnh theo khuyến nghị (end-to-end user photography quality improvement). Đây là khoảng trống chung của cả lĩnh vực.

### G2 — Bố cục ảnh chưa tính tới 3D và chiều sâu cảnh

UNIC và các phương pháp liên quan đều xử lý bài toán trong không gian 2D. Thông tin hình học 3D của cảnh (depth, surface normal, camera intrinsics) hoàn toàn bị bỏ qua. Trong khi đó, aesthetic viewpoint trong photography thực tế phụ thuộc nhiều vào perspective và foreground-background separation, cả hai đều liên quan trực tiếp đến depth.

### G3 — Dataset quá thiên về ảnh du lịch/ngoài trời

GAICD, CPC, FLMS — tất cả đều chủ yếu chứa ảnh phong cảnh, chân dung ngoài trời. Không có đánh giá trên ảnh trong nhà, ảnh macro, ảnh đêm, hay ảnh tài liệu — những ngữ cảnh mà quy tắc bố cục có thể khác đáng kể.

### G4 — Không có tích hợp thẩm mỹ cá nhân hóa (personalized aesthetics)

UNIC và các baseline đều học một hàm thẩm mỹ chung (universal aesthetic function) từ nhãn crowd-sourced. Sở thích bố cục của từng nhiếp ảnh gia (personalized composition preference) không được mô hình hóa, mặc dù đây là nhu cầu thực tế rõ ràng.

### G5 — Thiếu ngữ nghĩa đối tượng trong bố cục (object-semantic composition)

Các phương pháp hiện tại (kể cả UNIC) học bố cục ảnh từ hộp giới hạn và điểm thẩm mỹ tổng thể, không hiểu các quy tắc bố cục phụ thuộc ngữ nghĩa như "rule of thirds áp dụng khác nhau cho chân dung người và phong cảnh" hay "đường dẫn thị giác với đối tượng chuyển động".

### G6 — Multi-step convergence chưa được lý thuyết hóa

UNIC đề xuất multi-step adjustment như một tính năng thực hành, nhưng không có phân tích lý thuyết hay thực nghiệm đầy đủ về: khi nào hội tụ, tốc độ hội tụ, hay điều kiện hội tụ tối ưu. Đây là khoảng trống kỹ thuật quan trọng.

---

## Khoảng trắng (White space — chưa ai làm)

### W1 — Khuyến nghị góc nhìn trong video thời gian thực (real-time video composition guidance)

UNIC chạy trên ảnh tĩnh đơn lẻ. Chưa có hệ thống nào cung cấp khuyến nghị góc nhìn liên tục, nhất quán theo thời gian (temporally consistent) cho video hoặc luồng camera trực tiếp (live camera stream). Bài toán này đòi hỏi xử lý nhanh và nhất quán thời gian — hoàn toàn mở.

### W2 — Kết hợp sensor chiều sâu vào khuyến nghị góc nhìn thẩm mỹ

Smartphone hiện đại có cảm biến ToF (Time-of-Flight) và LiDAR. Chưa có phương pháp bố cục ảnh nào tận dụng depth map từ sensor để ước lượng vùng ngoài biên thực tế, thay vì ngoại suy mù.

### W3 — Bố cục ảnh không giới hạn với điều kiện ánh sáng và thời tiết

Toàn bộ bài toán unbounded composition hiện tại giả định cảnh tĩnh, ánh sáng ổn định. Không có công trình nào xét đến sự thay đổi ánh sáng (golden hour, backlight), thời tiết (mưa, sương mù), hay điều kiện thách thức (low-light) trong quá trình khuyến nghị góc nhìn.

### W4 — Tích hợp ngôn ngữ tự nhiên vào khuyến nghị góc nhìn

Chưa có hệ thống nào cho phép người dùng mô tả bằng ngôn ngữ tự nhiên ý định sáng tác ("tôi muốn chụp cảnh hoàng hôn với người ở góc trái") và nhận khuyến nghị góc nhìn tương ứng. Đây là hướng hoàn toàn mở với sự phát triển của vision-language models.

### W5 — Đánh giá FEM qua thực nghiệm cảm nhận người dùng (perceptual evaluation)

Hiệu quả của FEM được đo qua IoU/Disp với ground truth, nhưng chưa có ai đánh giá liệu ngoại suy đặc trưng có tạo ra cảm giác "đúng cảnh" hay không đối với người dùng, hay chỉ cải thiện metric mà không cải thiện trải nghiệm.

---

## Cơ hội nghiên cứu (Opportunities)

| # | Gap | Vì sao quan trọng | Hướng khả thi | Dựa trên bài |
|---|-----|-------------------|---------------|--------------|
| O1 | **G2 + W2** — Bố cục ảnh không giới hạn với chiều sâu | Depth giúp ước lượng vùng ngoài biên chính xác hơn (biết trước có tường hay không gian mở), giảm uncertainty cho FEM | Tích hợp monocular depth (e.g., MiDaS, Depth Anything) làm thêm kênh đặc trưng đầu vào; dùng depth map làm prior để constrain vùng FEM ngoại suy | 034 (FEM), 008 (3D Aesthetic Field) |
| O2 | **G1** — Đánh giá thực tế end-to-end | Toàn bộ field thiếu bằng chứng rằng khuyến nghị tự động thực sự giúp người chụp ảnh đẹp hơn | Thiết kế nghiên cứu người dùng: cho N người dùng chụp ảnh với/không có UNIC, đánh giá chất lượng bằng crowd-sourced aesthetics scoring | 034 (UNIC) |
| O3 | **W1** — Khuyến nghị góc nhìn theo thời gian thực | Video composition guidance là use case thực tế cao (livestream, video call, content creation) nhưng hoàn toàn chưa có | Lightweight UNIC variant + temporal consistency loss; điều kiện: latency < 33ms cho 30fps; có thể dùng student-teacher distillation để nén mô hình | 034 (multi-step), 001 (I2V-based composition) |
| O4 | **G4** — Cá nhân hóa bố cục (personalized composition) | Sở thích bố cục khác nhau theo phong cách nhiếp ảnh (street, portrait, landscape); một mô hình chung không đủ | Fine-tune UNIC với adapter layer trên lịch sử ảnh của từng người dùng; dùng pairwise feedback (ảnh nào đẹp hơn?) để học sở thích — kết nối trực tiếp với 004 (pairwise aesthetic comparison) | 034 (UNIC backbone), 004 (pairwise comparison) |
| O5 | **G5** — Bố cục ngữ nghĩa (semantic-aware composition) | Rule of thirds, leading lines, và symmetry áp dụng khác nhau tùy loại đối tượng; hiện tại model không phân biệt | Bổ sung object detection + semantic segmentation vào pipeline UNIC; condition FEM và decoder trên loại đối tượng chính (person, landscape, architecture) | 034 (cDETR backbone) |
| O6 | **W4** — Tích hợp ngôn ngữ tự nhiên | Vision-language interface là xu hướng tất yếu; "shoot the sunset with the couple on the left" là query tự nhiên | Dùng CLIP hoặc LLaVA để encode text intent; cross-attention giữa text embedding và $\mathcal{Z}_{vis}$ để condition FEM và decoder | 034 (cDETR + FEM) |
| O7 | **G6** — Lý thuyết hội tụ multi-step | Không có đảm bảo lý thuyết rằng UNIC hội tụ — quan trọng cho deployment thực tế | Phân tích UNIC như một dynamical system: chứng minh fixed point tồn tại và điều kiện hội tụ; thực nghiệm với nhiều loại cảnh và số bước hơn (> 3) | 034 (Section 4.4 multi-step ablation) |
| O8 | **G3** — Đa dạng domain dataset | Mô hình train trên ảnh outdoor có thể kém trên indoor, macro, low-light | Xây dựng benchmark đa domain cho unbounded composition: thu thập và annotate ảnh indoor, macro, đêm; đánh giá cross-domain transfer | 034 (dataset construction Section 3.4) |

---

## Ghi chú: ARS hand-off

Phân tích trên là single-paper gap analysis, giới hạn ở những gì có thể suy ra từ paper 034 và ngữ cảnh lĩnh vực chung. Để có **gap analysis toàn diện hơn** với tham chiếu literature rộng (ví dụ: so sánh gap của 034 với 001, 003, 008, hoặc tìm kiếm bài báo mới nhất liên quan), nên sử dụng `/ars-lit-review` hoặc `/ars-full` để tra cứu và tổng hợp literature có kiểm chứng chéo.

---

## Thuật ngữ (Glossary)

| English | Tiếng Việt | Giải thích ngắn |
|---------|------------|-----------------|
| Research gap | Khoảng trống nghiên cứu | Vấn đề chưa được giải quyết hoặc giải quyết chưa thỏa đáng trong lĩnh vực |
| White space | Khoảng trắng nghiên cứu | Hướng chưa ai khai phá trong lĩnh vực |
| Personalized aesthetics | Thẩm mỹ cá nhân hóa | Mô hình thẩm mỹ phụ thuộc vào sở thích của từng người dùng |
| Temporal consistency | Nhất quán thời gian | Tính liên tục và ổn định của kết quả qua các khung hình liên tiếp |
| Monocular depth estimation | Ước lượng chiều sâu đơn ảnh | Kỹ thuật ước lượng độ sâu cảnh từ một ảnh 2D duy nhất |
| User study | Nghiên cứu người dùng | Thực nghiệm đánh giá với người dùng thực để đo tác động thực tế |
| End-to-end evaluation | Đánh giá đầu-cuối | Đánh giá toàn bộ hệ thống từ đầu vào người dùng đến kết quả thực tế |
| Distribution shift | Dịch chuyển phân phối | Sự khác biệt giữa phân phối dữ liệu train và dữ liệu test/triển khai thực tế |
| Time-of-Flight (ToF) | Cảm biến đo chiều sâu bằng thời gian bay | Công nghệ camera đo khoảng cách bằng cách đo thời gian tín hiệu ánh sáng phản xạ |
| Semantic-aware composition | Bố cục nhận thức ngữ nghĩa | Hệ thống bố cục ảnh có hiểu loại đối tượng và áp dụng quy tắc phù hợp |
| Fixed point (dynamical system) | Điểm cố định (hệ động lực học) | Trạng thái mà hệ thống lặp không thay đổi nữa — liên quan đến đảm bảo hội tụ multi-step |
| Camera pose | Tư thế máy ảnh | Hướng và góc nghiêng của máy ảnh (tilt, rotation), khác với pan/zoom |
| Vision-language model | Mô hình thị giác-ngôn ngữ | Mô hình AI kết hợp hiểu ảnh và ngôn ngữ tự nhiên (ví dụ: CLIP, LLaVA) |
