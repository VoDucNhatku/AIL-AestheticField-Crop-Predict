# Phân loại ưu tiên đọc — viewpoint-suggestion
> Câu hỏi: "Viewpoint Suggestion" · Worker: reading-triage · Ngày: 2026-06-18

---

## Bảng xếp hạng (Ranked)

| Hạng | Bài | Tiêu đề rút gọn | Mức liên quan | Lý do (1 dòng) | Hành động | Worker kế tiếp |
|------|-----|-----------------|---------------|----------------|-----------|----------------|
| 1 | `008` | Aesthetic Camera Viewpoint Suggestion with 3D Aesthetic Field | **Cao** | Bài báo trực tiếp giải quyết bài toán viewpoint suggestion — đề xuất 3D aesthetic field (Gaussian Splatting) để tìm góc máy đẹp từ sparse captures, không cần RL. | Đọc kỹ | `methodology-read`, `pipeline-extract` |
| 2 | `003` | Geometric Viewpoint Learning with Hyper-Rays and Harmonics Encoding | **Cao** | Xây dựng biểu diễn 6DoF viewpoint (hyper-ray) và framework học viewpoint pattern — nền tảng lý thuyết trực tiếp cho viewpoint suggestion. | Đọc kỹ | `methodology-read` |
| 3 | `001` | Photography Perspective Composition Towards Aesthetic Perspective Recommendation | **Trung bình** | Đề xuất góc nhìn qua perspective transformation (3D recomposition) — liên quan nhưng tiếp cận từ hướng composition recommendation, không phải 3D viewpoint exploration. | Lướt | `paper-overview`, `paper-summary` |
| 4 | `004` | Image Aesthetic Assessment Based on Pairwise Comparison | **Thấp** | Thuần túy về đánh giá thẩm mỹ ảnh (score regression + binary classification); không đề cập đến viewpoint, chỉ có thể dùng làm scoring component tham khảo. | Bỏ qua (hoặc tham khảo nếu cần scoring) | `paper-overview` (nếu cần) |

---

## Thứ tự đọc đề xuất (Suggested reading order)

1. **`008`** — Đọc kỹ toàn bộ: đây là bài cốt lõi nhất, định nghĩa rõ bài toán viewpoint suggestion trong không gian 3D.
2. **`003`** — Đọc kỹ phương pháp: cung cấp nền tảng biểu diễn viewpoint (hyper-ray, 6DoF) cho việc hiểu sâu `008`.
3. **`001`** — Lướt: bổ sung góc nhìn về perspective-based recommendation; đọc abstract + kết quả thực nghiệm là đủ.
4. **`004`** — Bỏ qua trong giai đoạn này: chỉ tham khảo nếu có nhu cầu hiểu cơ chế aesthetic scoring bằng pairwise comparison.

---

## Ghi chú triage

- **`008` vs `003`**: `008` trả lời câu hỏi *"gợi ý viewpoint nào đẹp?"* trong một scene cụ thể; `003` trả lời *"làm sao biểu diễn và học viewpoint?"* ở mức tổng quát hơn. Cả hai đều cần đọc kỹ.
- **`001`** tiếp cận bài toán từ hướng perspective composition (điều chỉnh góc nhìn để cải thiện bố cục ảnh) — liên quan nhưng phạm vi hẹp hơn (mobile photography, không phải 3D scene exploration).
- **`004`** không có thành phần viewpoint nào; phù hợp chỉ khi cần hiểu kỹ thuật so sánh thẩm mỹ dùng pairwise comparison làm building block.

---

## Thuật ngữ (Glossary)

| English | Tiếng Việt | Giải thích ngắn |
|---------|-----------|-----------------|
| Viewpoint Suggestion | Gợi ý góc máy / điểm nhìn | Tự động đề xuất vị trí + hướng camera tối ưu cho chất lượng thẩm mỹ |
| Aesthetic Field | Trường thẩm mỹ | Hàm liên tục ánh xạ từ camera pose sang điểm số thẩm mỹ trong không gian 3D |
| 6DoF (Six Degrees of Freedom) | 6 bậc tự do | Camera pose gồm 3 vị trí (x, y, z) + 3 góc quay (pitch, yaw, roll) |
| Hyper-ray | Siêu tia | Biểu diễn 6D viewing ray dùng quaternion để mã hóa đủ 6DoF một cách đơn trị |
| Gaussian Splatting | Gaussian Splatting | Kỹ thuật tái tạo cảnh 3D bằng tập hợp các Gaussian 3D, cho phép render nhanh |
| Sparse captures | Ảnh thưa | Tập ảnh đầu vào ít (vài ảnh), không cần quét dày như NeRF truyền thống |
| Perspective Composition (PPC) | Bố cục phối cảnh | Điều chỉnh góc nhìn (perspective) để cải thiện bố cục ảnh, vượt ra ngoài cắt xén 2D |
| Pairwise Comparison | So sánh từng cặp | Kỹ thuật đánh giá chất lượng bằng cách so sánh hai đối tượng thay vì chấm điểm tuyệt đối |
| Score Regression | Hồi quy điểm số | Dự đoán điểm thẩm mỹ dạng giá trị liên tục cho một ảnh |
