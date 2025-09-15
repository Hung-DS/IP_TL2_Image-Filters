Dự án triển khai các bài tập kèm một ứng dụng Streamlit giao diện tiếng Việt để thao tác trực quan.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng Streamlit

```bash
streamlit run streamlit_app.py
```
- hoặc chạy file run_app.py

## Tính năng chính (UI)

- Bài 1 (Làm mờ và đánh giá):
  - Bộ lọc: Mean, Gaussian, Median, Bilateral.
  - Nhiễu: Gaussian noise, Salt & Pepper (chọn loại và mức độ).
  - Đánh giá: PSNR, SSIM; Histogram ảnh gốc, ảnh nhiễu và ảnh sau lọc.

- Bài 2 (Phát hiện biên):
  - Cài đặt từ đầu: Sobel, Prewitt, Laplacian (tự kernel + tích chập), so sánh với Canny.
  - Hiển thị pipeline: Gradient X, Gradient Y, Magnitude, ảnh sau ngưỡng cho từng phương pháp.
  - Phân tích ngưỡng (Sobel): chọn nhiều ngưỡng để so sánh.

- Bài 3 (Tăng cường ảnh mờ/thiếu sáng):
  - Kết hợp CLAHE với làm sắc nét: Laplacian sharpen, Unsharp Masking (giữ màu bằng LAB).
  - Hiển thị đồng thời 4 kết quả: Laplacian, Laplacian+CLAHE, Unsharp, Unsharp+CLAHE và 4 histogram tương ứng.
  - Workflow gợi ý cho ảnh điện thoại: Gaussian mờ nhẹ → CLAHE → Unsharp.
  - Ảnh hiển thị và ảnh tải xuống luôn ở dạng màu (ảnh xám tự chuyển sang RGB).

- Bài 4 (Ảnh y tế – demo):
  - Làm mờ Gaussian để khử nhiễu nhẹ, phát hiện biên Sobel/Canny, so sánh trực quan.

## Sử dụng

1) Chuẩn bị ảnh PNG/JPG và tải ảnh ở Sidebar.
2) Chọn tác vụ (Bài 1/2/3/4) và điều chỉnh tham số nếu có.
3) Ở cuối trang, chọn ảnh muốn xuất trong danh sách và bấm “Tải ảnh đã chọn (PNG)”.

