import os
import io
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
 

# =============================
# TIỆN ÍCH XỬ LÝ ẢNH
# =============================

# ----------BÀI 1: NHIỄU ----------

def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, std: float = 25.0):
	"""
	Thêm nhiễu Gaussian vào ảnh xám hoặc ảnh màu.
	- image: np.ndarray (H,W) hoặc (H,W,3) kiểu uint8
	- mean, std: tham số nhiễu
	"""
	noise = np.random.normal(mean, std, image.shape)
	noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
	return noisy


def add_salt_pepper(img, prob=0.02):
    """
    Thêm nhiễu muối tiêu (Salt & Pepper) vào ảnh xám hoặc ảnh màu.
    - img: np.ndarray (H, W) hoặc (H, W, 3) kiểu uint8
    - prob: xác suất tổng (trong khoảng 0..1) pixel bị nhiễu; mỗi loại muối/tiêu ~ prob/2
    """
    noisy = img.copy()
    H, W = img.shape[:2]
    rnd = np.random.rand(H, W)

    # Tạo mặt nạ cho pixel bị đặt về 0 (tiêu) và 255 (muối)
    mask_pepper = rnd < (prob / 2) #pepper (đen)
    mask_salt   = rnd > 1 - (prob / 2) #salt (trắng)

    if img.ndim == 2:  # ảnh xám
        noisy[mask_pepper] = 0
        noisy[mask_salt]   = 255
    else:  # ảnh màu (H, W, 3)
        noisy[mask_pepper] = [0, 0, 0]
        noisy[mask_salt]   = [255, 255, 255]

    return noisy


# ---------- BỘ LỌC LÀM MỜ ----------

def mean_filter(image: np.ndarray, ksize: int = 5):
    """Làm mờ trung bình (from scratch)."""
    kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    return cv2.filter2D(image,-1, kernel)



def gaussian_filter(image: np.ndarray, ksize: int = 5, sigma: float = 1.0):
    """Làm mờ Gaussian (from scratch)."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    # Công thức Gaussian 2D với hệ số chuẩn hóa
    kernel = (1.0 / (2.0 * np.pi * sigma**2)) * np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))

    # Chuẩn hóa sao cho tổng = 1 (đảm bảo không đổi độ sáng ảnh)
    kernel = kernel / np.sum(kernel)

    # Dùng hàm tích chập tự viết (không dùng cv2.filter2D)
    return cv2.filter2D(image, -1, kernel.astype(np.float32))



def median_filter(image: np.ndarray, ksize: int = 5):
    """Lọc Median dùng OpenCV; hỗ trợ ảnh xám/màu, kernel lẻ."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur(image, ksize)



def bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75):
	"""Bộ lọc Bilateral: làm mờ đồng thời vẫn giữ biên.
	Tham số:
	- image: np.ndarray (H,W) hoặc (H,W,3) kiểu uint8
	- d: đường kính vùng lân cận (pixel). Nếu d <= 0, OpenCV suy ra từ sigmaSpace.
	- sigma_color: độ lệch chuẩn trong không gian màu; lớn hơn → làm mờ mạnh hơn theo màu.
	- sigma_space: độ lệch chuẩn trong không gian toạ độ; lớn hơn → ảnh hưởng xa hơn.
	"""
	return cv2.bilateralFilter(image, d, sigma_color, sigma_space) 


# ---------- ĐÁNH GIÁ CHẤT LƯỢNG ----------

def compute_psnr(orig: np.ndarray, proc: np.ndarray) -> float:
	"""Tính PSNR giữa ảnh gốc và ảnh xử lý."""
	return psnr(orig, proc, data_range=255)


def compute_ssim(orig: np.ndarray, proc: np.ndarray) -> float:
	"""Tính SSIM giữa ảnh gốc và ảnh xử lý (yêu cầu ảnh xám).
	Trả về: float
	"""
	if orig.ndim == 3:
		orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
	else:
		orig_gray = orig
	if proc.ndim == 3:
		proc_gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
	else:
		proc_gray = proc
	return ssim(orig_gray, proc_gray, data_range=255)


# ----------BÀI 2: PHÁT HIỆN BIÊN (TỪ ĐẦU) ----------
def pad_reflect(img, pad):
    """Padding phản chiếu ở biên để giảm artefact khi lọc."""
    return np.pad(img, ((pad,pad),(pad,pad)), mode='reflect')

def convolve2d(img, kernel):
    """Tự cài đặt tích chập 2D (valid trên ảnh, có pad để giữ kích thước).
    - img: (H,W)
    - kernel: (k,k)
    Trả về: (H,W)
    """
    img = np.asarray(img, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    kh, kw = k.shape
    assert kh == kw and kh % 2 == 1, "Kernel phải là ma trận vuông kích thước lẻ."
    r = kh//2
    padded = pad_reflect(img, r)
    H, W = img.shape
    out = np.zeros_like(img, dtype=np.float64)
    kf = np.flip(np.flip(k, 0), 1)  # lật kernel cho chuẩn tích chập
    for i in range(H):
        for j in range(W):
            region = padded[i:i+kh, j:j+kw]
            out[i,j] = np.sum(region * kf)
    return out


def sobel_from_scratch(image: np.ndarray, threshold: float = 50.0):
    """Sobel : trả về edges, gx, gy, mag.
    - gx: gradient theo X (phát hiện biên dọc)
    - gy: gradient theo Y (phát hiện biên ngang)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Kernel Sobel
    sobel_x = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)   # Gx (biên dọc)

    sobel_y = np.array([[-1,  0,  1],
                        [-2,  0,  2],
                        [-1,  0,  1]], dtype=np.float32)   # Gy (biên ngang)

    gx = convolve2d(gray, sobel_x)
    gy = convolve2d(gray, sobel_y)
    mag = np.sqrt(gx ** 2 + gy ** 2)  # kết hợp 2 hướng
    edges = (mag > threshold).astype(np.uint8) * 255
    return edges.astype(np.uint8), gx.astype(np.float32), gy.astype(np.float32), mag.astype(np.float32)


def prewitt_from_scratch(image: np.ndarray, threshold: float = 50.0):
	"""Prewitt: trả về edges, gx, gy, mag."""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
	px = np.array([[-1, 0, 1],
				  [-1, 0, 1],
				  [-1, 0, 1]], dtype=np.float32)
	py = np.array([[-1, -1, -1],
				  [ 0,  0,  0],
				  [ 1,  1,  1]], dtype=np.float32)
	gx = convolve2d(gray, px)
	gy = convolve2d(gray, py)
	mag = np.sqrt(gx ** 2 + gy ** 2) # kết hợp 2 hướng
	edges = (mag > threshold).astype(np.uint8) * 255
	return edges.astype(np.uint8), gx.astype(np.float32), gy.astype(np.float32), mag.astype(np.float32)


def laplacian_from_scratch(image: np.ndarray, threshold: float = 50.0):
	"""Laplacian: trả về edges, response."""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
	lap = np.array([[0, -1, 0],
				   [-1, 4, -1],
				   [0, -1, 0]], dtype=np.float32)
	resp = convolve2d(gray, lap)
	edges = (np.abs(resp) > threshold).astype(np.uint8) * 255
	return edges.astype(np.uint8), resp.astype(np.float32)


def canny_reference(image: np.ndarray, low: int = 50, high: int = 150):
	"""So sánh với Canny của OpenCV."""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
	return cv2.Canny(gray, low, high)


# ----------BÀI 3: TĂNG CƯỜNG ẢNH ----------

def laplacian_sharpen(image: np.ndarray, alpha: float = 0.5):
	"""Làm sắc nét bằng Laplacian: ưu tiên giữ màu (xử lý kênh sáng trong LAB)."""
	if image.ndim == 3:
		lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
		L, A, B = cv2.split(lab)
		lap_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
		lap = convolve2d(L, lap_kernel)
		L_sharp = np.clip(L.astype(np.float32) + alpha * (L.astype(np.float32) - lap), 0, 255).astype(np.uint8)
		lab_sharp = cv2.merge([L_sharp, A, B])
		return cv2.cvtColor(lab_sharp, cv2.COLOR_LAB2BGR)
	else:
		gray = image
		lap_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
		lap = convolve2d(gray, lap_kernel)
		sharp = np.clip(gray.astype(np.float32) + alpha * (gray.astype(np.float32) - lap), 0, 255)
		return sharp.astype(np.uint8)


def unsharp_mask(image: np.ndarray, ksize: int = 5, sigma: float = 1.0, amount: float = 1.5):
	"""Unsharp Masking: xử lý kênh sáng để giữ màu; trả về ảnh màu nếu đầu vào màu."""
	if image.ndim == 3:
		lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
		L, A, B = cv2.split(lab)
		blurred = gaussian_filter(L, ksize=ksize, sigma=sigma)
		high_boost = L.astype(np.float32) - blurred.astype(np.float32)
		L_sharp = np.clip(L.astype(np.float32) + amount * high_boost, 0, 255).astype(np.uint8)
		lab_sharp = cv2.merge([L_sharp, A, B])
		return cv2.cvtColor(lab_sharp, cv2.COLOR_LAB2BGR)
	else:
		gray = image
		blurred = gaussian_filter(gray, ksize=ksize, sigma=sigma)
		high_boost = gray.astype(np.float32) - blurred.astype(np.float32)
		sharp = np.clip(gray.astype(np.float32) + amount * high_boost, 0, 255)
		return sharp.astype(np.uint8)




def clahe_equalization(image: np.ndarray, clip: float = 2.0, tile: int = 8):
	"""Cân bằng lược đồ thích nghi (CLAHE) giữ màu bằng cách tăng cường kênh sáng (LAB).
	Trả về: np.ndarray (uint8, BGR hoặc Gray)
	"""
	# Chuẩn hoá tileGridSize
	tile_size = (tile, tile) if not isinstance(tile, tuple) else tile
	if image.ndim == 3:
		lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
		L, A, B = cv2.split(lab)
		clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile_size)
		L_eq = clahe.apply(L)
		lab_eq = cv2.merge([L_eq, A, B])
		return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
	else:
		gray = image
		clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile_size)
		return clahe.apply(gray)


# ---------- BÀI 4: ẢNH Y TẾ  ----------
def medical_edge_detection_pipeline(bgr_image: np.ndarray, gaussian_ksize: int = 5, gaussian_sigma: float = 1.0,
									   threshold: int = 50, canny_low: int = 30, canny_high: int = 80):
	"""
	Pipeline phát hiện biên cho ảnh y tế dùng các phương pháp ở Bài 2.
	Trả về: dict gồm gray, denoised, s_edges, p_edges, l_edges, c_edges
	"""
	gray = ensure_gray(bgr_image)
	denoised = gaussian_filter(gray, ksize=gaussian_ksize, sigma=gaussian_sigma)
	s_edges, _, _, _ = sobel_from_scratch(denoised, threshold)
	p_edges, _, _, _ = prewitt_from_scratch(denoised, threshold)
	l_edges, _ = laplacian_from_scratch(gray, threshold)
	c_edges = canny_reference(denoised, canny_low, canny_high)
	return {
		"gray": gray,
		"denoised": denoised,
		"s_edges": s_edges,
		"p_edges": p_edges,
		"l_edges": l_edges,
		"c_edges": c_edges,
	}


# ---------- HỖ TRỢ VẼ HISTOGRAM ----------

def plot_hist(image: np.ndarray, title: str = "Histogram") -> plt.Figure:
	"""Trả về Figure matplotlib của histogram ảnh (xám)."""
	if image.ndim == 3:
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	else:
		gray = image
	fig, ax = plt.subplots(figsize=(4, 3))
	hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
	ax.plot(hist, color='black')
	ax.set_title(title)
	ax.set_xlabel('Giá trị mức xám')
	ax.set_ylabel('Tần suất')
	fig.tight_layout()
	return fig


# ---------- TIỆN ÍCH UI ----------

def to_image(arr: np.ndarray) -> Image.Image:
	"""Chuyển mảng numpy (BGR hoặc Gray) sang PIL.Image để hiển thị.
	Trả về: PIL.Image.Image
	"""
	if arr.ndim == 2:
		return Image.fromarray(arr)
	# BGR -> RGB
	rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
	return Image.fromarray(rgb)


def ensure_gray(image: np.ndarray):
	"""Đảm bảo ảnh xám.
	Trả về: np.ndarray (uint8)
	"""
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image



# =============================
# STREAMLIT APP
# =============================

st.set_page_config(page_title="Mini Photo Editor - Image Processing", layout="wide")
st.title("Mini Photo Editor")
st.caption("Làm mờ, Phát hiện biên, Tăng cường ảnh, Ứng dụng y tế")

# Sidebar: tải ảnh và chọn tác vụ
with st.sidebar:
	#st.header("Cấu hình")
	uploaded = st.file_uploader("Tải ảnh (PNG/JPG)", type=["png", "jpg", "jpeg"]) 

	if uploaded is not None:
		# Chỉ hỗ trợ PNG/JPG: decode trực tiếp bằng OpenCV
		file_bytes = uploaded.read()
		arr = np.frombuffer(file_bytes, np.uint8)
		bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
		if bgr is None:
			st.error("Không đọc được ảnh. Vui lòng dùng PNG/JPG.")
			bgr = None
	else:
		bgr = None

	task = st.selectbox(
		"Chọn tác vụ",
		[
			"So sánh bộ lọc làm mờ",
			"Phát hiện biên",
			"Tăng cường ảnh",
			"Ảnh y tế (demo pipeline)",
		]
	)

if bgr is None:
	st.info("Vui lòng tải ảnh (PNG/JPG) ở Sidebar để bắt đầu.")
	st.stop()

# Hiển thị ảnh gốc
col1, col2 = st.columns([2, 1])
with col1:
	st.subheader("Ảnh gốc")
	st.image(to_image(bgr), use_column_width=True)
with col2:
	st.subheader("Histogram ảnh gốc")
	st.pyplot(plot_hist(bgr, title="Histogram trước xử lý"))

# =============== BÀI 1 ===============
if task == "So sánh bộ lọc làm mờ":
	st.subheader("So sánh và phân tích các bộ lọc làm mờ")

	# Thêm nhiễu
	noise_type = st.radio("Chọn loại nhiễu", ["Gaussian noise", "Salt & Pepper"], horizontal=True)
	if noise_type == "Gaussian noise":
		std = st.slider("Độ lệch chuẩn (std)", 1, 60, 25, 1)
		noisy = add_gaussian_noise(bgr, std=std)
	else:
		prob = st.slider("Xác suất nhiễu (prob)", 0.0, 0.2, 0.02, 0.01)
		noisy = add_salt_pepper(bgr, prob=prob)

	st.markdown("**Ảnh có nhiễu**")
	st.image(to_image(noisy), use_column_width=True)
	#st.pyplot(plot_hist(noisy, title="Histogram sau thêm nhiễu"))

	# Tham số bộ lọc
	st.markdown("---")
	st.sidebar.markdown("**Tham số của các bộ lọc**")
	st.markdown("**Kết quả sau lọc**")
	ksize = st.sidebar.slider("Kích thước kernel (lẻ)", 3, 21, 5, 2)
	sigma = st.sidebar.slider("Sigma (Gaussian)", 0.1, 5.0, 1.0, 0.1)
	d_bi = st.sidebar.slider("d (Bilateral)", 3, 21, 9, 2)
	sc_bi = st.sidebar.slider("sigmaColor (Bilateral)", 10, 150, 75, 5)
	ss_bi = st.sidebar.slider("sigmaSpace (Bilateral)", 10, 150, 75, 5)

	# Áp dụng
	mean_img = mean_filter(noisy, ksize)
	gaus_img = gaussian_filter(noisy, ksize, sigma)
	medi_img = median_filter(noisy, ksize)
	bila_img = bilateral_filter(noisy, d_bi, sc_bi, ss_bi)

	# Đánh giá PSNR/SSIM (so với ảnh gốc)
	psnrs = {}
	ssims = {}
	for name, img in {
		"Mean": mean_img,
		"Gaussian": gaus_img,
		"Median": medi_img,
		"Bilateral": bila_img,
	}.items():
		psnrs[name] = compute_psnr(ensure_gray(bgr), ensure_gray(img))
		ssims[name] = compute_ssim(bgr, img)

	

	# Hiển thị 4 ảnh sau lọc theo hàng ngang
	col_a, col_b, col_c, col_d = st.columns(4)
	with col_a:
		st.image(to_image(mean_img), caption=f"Mean (kernel={ksize})", use_column_width=True)
	with col_b:
		st.image(to_image(gaus_img), caption=f"Gaussian (kernel={ksize}, sigma={sigma})", use_column_width=True)
	with col_c:
		st.image(to_image(medi_img), caption=f"Median (kernel={ksize})", use_column_width=True)
	with col_d:
		st.image(to_image(bila_img), caption=f"Bilateral (d={d_bi}, sc={sc_bi}, ss={ss_bi})", use_column_width=True)

	st.markdown("---")
	st.markdown("**So sánh kết quả (PSNR / SSIM)**")
	st.write(f"Mean (kernel={ksize}): PSNR={psnrs['Mean']:.2f} dB, SSIM={ssims['Mean']:.4f}")
	st.write(f"Gaussian (kernel={ksize}, sigma={sigma}): PSNR={psnrs['Gaussian']:.2f} dB, SSIM={ssims['Gaussian']:.4f}")
	st.write(f"Median (kernel={ksize}): PSNR={psnrs['Median']:.2f} dB, SSIM={ssims['Median']:.4f}")
	st.write(f"Bilateral (d={d_bi}, sc={sc_bi}, ss={ss_bi}): PSNR={psnrs['Bilateral']:.2f} dB, SSIM={ssims['Bilateral']:.4f}")
	
	st.markdown("---")
	st.subheader("Histogram trước/sau lọc")
	# Histogram của ảnh gốc, ảnh sau khi thêm nhiễu và ảnh sau lọc
	h0c1, h0c2 = st.columns(2)
	with h0c1:
		st.pyplot(plot_hist(bgr, title="Gốc"))
	with h0c2:
		st.pyplot(plot_hist(noisy, title="Sau thêm nhiễu (" + noise_type + ")"))
	h_cols = st.columns(4)
	with h_cols[0]:
		st.pyplot(plot_hist(mean_img, title=f"Mean (kernel={ksize})"))
	with h_cols[1]:
		st.pyplot(plot_hist(gaus_img, title=f"Gaussian (kernel={ksize}, sigma={sigma})"))
	with h_cols[2]:
		st.pyplot(plot_hist(medi_img, title=f"Median (kernel={ksize})"))
	with h_cols[3]:
		st.pyplot(plot_hist(bila_img, title=f"Bilateral (d={d_bi}, sc={sc_bi}, ss={ss_bi})"))

# =============== BÀI 2 ===============
elif task == "Phát hiện biên":
	st.subheader("Edge Detection")
	gray = ensure_gray(bgr)

	thr = st.sidebar.slider("Ngưỡng (threshold)", 1, 200, 50, 1)
	low = st.sidebar.slider("Canny low", 1, 200, 50, 1)
	high = st.sidebar.slider("Canny high", 1, 300, 150, 1)

	s_edges, s_gx, s_gy, s_mag = sobel_from_scratch(bgr, thr)
	p_edges, p_gx, p_gy, p_mag = prewitt_from_scratch(bgr, thr)
	l_edges, l_resp = laplacian_from_scratch(bgr, thr)
	c_edges = canny_reference(bgr, low, high)

		# Pipeline Prewitt
	st.markdown("**Prewitt – các bước**")
	pc0, pc1, pc2, pc3, pc4 = st.columns(5)
	with pc0:
		st.image(to_image(gray), caption="Ảnh xám gốc", use_column_width=True)
	with pc1:
		st.image(to_image((np.abs(p_gx) / (np.max(np.abs(p_gx))+1e-5) * 255).astype(np.uint8)), caption="Gradient X", use_column_width=True)
	with pc2:
		st.image(to_image((np.abs(p_gy) / (np.max(np.abs(p_gy))+1e-5) * 255).astype(np.uint8)), caption="Gradient Y", use_column_width=True)
	with pc3:
		st.image(to_image((p_mag / (np.max(p_mag)+1e-5) * 255).astype(np.uint8)), caption="Magnitude", use_column_width=True)
	with pc4:
		st.image(to_image(p_edges), caption="Thresholded (" + str(thr) + ")", use_column_width=True)

	# Hiển thị pipeline: gray, gx, gy, magnitude, threshold
	st.markdown("**Sobel – các bước**")
	c0, c1, c2, c3, c4 = st.columns(5)
	with c0:
		st.image(to_image(gray), caption="Ảnh xám gốc", use_column_width=True)
	with c1:
		st.image(to_image((np.abs(s_gx) / (np.max(np.abs(s_gx))+1e-5) * 255).astype(np.uint8)), caption="Gradient X", use_column_width=True)
	with c2:
		st.image(to_image((np.abs(s_gy) / (np.max(np.abs(s_gy))+1e-5) * 255).astype(np.uint8)), caption="Gradient Y", use_column_width=True)
	with c3:
		st.image(to_image((s_mag / (np.max(s_mag)+1e-5) * 255).astype(np.uint8)), caption="Magnitude", use_column_width=True)
	with c4:
		st.image(to_image(s_edges), caption="Thresholded (" + str(thr) + ")", use_column_width=True)



	# Pipeline Laplacian
	st.markdown("**Laplacian – các bước**")
	lc0, lc1, lc2, lc3 = st.columns(4)
	with lc0:
		st.image(to_image(gray), caption="Ảnh xám gốc", use_column_width=True)
	with lc1:
		st.image(to_image(((l_resp - l_resp.min()) / (l_resp.max() - l_resp.min() + 1e-5) * 255).astype(np.uint8)), caption="Response (chuẩn hoá)", use_column_width=True)
	with lc2:
		st.image(to_image((np.abs(l_resp) / (np.max(np.abs(l_resp))+1e-5) * 255).astype(np.uint8)), caption="|Response|", use_column_width=True)
	with lc3:
		st.image(to_image(l_edges), caption="Thresholded (" + str(thr) + ")", use_column_width=True)

	st.markdown("---")
	st.markdown("**So sánh phương pháp**")
	cc1, cc2, cc3, cc4, cc5 = st.columns(5)
	with cc1:
		st.image(to_image(gray),caption="Ảnh xám gốc", use_column_width=True)
	with cc2:
		st.image(to_image(p_edges), caption="Prewitt", use_column_width=True)
	with cc3:
		st.image(to_image(s_edges), caption="Sobel", use_column_width=True)
	with cc4:
		st.image(to_image(l_edges), caption="Laplacian", use_column_width=True)
	with cc5:
		st.image(to_image(c_edges), caption="Canny (tham chiếu)", use_column_width=True)

	# Phân tích độ nhạy ngưỡng
	st.markdown("---")
	st.subheader("Độ nhạy tham số ngưỡng (Sobel)")
	ths = st.multiselect("Chọn nhiều ngưỡng để so sánh", [10, 30, 50, 70, 90, 110, 130, 150], default=[10, 50, 90])
	cols = st.columns(len(ths) if len(ths) > 0 else 1)
	for col, t in zip(cols, ths):
		ed, _, _, _ = sobel_from_scratch(bgr, t)
		with col:
			st.image(to_image(ed), caption=f"Threshold={t}", use_column_width=True)

# =============== BÀI 3 ===============
elif task == "Tăng cường ảnh":
	st.subheader("Image Enhancement")
	st.markdown("Mục tiêu: làm sắc nét (Laplacian/Unsharp) + cân bằng histogram để cải thiện ảnh mờ/thiếu sáng.")

	# Tham số
	alpha = st.sidebar.slider("Alpha (Laplacian sharpen)", 0.1, 3.0, 0.5, 0.1)
	ks = st.sidebar.slider("Kernel Gaussian (Unsharp Masking)", 3, 21, 5, 2)
	sig = st.sidebar.slider("Sigma Gaussian (Unsharp Masking)", 0.1, 5.0, 1.0, 0.1)
	amt = st.sidebar.slider("Amount (Unsharp Masking)", 0.1, 3.0, 1.5, 0.1)
	

	lap_sharp = laplacian_sharpen(bgr, alpha)
	unsharp = unsharp_mask(bgr, ksize=ks, sigma=sig, amount=amt)
	lap_sharp_eq = clahe_equalization(lap_sharp, clip=2.0, tile=(8,8)) 
	unsharp_eq = clahe_equalization(unsharp, clip=2.0, tile=(8,8))

	c1, c2, c3, c4, c5 = st.columns(5)
	with c1:
		st.image(to_image(bgr), caption="Original", use_column_width=True)
	with c2:
		st.image(to_image(lap_sharp), caption="Laplacian sharpen", use_column_width=True)
	with c3:
		st.image(to_image(lap_sharp_eq), caption="Laplacian sharpen with CLAHE", use_column_width=True)
	with c4:
		st.image(to_image(unsharp), caption="Unsharp Masking", use_column_width=True)
	with c5:
		st.image(to_image(unsharp_eq), caption="Unsharp Masking with CLAHE", use_column_width=True)

	# Histogram của 4 ảnh 
	hc1, hc2, hc3, hc4, hc5 = st.columns(5)
	with hc1:
		st.pyplot(plot_hist(bgr, title="Hist: Original"))
	with hc2:
		st.pyplot(plot_hist(lap_sharp, title="Hist: Laplacian"))
	with hc3:
		st.pyplot(plot_hist(lap_sharp_eq, title="Hist: Lap+CLAHE"))
	with hc4:
		st.pyplot(plot_hist(unsharp, title="Hist: Unsharp"))
	with hc5:
		st.pyplot(plot_hist(unsharp_eq, title="Hist: Unsharp+CLAHE"))

	st.markdown("Giải thích: CLAHE tăng tương phản cục bộ ở vùng thiếu sáng; Unsharp tăng chi tiết biên sau khi tương phản được cải thiện → ảnh sắc nét và rõ ràng hơn.")

	# Gợi ý workflow cho ảnh điện thoại: giảm nhiễu nhẹ -> CLAHE -> Unsharp
	st.markdown("---")
	st.markdown("**Workflow gợi ý (ảnh điện thoại):** Gaussian mờ nhẹ → CLAHE → Unsharp")
	phone_sigma = st.sidebar.slider("Sigma Gaussian (Workflow)", 0.0, 2.0, 0.6, 0.1)
	pre_denoise = gaussian_filter(bgr, ksize=5, sigma=phone_sigma) if phone_sigma > 0 else bgr
	phone_eq = clahe_equalization(pre_denoise, clip=2.0, tile=(8,8))
	phone_sharp = unsharp_mask(phone_eq, ksize=ks, sigma=sig, amount=amt)



	st.markdown("---")
	st.subheader("Demo ảnh mờ → sắc nét (Workflow điện thoại)")
	w1, w2, w3 = st.columns(3)
	with w1:
		st.image(to_image(pre_denoise), caption="B1: Giảm nhiễu nhẹ", use_column_width=True)
	with w2:
		st.image(to_image(phone_eq), caption="B2: CLAHE", use_column_width=True)
	with w3:
		st.image(to_image(phone_sharp), caption="B3: Unsharp", use_column_width=True)

	

# =============== BÀI 4 ===============
else:
	st.subheader("Phát hiện biên trên ảnh y tế")
	st.markdown("Nguồn dữ liệu công khai: Kaggle/NIH")

	# Tham số cho pipeline y tế
	gaussian_ksize = st.sidebar.slider("Gaussian kernel (lẻ)", 3, 21, 5, 2)
	gaussian_sigma = st.sidebar.slider("Gaussian sigma", 0.1, 5.0, 1.0, 0.1)
	threshold = st.sidebar.slider("Threshold (Sobel/Prewitt/Laplacian)", 1, 200, 50, 1)
	canny_low = st.sidebar.slider("Canny low", 1, 200, 30, 1)
	canny_high = st.sidebar.slider("Canny high", 1, 300, 80, 1)

	# Gọi hàm pipeline bài 4
	med = medical_edge_detection_pipeline(bgr, gaussian_ksize=gaussian_ksize, gaussian_sigma=gaussian_sigma,
										 threshold=threshold, canny_low=canny_low, canny_high=canny_high)

	# Hiển thị pipeline đầu vào và kết quả biên
	m1, m2, m3 = st.columns(3)
	with m1:
		st.image(to_image(med["gray"]), caption="Ảnh y tế (xám)", use_column_width=True)
	with m2:
		st.image(to_image(med["denoised"]), caption="Sau Gaussian smoothing", use_column_width=True)
	with m3:
		st.image(to_image(med["s_edges"]), caption="Biên (Sobel)", use_column_width=True)

	st.markdown("---")
	st.subheader("So sánh các phương pháp phát hiện biên")
	cx1, cx2, cx3, cx4 = st.columns(4)
	with cx1:
		st.image(to_image(med["s_edges"]), caption="Sobel (threshold=" + str(threshold) + ")", use_column_width=True)
	with cx2:
		st.image(to_image(med["p_edges"]), caption="Prewitt (threshold=" + str(threshold) + ")", use_column_width=True)
	with cx3:
		st.image(to_image(med["l_edges"]), caption="Laplacian (threshold=" + str(threshold) + ")", use_column_width=True)
	with cx4:
		st.image(to_image(med["c_edges"]), caption="Canny", use_column_width=True)

	st.markdown("Nhận xét: Làm mờ Gaussian giúp giảm nhiễu trước khi phát hiện biên; Sobel/Prewitt/Laplacian nhạy với nhiễu hơn Canny, nên bước khử nhiễu là cần thiết.")

# ---------- LƯU/XUẤT ẢNH ----------
st.markdown("---")
st.subheader("Xuất/Lưu ảnh")

# Cho phép người dùng tự chọn ảnh muốn xuất theo từng tác vụ
export_options = {}
if task == "So sánh bộ lọc làm mờ":
	export_options = {
		"Mean": mean_img,
		"Gaussian": gaus_img,
		"Median": medi_img,
		"Bilateral": bila_img,
	}
elif task == "Phát hiện biên":
	export_options = {
		"Sobel - Edges": s_edges,
		"Prewitt - Edges": p_edges,
		"Laplacian - Edges": l_edges,
		"Canny": c_edges,
		
	}
elif task == "Tăng cường ảnh":
	export_options = {
		"Laplacian sharpen": lap_sharp,
		"Unsharp Masking": unsharp,
		"Laplacian sharpen with CLAHE": lap_sharp_eq,
		"Unsharp Masking with CLAHE": unsharp_eq,
		"Workflow - B1 Giảm nhiễu nhẹ": pre_denoise,
		"Workflow - B2 CLAHE": phone_eq,
		"Workflow - B3 Unsharp": phone_sharp,
	}
else:
	export_options = {
		"Ảnh y tế (xám)": med["gray"],
		"Sau Gaussian smoothing": med["denoised"],
		"Biên (Sobel)": med["s_edges"],
		"Biên (Prewitt)": med["p_edges"],
		"Biên (Laplacian)": med["l_edges"],
		"Biên (Canny)": med["c_edges"],
	}

selected_export = st.selectbox("Chọn ảnh để xuất", list(export_options.keys()))
export_img = export_options.get(selected_export)

if export_img is not None:
	if export_img.ndim == 2:
		img_to_save = export_img
	else:
		img_to_save = cv2.cvtColor(export_img, cv2.COLOR_BGR2RGB)
	pil_img = Image.fromarray(img_to_save)
	buf = io.BytesIO()
	pil_img.save(buf, format="PNG")
	byte_im = buf.getvalue()
	st.download_button("Tải ảnh đã chọn (PNG)", data=byte_im, file_name="output.png", mime="image/png")


