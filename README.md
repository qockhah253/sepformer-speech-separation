# 🎙️ SepFormer — Phân tách nguồn âm thanh đơn kênh

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![HuggingFace Demo](https://img.shields.io/badge/🤗%20Demo-HuggingFace-yellow)](https://huggingface.co/spaces/quockhanh25032005/sepformer-demo)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Triển khai kiến trúc **SepFormer (Separation Transformer)** cho bài toán phân tách nguồn âm thanh đơn kênh — tách biệt giọng nói của từng người từ một file âm thanh hỗn hợp chứa nhiều người nói cùng lúc.

---

## 🎯 Demo trực tuyến

👉 **[Thử ngay tại đây](https://huggingface.co/spaces/quockhanh25032005/sepformer-demo)**

Upload file âm thanh chứa 2 người nói → Model tự động tách thành 2 nguồn âm riêng biệt.

---

## 📖 Giới thiệu

### Bài toán

Bài toán **phân tách nguồn âm** (Speech Separation) hay còn gọi là **Cocktail Party Problem** — khả năng tập trung vào một nguồn âm duy nhất trong môi trường có nhiều người nói và tạp âm đan xen.

Về mặt toán học, bài toán được phát biểu như sau: Giả sử có N nguồn âm độc lập s₁(t), s₂(t), ..., sₙ(t), tín hiệu thu được tại micro là hỗn hợp tuyến tính:

```
x(t) = s₁(t) + s₂(t) + ... + sₙ(t)
```

Mục tiêu là ước lượng lại từng ŝᵢ(t) sao cho ŝᵢ(t) ≈ sᵢ(t), chỉ từ tín hiệu quan sát x(t).

### Tại sao SepFormer?

Các phương pháp truyền thống (ICA, NMF) gặp nhiều hạn chế khi áp dụng vào bài toán thực tế. Sự ra đời của học sâu đã mang lại bước ngoặt lớn — đặc biệt là **SepFormer** (2021), kiến trúc đầu tiên thay thế hoàn toàn RNN bằng Transformer trong bài toán phân tách nguồn âm, đạt kết quả vượt trội so với tất cả các phương pháp trước đó.

### Ứng dụng thực tiễn

- 🎤 **Nhận dạng giọng nói (ASR)** — Cải thiện độ chính xác trong môi trường ồn ào
- 🤖 **Trợ lý ảo** — Siri, Google Assistant, Alexa
- 📹 **Hội nghị truyền hình** — Tách biệt giọng từng người tham gia
- 👂 **Máy trợ thính** — Lọc bỏ tạp âm, khuếch đại giọng mục tiêu
- 🔍 **Phân tích pháp y âm thanh** — Phục hồi giọng nói từ bản ghi âm chất lượng thấp

---

## 🏗️ Kiến trúc

```
Tín hiệu hỗn hợp x(t)
        │
        ▼
┌─────────────┐
│   Encoder   │  ← 1D Conv học được (thay thế STFT)
└─────────────┘
        │  E ∈ ℝ^(N×T')
        ▼
┌──────────────────────────────────────┐
│          SepFormer Block             │
│                                      │
│   Chunking: E → E_chunk ∈ ℝ^(N×K×S) │
│                                      │
│  ┌────────────────────────────────┐  │
│  │      IntraTransformer          │  │
│  │  (Self-Attention trong chunk)  │  │  ← Học phụ thuộc cục bộ
│  │      O(S · K²)                 │  │
│  └────────────────────────────────┘  │
│                  ↓                   │
│  ┌────────────────────────────────┐  │
│  │      InterTransformer          │  │
│  │  (Self-Attention giữa chunk)   │  │  ← Học phụ thuộc toàn cục
│  │      O(K · S²)                 │  │
│  └────────────────────────────────┘  │
│         × R lần (R=2)                │
└──────────────────────────────────────┘
        │
        ▼
┌─────────────┐
│  Mask Net   │  ← PReLU + Linear + ReLU → M₁, M₂
└─────────────┘
        │  Mᵢ ∈ ℝ^(N×T')
        ▼
   Eᵢ = Mᵢ ⊙ E   (Element-wise multiplication)
        │
        ▼
┌─────────────┐
│   Decoder   │  ← 1D ConvTranspose tái tạo tín hiệu
└─────────────┘
        │
        ▼
  ŝ₁(t),  ŝ₂(t)
```

### Ưu điểm của Dual-Path Processing

Thay vì áp dụng Self-Attention trực tiếp lên toàn bộ chuỗi (độ phức tạp O(T²)), SepFormer chia tín hiệu thành các **chunk** nhỏ và xử lý theo 2 chiều:

| | IntraTransformer | InterTransformer |
|---|---|---|
| **Xử lý** | Bên trong từng chunk | Giữa các chunk |
| **Học** | Phụ thuộc cục bộ | Phụ thuộc toàn cục |
| **Độ phức tạp** | O(S · K²) | O(K · S²) |
| **Tổng** | **O(T^1.5)** thay vì O(T²) | |

---

## 📊 Kết quả thực nghiệm

### Quá trình huấn luyện

Mô hình được huấn luyện trên **Libri2Mix** với cấu hình:

| Siêu tham số | Giá trị |
|---|---|
| Số epoch | 10 |
| Batch size | 1 |
| Learning rate | 1.5×10⁻⁴ |
| Optimizer | Adam |
| Độ dài tín hiệu tối đa | 32.000 mẫu (4 giây) |
| Hàm mất mát | SI-SNR + PIT |
| GPU | NVIDIA Tesla T4 × 2 |

### Loss curve

| Epoch | Train SI-SNR (dB) | Valid SI-SNR (dB) |
|:---:|:---:|:---:|
| 1 | -15.2 | -16.6 |
| 2 | -16.8 | -17.2 |
| 3 | -17.2 | -17.3 |
| 4 | -17.5 | -17.7 |
| 5 | -17.6 | -17.8 |
| 6 | -17.8 | -17.9 |
| 7 | -18.0 | -18.0 |
| 8 | -18.0 | -18.1 |
| 9 | -18.2 | -18.1 |
| 10 | -18.2 | -18.3 |

> **Nhận xét:** SI-SNR giảm đều đặn qua các epoch, tốc độ cải thiện chậm dần từ epoch 7 cho thấy mô hình đang hội tụ. Khoảng cách nhỏ giữa train và valid SI-SNR cho thấy không có hiện tượng overfitting.

### So sánh với các mô hình khác

| Mô hình | Năm | Dataset | SI-SNRi (dB) | SDRi (dB) | Tham số |
|---------|:---:|---------|:---:|:---:|:---:|
| Conv-TasNet | 2019 | WSJ0-2mix | 15.3 | 15.6 | 5.1M |
| DPRNN | 2020 | WSJ0-2mix | 18.8 | 19.0 | 2.6M |
| SepFormer (gốc) | 2021 | WSJ0-2mix | 22.3 | 22.4 | 25.7M |
| **SepFormer (đề tài)** | **2024** | **Libri2Mix** | **21.60** | **—** | **25.7M** |

> **Kết quả 21.60 dB SI-SNRi** vượt trội so với Conv-TasNet (+6.3 dB) và DPRNN (+2.8 dB), xấp xỉ kết quả gốc của SepFormer (22.3 dB trên WSJ0-2mix).

### Phân tích kết quả

- ✅ **Vượt trội so với các phương pháp cũ** — Cải thiện 6.3 dB so với Conv-TasNet
- ✅ **Không overfitting** — Khoảng cách train/valid ổn định
- ✅ **Hội tụ ổn định** — Loss giảm đều qua 10 epoch
- ⚠️ **Chênh lệch nhỏ so với gốc** — Do khác dataset (Libri2Mix vs WSJ0-2mix) và ít epoch hơn (10 vs 200+)

---

## 📁 Cấu trúc thư mục

```
sepformer-speech-separation/
│
├── model/
│   ├── encoder.py          # Learned Encoder (1D Conv)
│   ├── decoder.py          # Learned Decoder (1D ConvTranspose)
│   ├── sepformer_block.py  # Dual-Path Transformer Block
│   └── sepformer.py        # Kiến trúc SepFormer hoàn chỉnh
│
├── utils/
│   ├── dataset.py          # Libri2Mix Dataset loader
│   └── metrics.py          # SI-SNR, SI-SNRi, PIT
│
├── train.py                # Script huấn luyện
├── evaluate.py             # Script đánh giá
├── app.py                  # Web demo (Gradio)
└── requirements.txt        # Thư viện cần thiết
```

---

## ⚙️ Cài đặt

```bash
# Clone repository
git clone https://github.com/qockhah253/sepformer-speech-separation
cd sepformer-speech-separation

# Cài đặt thư viện
pip install -r requirements.txt
```

**Yêu cầu:**
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (khuyến nghị để train)

---

## 🚀 Sử dụng

### Huấn luyện

```bash
python train.py
```

Cấu hình mặc định trong `train.py`:
```python
DATA_DIR    = "librimix_data"   # Đường dẫn đến Libri2Mix
EPOCHS      = 10
BATCH_SIZE  = 1
LR          = 1.5e-4
SAMPLE_RATE = 8000
MAX_LEN     = 32000             # 4 giây
```

### Đánh giá

```bash
python evaluate.py
```

### Chạy Web Demo (local)

```bash
python app.py
```

Sau đó mở trình duyệt tại `http://localhost:7860`

### Sử dụng model trong code

```python
import torch
from model.sepformer import SepFormer

# Khởi tạo model
model = SepFormer(
    num_filters=256,
    filter_length=16,
    stride=8,
    d_model=256,
    nhead=8,
    num_spks=2
)

# Load checkpoint
checkpoint = torch.load("checkpoints/checkpoint_epoch_10.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Phân tách nguồn âm
with torch.no_grad():
    mixture = torch.randn(1, 32000)   # (batch, T)
    sources = model(mixture)           # (batch, 2, T)
    source1 = sources[0, 0, :]        # Nguồn âm thứ nhất
    source2 = sources[0, 1, :]        # Nguồn âm thứ hai
```

---

## 🛠️ Công nghệ sử dụng

| Công nghệ | Phiên bản | Mục đích |
|-----------|:---:|---------|
| PyTorch | 2.0+ | Framework học sâu |
| SpeechBrain | 1.1.0 | Công cụ xử lý tiếng nói |
| Librosa | 0.11.0 | Xử lý tín hiệu âm thanh |
| Gradio | 6.x | Giao diện web demo |
| Kaggle Notebooks | — | Môi trường huấn luyện (GPU T4 x2) |

---

## 📚 Tài liệu tham khảo

1. **Subakan et al. (2021)** — [Attention is All You Need in Speech Separation](https://arxiv.org/abs/2010.13154) — *Bài báo gốc SepFormer*
2. **Luo & Mesgarani (2019)** — [Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking](https://arxiv.org/abs/1809.07454)
3. **Luo et al. (2020)** — [Dual-Path RNN: Efficient Long Sequence Modeling](https://arxiv.org/abs/1910.06379)
4. **Vaswani et al. (2017)** — [Attention is All You Need](https://arxiv.org/abs/1706.03762) — *Transformer gốc*
5. **Cosentino et al. (2020)** — [LibriMix: An Open-Source Dataset for Generalizable Speech Separation](https://arxiv.org/abs/2005.11262)
6. **Hershey et al. (2016)** — [Deep Clustering: Discriminative Embeddings for Segmentation and Separation](https://arxiv.org/abs/1508.04306)
7. **Yu et al. (2017)** — [Permutation Invariant Training of Deep Models for Speaker-Independent Multi-talker Speech Separation](https://arxiv.org/abs/1607.00325)

---

## 👨‍💻 Tác giả

| | |
|---|---|
| **Sinh viên** | Triệu Quốc Khánh |
| **Giảng viên hướng dẫn** | CN. Vi Anh Quân |
| **Trường** | Đại học Khoa học Tự nhiên — ĐHQGHN |
| **Khoa** | Vật lý — Bộ môn Tin học Vật lý |
| **Năm** | 2024 |

---

<div align="center">
  <i>Nếu thấy hữu ích, hãy ⭐ star repo này!</i>
</div>
