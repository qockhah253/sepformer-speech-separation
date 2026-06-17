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

Bài toán **phân tách nguồn âm** (Speech Separation) hay còn gọi là **Cocktail Party Problem** — khả năng tập trung vào một nguồn âm duy nhất trong môi trường có nhiều người nói và tạp âm đan xen.

**SepFormer** (Subakan et al., 2021) là một trong những mô hình học sâu đạt hiệu năng hàng đầu cho bài toán này, khai thác cơ chế **Self-Attention** của Transformer thông qua chiến lược xử lý **Dual-Path**.

---

## 🏗️ Kiến trúc

```
Tín hiệu hỗn hợp x(t)
        │
        ▼
┌─────────────┐
│   Encoder   │  ← 1D Conv học được (thay thế STFT)
└─────────────┘
        │
        ▼
┌──────────────────────────────┐
│       SepFormer Block        │
│  ┌─────────────────────────┐ │
│  │  IntraTransformer       │ │  ← Học phụ thuộc cục bộ
│  └─────────────────────────┘ │
│  ┌─────────────────────────┐ │
│  │  InterTransformer       │ │  ← Học phụ thuộc toàn cục
│  └─────────────────────────┘ │
│         × R lần              │
└──────────────────────────────┘
        │
        ▼
┌─────────────┐
│  Mask Net   │  ← Ước lượng mặt nạ phân tách
└─────────────┘
        │
        ▼
┌─────────────┐
│   Decoder   │  ← 1D ConvTranspose tái tạo tín hiệu
└─────────────┘
        │
        ▼
  ŝ₁(t),  ŝ₂(t)
```

---

## 📊 Kết quả thực nghiệm

| Mô hình | Năm | Dataset | SI-SNRi (dB) | SDRi (dB) |
|---------|-----|---------|:---:|:---:|
| Conv-TasNet | 2019 | WSJ0-2mix | 15.3 | 15.6 |
| DPRNN | 2020 | WSJ0-2mix | 18.8 | 19.0 |
| SepFormer (gốc) | 2021 | WSJ0-2mix | 22.3 | 22.4 |
| **SepFormer (đề tài)** | 2024 | **Libri2Mix** | **21.60** | — |

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

---

## 🚀 Sử dụng

### Huấn luyện

```bash
python train.py
```

Cấu hình mặc định:
- Dataset: Libri2Mix (8kHz)
- Epochs: 10
- Batch size: 1
- Learning rate: 1.5e-4
- Optimizer: Adam

### Đánh giá

```bash
python evaluate.py
```

### Chạy Web Demo

```bash
python app.py
```

---

## 🛠️ Công nghệ sử dụng

| Công nghệ | Mục đích |
|-----------|---------|
| PyTorch | Framework học sâu |
| SpeechBrain | Công cụ xử lý tiếng nói |
| Librosa | Xử lý tín hiệu âm thanh |
| Gradio | Giao diện web demo |
| Kaggle (GPU T4 x2) | Môi trường huấn luyện |

---

## 📚 Tài liệu tham khảo

1. Subakan et al. (2021) — [Attention is All You Need in Speech Separation](https://arxiv.org/abs/2010.13154)
2. Luo & Mesgarani (2019) — [Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking](https://arxiv.org/abs/1809.07454)
3. Luo et al. (2020) — [Dual-Path RNN](https://arxiv.org/abs/1910.06379)
4. Vaswani et al. (2017) — [Attention is All You Need](https://arxiv.org/abs/1706.03762)
5. Cosentino et al. (2020) — [LibriMix: An Open-Source Dataset for Generalizable Speech Separation](https://arxiv.org/abs/2005.11262)

---

## 👨‍💻 Tác giả

| | |
|---|---|
| **Sinh viên** | Triệu Quốc Khánh |
| **Giảng viên hướng dẫn** | CN. Vi Anh Quân |
| **Trường** | Đại học Khoa học Tự nhiên — ĐHQGHN |
| **Khoa** | Vật lý — Bộ môn Tin học Vật lý |
| **Năm** | 2024 |
