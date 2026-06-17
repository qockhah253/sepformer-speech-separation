import gradio as gr
import torch
import tempfile
import os
import numpy as np
import librosa
import soundfile as sf
from speechbrain.inference import SepformerSeparation as separator

# Load model
model = separator.from_hparams(
    source="speechbrain/sepformer-libri2mix",
    savedir="pretrained_models/sepformer-libri2mix"
)

def separate_audio(audio_file):
    if audio_file is None:
        return None, None
    
    try:
        # Load và resample về 8kHz bằng librosa
        waveform, _ = librosa.load(audio_file, sr=8000, mono=True)
        
        # Lưu file tạm
        tmp_path = "/tmp/input.wav"
        sf.write(tmp_path, waveform, 8000)
        
        # Tách nguồn âm
        est_sources = model.separate_file(path=tmp_path)
        
        # Lấy kết quả
        source1 = est_sources[:, :, 0].detach().cpu().numpy().squeeze()
        source2 = est_sources[:, :, 1].detach().cpu().numpy().squeeze()
        
        # Lưu kết quả bằng soundfile
        source1_path = "/tmp/source1.wav"
        source2_path = "/tmp/source2.wav"
        sf.write(source1_path, source1, 8000)
        sf.write(source2_path, source2, 8000)
        
        return source1_path, source2_path
    
    except Exception as e:
        print(f"Error: {e}")
        raise gr.Error(f"Lỗi: {str(e)}")

demo = gr.Interface(
    fn=separate_audio,
    inputs=gr.Audio(
        label="Upload file âm thanh hỗn hợp (2 người nói)",
        type="filepath"
    ),
    outputs=[
        gr.Audio(label="Nguồn âm thứ nhất"),
        gr.Audio(label="Nguồn âm thứ hai")
    ],
    title="SepFormer — Phân tách nguồn âm thanh đơn kênh",
    description="""
Demo ứng dụng kiến trúc SepFormer cho bài toán phân tách nguồn âm thanh đơn kênh.

**Hướng dẫn:**
1. Upload file âm thanh chứa 2 người nói cùng lúc
2. Nhấn Submit
3. Nghe và download 2 nguồn âm đã được tách

**Lưu ý:** Hỗ trợ định dạng WAV, MP3, FLAC.

> ⚠️ **Khuyến nghị:** Nên sử dụng file âm thanh có tần số lấy mẫu **8kHz** và định dạng file có đuôi .wav để đạt chất lượng phân tách tốt nhất.
    """
)

demo.launch()