import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Learned decoder — tái tạo tín hiệu âm thanh từ không gian ẩn.
    Dùng 1D Transposed Convolution (ConvTranspose1d).
    """
    def __init__(self, num_filters=256, filter_length=16, stride=8):
        super(Decoder, self).__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=num_filters,
            out_channels=1,
            kernel_size=filter_length,
            stride=stride,
            padding=0,
            bias=False
        )

    def forward(self, x):
        """
        Args:
            x: (batch, N, T') — biểu diễn đã nhân với mặt nạ
        Returns:
            out: (batch, 1, T) — tín hiệu âm thanh tái tạo
        """
        return self.conv_transpose(x)