import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Learned encoder thay thế STFT.
    Dùng 1D Convolution để học biểu diễn tín hiệu âm thanh.
    """
    def __init__(self, num_filters=256, filter_length=16, stride=8):
        super(Encoder, self).__init__()
        self.num_filters = num_filters
        self.filter_length = filter_length
        self.stride = stride

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=filter_length,
            stride=stride,
            padding=0,
            bias=False
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (batch, 1, T) — tín hiệu âm thanh đầu vào
        Returns:
            E: (batch, N, T') — biểu diễn trong không gian ẩn
        """
        return self.relu(self.conv(x))