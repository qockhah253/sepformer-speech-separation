import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from model.sepformer_block import SepFormerBlock

class SepFormer(nn.Module):
    """
    Kiến trúc SepFormer hoàn chỉnh:
    Encoder -> SepFormerBlock -> Mask -> Decoder
    """
    def __init__(
        self,
        num_filters=256,
        filter_length=16,
        stride=8,
        d_model=256,
        nhead=8,
        d_ff=1024,
        chunk_size=250,
        num_layers=2,
        num_spks=2,
        dropout=0.1
    ):
        super(SepFormer, self).__init__()
        self.num_spks = num_spks

        # Encoder
        self.encoder = Encoder(num_filters, filter_length, stride)

        # Layer chuẩn hóa đầu vào
        self.ln = nn.LayerNorm(num_filters)
        self.proj = nn.Linear(num_filters, d_model)

        # SepFormer Block (Dual-Path Transformer)
        self.sepformer = SepFormerBlock(
            d_model=d_model,
            nhead=nhead,
            d_ff=d_ff,
            chunk_size=chunk_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # Mask estimation
        self.mask_net = nn.Sequential(
            nn.PReLU(),
            nn.Linear(d_model, num_filters * num_spks),
            nn.ReLU()
        )

        # Decoder
        self.decoder = Decoder(num_filters, filter_length, stride)

    def forward(self, x):
        """
        Args:
            x: (batch, T) — tín hiệu âm thanh đầu vào
        Returns:
            sources: (batch, num_spks, T) — các nguồn âm đã tách
        """
        # Encoder
        x = x.unsqueeze(1)  # (batch, 1, T)
        enc = self.encoder(x)  # (batch, N, T')

        # Chuẩn hóa và chiếu
        enc_norm = self.ln(enc.permute(0, 2, 1))  # (batch, T', N)
        enc_proj = self.proj(enc_norm).permute(0, 2, 1)  # (batch, d_model, T')

        # SepFormer Block
        sep_out = self.sepformer(enc_proj)  # (batch, d_model, T')

        # Ước lượng mặt nạ
        masks = self.mask_net(sep_out.permute(0, 2, 1))
        # (batch, T', N * num_spks)
        masks = masks.permute(0, 2, 1)
        # (batch, N * num_spks, T')
        masks = masks.view(
            masks.shape[0], self.num_spks, -1, masks.shape[-1]
        )
        # (batch, num_spks, N, T')

        # Áp dụng mặt nạ và giải mã
        sources = []
        for i in range(self.num_spks):
            masked = masks[:, i, :, :] * enc  # (batch, N, T')
            source = self.decoder(masked)      # (batch, 1, T)
            sources.append(source.squeeze(1))  # (batch, T)

        return torch.stack(sources, dim=1)  # (batch, num_spks, T)