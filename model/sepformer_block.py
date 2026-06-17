import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional Encoding dạng sin/cos."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerBlock(nn.Module):
    """
    Một Transformer block gồm:
    - Multi-Head Self-Attention
    - Feed-Forward Network
    - LayerNorm + Residual Connection
    """
    def __init__(self, d_model=256, nhead=8, d_ff=1024, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Attention + Residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN + Residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class SepFormerBlock(nn.Module):
    """
    SepFormer Block với chiến lược Dual-Path Processing.
    Gồm IntraTransformer và InterTransformer xen kẽ nhau.
    """
    def __init__(self, d_model=256, nhead=8, d_ff=1024,
                 chunk_size=250, num_layers=2, dropout=0.1):
        super(SepFormerBlock, self).__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model

        # IntraTransformer — học phụ thuộc cục bộ trong chunk
        self.intra_transformers = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.intra_pos_enc = PositionalEncoding(d_model)

        # InterTransformer — học phụ thuộc toàn cục giữa các chunk
        self.inter_transformers = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.inter_pos_enc = PositionalEncoding(d_model)

        self.norm = nn.LayerNorm(d_model)

    def segment(self, x):
        """
        Chia tín hiệu thành các chunk.
        Args:
            x: (batch, N, T')
        Returns:
            chunks: (batch, N, chunk_size, num_chunks)
        """
        batch, N, T = x.shape
        K = self.chunk_size
        # Padding nếu cần
        pad_len = (K - T % K) % K
        if pad_len > 0:
            x = nn.functional.pad(x, (0, pad_len))
        T_padded = x.shape[-1]
        num_chunks = T_padded // K
        # Reshape thành chunks
        x = x.view(batch, N, num_chunks, K)
        return x, T, pad_len

    def overlap_add(self, x, orig_len, pad_len):
        """
        Tái cấu trúc từ chunks về dạng chuỗi.
        """
        batch, N, num_chunks, K = x.shape
        x = x.reshape(batch, N, num_chunks * K)
        if pad_len > 0:
            x = x[:, :, :-pad_len]
        return x

    def forward(self, x):
        """
        Args:
            x: (batch, N, T')
        Returns:
            x: (batch, N, T')
        """
        batch, N, T = x.shape
        K = self.chunk_size

        # Segment thành chunks
        x_chunked, orig_len, pad_len = self.segment(x)
        # x_chunked: (batch, N, num_chunks, K)
        num_chunks = x_chunked.shape[2]

        # ===== IntraTransformer =====
        # Xử lý từng chunk độc lập
        x_intra = x_chunked.permute(0, 2, 3, 1)
        # (batch, num_chunks, K, N)
        x_intra = x_intra.reshape(batch * num_chunks, K, N)
        x_intra = self.intra_pos_enc(x_intra)
        for layer in self.intra_transformers:
            x_intra = layer(x_intra)
        x_intra = x_intra.reshape(batch, num_chunks, K, N)
        x_intra = x_intra.permute(0, 3, 1, 2)
        # (batch, N, num_chunks, K)

        # ===== InterTransformer =====
        # Xử lý cùng vị trí giữa các chunk
        x_inter = x_intra.permute(0, 3, 2, 1)
        # (batch, K, num_chunks, N)
        x_inter = x_inter.reshape(batch * K, num_chunks, N)
        x_inter = self.inter_pos_enc(x_inter)
        for layer in self.inter_transformers:
            x_inter = layer(x_inter)
        x_inter = x_inter.reshape(batch, K, num_chunks, N)
        x_inter = x_inter.permute(0, 3, 2, 1)
        # (batch, N, num_chunks, K)

        # Tái cấu trúc
        x_out = self.overlap_add(x_inter, orig_len, pad_len)
        return self.norm(x_out.permute(0, 2, 1)).permute(0, 2, 1)