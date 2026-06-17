import os
import torch
import torchaudio
from torch.utils.data import Dataset

class Libri2MixDataset(Dataset):
    """Dataset cho Libri2Mix."""
    def __init__(self, data_dir, split="train", sample_rate=8000,
                 max_len=32000):
        self.sample_rate = sample_rate
        self.max_len = max_len

        mix_dir = os.path.join(data_dir, split, "mix_clean")
        s1_dir  = os.path.join(data_dir, split, "s1")
        s2_dir  = os.path.join(data_dir, split, "s2")

        self.mix_files = sorted([
            os.path.join(mix_dir, f) for f in os.listdir(mix_dir)
        ])
        self.s1_files = sorted([
            os.path.join(s1_dir, f) for f in os.listdir(s1_dir)
        ])
        self.s2_files = sorted([
            os.path.join(s2_dir, f) for f in os.listdir(s2_dir)
        ])

    def __len__(self):
        return len(self.mix_files)

    def __getitem__(self, idx):
        mix, sr = torchaudio.load(self.mix_files[idx])
        s1,  _  = torchaudio.load(self.s1_files[idx])
        s2,  _  = torchaudio.load(self.s2_files[idx])

        # Resample nếu cần
        if sr != self.sample_rate:
            resample = torchaudio.transforms.Resample(sr, self.sample_rate)
            mix = resample(mix)
            s1  = resample(s1)
            s2  = resample(s2)

        # Cắt hoặc pad về max_len
        mix = self._fix_length(mix.squeeze(0))
        s1  = self._fix_length(s1.squeeze(0))
        s2  = self._fix_length(s2.squeeze(0))

        sources = torch.stack([s1, s2], dim=0)
        return mix, sources

    def _fix_length(self, x):
        if x.shape[-1] > self.max_len:
            start = torch.randint(0, x.shape[-1] - self.max_len, (1,))
            x = x[start:start + self.max_len]
        elif x.shape[-1] < self.max_len:
            x = torch.nn.functional.pad(x, (0, self.max_len - x.shape[-1]))
        return x