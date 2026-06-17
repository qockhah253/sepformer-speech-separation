import os
import torch
import torchaudio
from model.sepformer import SepFormer
from utils.metrics import si_snr

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT  = "checkpoints/checkpoint_epoch_10.pt"
DATA_DIR    = "librimix_data"
SAMPLE_RATE = 8000

# Load model
model = SepFormer().to(DEVICE)
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)

mix_dir = os.path.join(DATA_DIR, "val", "mix_clean")
s1_dir  = os.path.join(DATA_DIR, "val", "s1")
s2_dir  = os.path.join(DATA_DIR, "val", "s2")

mix_files   = sorted(os.listdir(mix_dir))[:100]
sisnri_list = []

with torch.no_grad():
    for fname in mix_files:
        mix, sr = torchaudio.load(os.path.join(mix_dir, fname))
        s1,  _  = torchaudio.load(os.path.join(s1_dir,  fname))
        s2,  _  = torchaudio.load(os.path.join(s2_dir,  fname))

        if sr != SAMPLE_RATE:
            mix = resample(mix)
            s1  = resample(s1)
            s2  = resample(s2)

        mix_input = mix.squeeze(0).to(DEVICE)
        estimates = model(mix_input.unsqueeze(0))

        est1 = estimates[0, 0, :].cpu()
        est2 = estimates[0, 1, :].cpu()
        s1   = s1.squeeze(0)
        s2   = s2.squeeze(0)
        mix  = mix.squeeze(0)

        min_len = min(est1.shape[0], s1.shape[0])
        est1 = est1[:min_len]
        est2 = est2[:min_len]
        s1   = s1[:min_len]
        s2   = s2[:min_len]
        mix  = mix[:min_len]

        perm1 = (si_snr(est1.unsqueeze(0), s1.unsqueeze(0)) +
                 si_snr(est2.unsqueeze(0), s2.unsqueeze(0))) / 2
        perm2 = (si_snr(est1.unsqueeze(0), s2.unsqueeze(0)) +
                 si_snr(est2.unsqueeze(0), s1.unsqueeze(0))) / 2
        best  = max(perm1, perm2)

        mix_snr = (si_snr(mix.unsqueeze(0), s1.unsqueeze(0)) +
                   si_snr(mix.unsqueeze(0), s2.unsqueeze(0))) / 2
        sisnri  = best - mix_snr
        sisnri_list.append(sisnri.item())
        print(f"{fname}: SI-SNRi = {sisnri.item():.2f} dB")

print(f"\nSI-SNRi trung binh: {sum(sisnri_list)/len(sisnri_list):.2f} dB")