import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model.sepformer import SepFormer
from utils.dataset import Libri2MixDataset
from utils.metrics import pit_si_snr

# ===== Cấu hình =====
DATA_DIR    = "librimix_data"
EPOCHS      = 10
BATCH_SIZE  = 1
LR          = 1.5e-4
SAMPLE_RATE = 8000
MAX_LEN     = 32000
SAVE_DIR    = "checkpoints"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# ===== Dataset =====
train_dataset = Libri2MixDataset(DATA_DIR, split="train",
                                  sample_rate=SAMPLE_RATE, max_len=MAX_LEN)
val_dataset   = Libri2MixDataset(DATA_DIR, split="val",
                                  sample_rate=SAMPLE_RATE, max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# ===== Model =====
model = SepFormer(
    num_filters=256,
    filter_length=16,
    stride=8,
    d_model=256,
    nhead=8,
    d_ff=1024,
    chunk_size=250,
    num_layers=2,
    num_spks=2
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"Training on: {DEVICE}")

# ===== Training Loop =====
for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0
    for batch_idx, (mix, sources) in enumerate(train_loader):
        mix     = mix.to(DEVICE)
        sources = sources.to(DEVICE)

        optimizer.zero_grad()
        estimates = model(mix)
        loss = pit_si_snr(estimates, sources)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for mix, sources in val_loader:
            mix     = mix.to(DEVICE)
            sources = sources.to(DEVICE)
            estimates = model(mix)
            loss = pit_si_snr(estimates, sources)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)

    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    # Lưu checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, f"{SAVE_DIR}/checkpoint_epoch_{epoch}.pt")

print("Training complete!")