import os
import sys
import time
import datetime

# --- Make sure repo root is on sys.path ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.liris_dataset import LirisVASequenceDataset
from src.models.va_resnet_gru import VAResNetGRU


def log(msg: str):
    """Simple logger with timestamp."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    mse = nn.MSELoss()
    total_loss = 0.0

    for vids, targets in tqdm(loader, desc=f"Train {epoch}", ncols=80):
        vids = vids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(vids)
        loss = mse(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * vids.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device, epoch):
    model.eval()
    mse = nn.MSELoss()
    total_loss = 0.0

    for vids, targets in tqdm(loader, desc=f"Val {epoch}", ncols=80):
        vids = vids.to(device)
        targets = targets.to(device)

        preds = model(vids)
        loss = mse(preds, targets)
        total_loss += loss.item() * vids.size(0)

    return total_loss / len(loader.dataset)


def main():
    labels_csv = "data/liris_discrete/labels.csv"
    # NOTE: your videos are under raw_videos/data
    video_root = "data/liris_discrete/raw_videos/data"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # Datasets / loaders
    train_ds = LirisVASequenceDataset(labels_csv, video_root, split="train")
    val_ds = LirisVASequenceDataset(labels_csv, video_root, split="val")

    log(f"Train set size: {len(train_ds)} clips")
    log(f"Val set size:   {len(val_ds)} clips")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)

    # Model
    log("Initializing VAResNetGRU model (ResNet18 backbone + bidirectional GRU)...")
    model = VAResNetGRU(hidden_dim=256, pretrained=True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    best_val = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    num_epochs = 10
    log(f"Starting training for {num_epochs} epochs")

    for epoch in range(1, num_epochs + 1):
        log(f"Epoch {epoch} started")
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = eval_epoch(model, val_loader, device, epoch)

        dt = time.time() - t0
        log(f"Epoch {epoch} finished in {dt:.1f}s")
        log(f"Epoch {epoch} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = "checkpoints/va_resnet_gru_best.pth"
            torch.save(model.state_dict(), ckpt_path)
            log(f"New best model saved to {ckpt_path} (Val MSE: {best_val:.4f})")

    log("Training complete.")


if __name__ == "__main__":
    main()
