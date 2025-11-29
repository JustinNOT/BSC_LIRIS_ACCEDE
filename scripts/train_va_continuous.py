import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.va_resnet_gru import VAResNetGRU
from src.datasets.liris_continuous_dataset import LirisContinuousWindowDataset


def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def main():
    # -----------------------------
    # Basic config (lighter than discrete training)
    # -----------------------------
    labels_csv  = "data/liris_continuous/labels_windows.csv"
    movies_root = "data/liris_continuous/movies"

    batch_size   = 2
    num_frames   = 16   # fewer than discrete (32)
    img_size     = 96   # smaller than 112
    num_epochs   = 4    # fewer than 10
    lr           = 1e-4
    weight_decay = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # -----------------------------
    # Datasets & loaders
    # -----------------------------
    train_ds = LirisContinuousWindowDataset(
        labels_csv=labels_csv,
        movies_root=movies_root,
        split="train",
        num_frames=num_frames,
        img_size=img_size,
    )

    val_ds = LirisContinuousWindowDataset(
        labels_csv=labels_csv,
        movies_root=movies_root,
        split="val",
        num_frames=num_frames,
        img_size=img_size,
    )

    log(f"Train windows: {len(train_ds)}")
    log(f"Val windows:   {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   # 0 on Windows to avoid spawn issues
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # -----------------------------
    # Model: start from discrete checkpoint & freeze backbone
    # -----------------------------
    model = VAResNetGRU(hidden_dim=256, pretrained=False).to(device)

    init_ckpt = "checkpoints/va_resnet_gru_best.pth"
    if os.path.isfile(init_ckpt):
        state = torch.load(init_ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        log(f"Loaded initial weights from {init_ckpt}")
    else:
        log(f"WARNING: discrete checkpoint {init_ckpt} not found, training from scratch")

    # Freeze ResNet18 backbone to reduce compute; train GRU + head only
    for p in model.backbone.parameters():
        p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    log(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

    best_val_mse = float("inf")
    best_ckpt_path = "checkpoints/va_resnet_gru_continuous_best.pth"
    os.makedirs("checkpoints", exist_ok=True)

    # -----------------------------
    # Training loop
    # -----------------------------
    log(f"Starting training for {num_epochs} epochs")

    for epoch in range(1, num_epochs + 1):
        log(f"Epoch {epoch} started")
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for clips, targets in tqdm(train_loader, desc=f"Train {epoch}", ncols=80):
            clips = clips.to(device)       # (B, T, 3, H, W)
            targets = targets.to(device)   # (B, 2)

            preds = model(clips)           # (B, 2)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = clips.size(0)
            train_loss_sum += loss.item() * bs
            train_count += bs

        train_mse = train_loss_sum / train_count

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for clips, targets in tqdm(val_loader, desc=f"Val {epoch}", ncols=80):
                clips = clips.to(device)
                targets = targets.to(device)

                preds = model(clips)
                loss = criterion(preds, targets)

                bs = clips.size(0)
                val_loss_sum += loss.item() * bs
                val_count += bs

        val_mse = val_loss_sum / val_count
        log(f"Epoch {epoch} | Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), best_ckpt_path)
            log(f"New best model saved to {best_ckpt_path} (Val MSE: {best_val_mse:.4f})")

    log("Training complete.")


if __name__ == "__main__":
    main()
