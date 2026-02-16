import os
import sys
import csv
import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- repo root on sys.path ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.datasets.liris_dataset import LirisVASequenceDataset
from src.models.va_resnet_gru import VAResNetGRU


def log(msg: str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def run_split(split: str, device: torch.device):
    labels_csv = "data/liris_discrete/labels.csv"
    video_root = "data/liris_discrete/raw_videos/data"

    log(f"Loading {split} split from {labels_csv}")
    ds = LirisVASequenceDataset(labels_csv, video_root, split=split)
    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    log(f"{split} set size: {len(ds)} clips")

    # --- model: load EXISTING discrete checkpoint (no training) ---
    model = VAResNetGRU(hidden_dim=256, pretrained=False).to(device)
    ckpt_path = "checkpoints/va_resnet_gru_best.pth"

    if not os.path.isfile(ckpt_path):
        log(f"ERROR: checkpoint {ckpt_path} not found. Train discrete model first.")
        return

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    log(f"Loaded weights from {ckpt_path}")

    preds_rows = []
    global_idx = 0

    with torch.no_grad():
        for vids, _ in tqdm(loader, desc=f"Infer {split}", ncols=80):
            vids = vids.to(device)          # (B, T, 3, H, W)
            outputs = model(vids)           # (B, 2)
            outputs = outputs.cpu()

            for out in outputs:
                v, a = out.tolist()
                preds_rows.append((global_idx, float(v), float(a)))
                global_idx += 1

    out_dir = "data/liris_discrete"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"preds_${split}_8s.csv".replace("${split}", split))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "pred_valence", "pred_arousal"])
        writer.writerows(preds_rows)

    log(f"Saved {len(preds_rows)} predictions to {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # Only inference on existing splits, no training.
    for split in ["train", "val"]:
        run_split(split, device)


if __name__ == "__main__":
    main()
