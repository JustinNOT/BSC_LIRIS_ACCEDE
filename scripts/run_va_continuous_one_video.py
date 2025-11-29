import sys
from pathlib import Path
import argparse
import csv

import numpy as np
import torch

# ------------------------------------------------------------------
# Make repo root importable
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------
# MODEL IMPORT
# ------------------------------------------------------------------
# EDIT THESE TRY BLOCKS TO MATCH HOW train_va_continuous.py IMPORTS
# THE MODEL (keep only the working one if you know it).
try:
    from models.va_resnet_gru import VAResNetGRU          # e.g. models/va_resnet_gru.py
except ImportError:
    try:
        from va_resnet_gru import VAResNetGRU             # e.g. va_resnet_gru.py in root
    except ImportError as e:
        raise ImportError(
            "Cannot import VAResNetGRU. "
            "Open scripts/train_va_continuous.py and copy the exact model import "
            "into this file, replacing the try/except block."
        ) from e

# ------------------------------------------------------------------
# DATASET IMPORT
# ------------------------------------------------------------------
# You need a dataset that, given a single video path, returns
# (frames, time_sec) batches with the SAME preprocessing as training.
#
# If you already have such a class (check your data/ or datasets/ folder),
# plug it into the try block below. Otherwise you must create one.
try:
    from data.va_continuous_dataset import SingleVideoContinuousDataset
except ImportError:
    try:
        from datasets.va_continuous_dataset import SingleVideoContinuousDataset
    except ImportError as e:
        raise ImportError(
            "Cannot import SingleVideoContinuousDataset. "
            "Use the same dataset module used in train_va_continuous.py, "
            "or implement a single-video dataset and import it here."
        ) from e


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = VAResNetGRU()  # add args if your training script does

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="va_output.csv")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    video_path = Path(args.video)
    out_csv = Path(args.out_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(ckpt_path, device)

    dataset = SingleVideoContinuousDataset(video_path)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    all_times = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            # Adjust this depending on what your dataset returns:
            #   frames, times = batch
            # or:
            #   frames, times = batch["frames"], batch["time"]
            frames, times = batch

            frames = frames.to(device)
            out = model(frames)  # [B, 2] -> (valence, arousal)

            all_preds.append(out.cpu().numpy())
            all_times.append(times.numpy())

    if not all_preds:
        return

    preds = np.concatenate(all_preds, axis=0)
    times = np.concatenate(all_times, axis=0)

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_sec", "valence", "arousal"])
        for t, (v, a) in zip(times, preds):
            writer.writerow([float(t), float(v), float(a)])


if __name__ == "__main__":
    main()
