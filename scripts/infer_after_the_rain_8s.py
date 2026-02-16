import os
import sys
import datetime
import csv

import cv2
import numpy as np
import torch

# --- repo root on sys.path ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.va_resnet_gru import VAResNetGRU


def log(msg: str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def extract_window_clip(cap, start_frame, end_frame, num_frames=32, img_size=112):
    """
    Sample num_frames evenly between start_frame and end_frame (exclusive),
    return torch.FloatTensor of shape (T, 3, H, W), ImageNet-normalized.
    """
    total = end_frame - start_frame
    if total <= 0:
        return None

    idxs = np.linspace(0, total - 1, num_frames).astype(int) + start_frame

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            return None

        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0  # H, W, C
        frame = np.transpose(frame, (2, 0, 1))    # C, H, W
        frames.append(frame)

    frames = np.stack(frames, axis=0)  # (T, C, H, W)
    clip = torch.from_numpy(frames)    # float32

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    clip = (clip - mean) / std

    return clip


def main():
    # Hard-coded paths for After_The_Rain
    movie_path = os.path.join(
        ROOT_DIR,
        "test",
        "after_the_rain",
        "After_The_Rain.mp4",
    )
    out_csv = os.path.join(
        ROOT_DIR,
        "test",
        "after_the_rain",
        "After_The_Rain_discrete_8s.csv",
    )

    if not os.path.isfile(movie_path):
        raise FileNotFoundError(f"Movie not found: {movie_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    log(f"Movie: {movie_path}")

    # ---- load DISCRETE model checkpoint ----
    model = VAResNetGRU(hidden_dim=256, pretrained=False).to(device)
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "va_resnet_gru_best.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    log(f"Loaded discrete weights from {ckpt_path}")

    # ---- open video ----
    cap = cv2.VideoCapture(movie_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {movie_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / max(fps, 1e-6)

    log(f"FPS: {fps:.3f}, total frames: {total_frames}, duration: {duration:.2f}s")

    window_sec = 8.0
    frames_per_window = int(round(window_sec * fps))
    if frames_per_window <= 0:
        raise ValueError("frames_per_window <= 0, check FPS/window_sec")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    f = open(out_csv, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["window_idx", "start_sec", "end_sec", "pred_valence", "pred_arousal"])

    window_idx = 0
    with torch.no_grad():
        while True:
            start_frame = window_idx * frames_per_window
            end_frame = start_frame + frames_per_window

            if end_frame > total_frames:
                break

            start_sec = start_frame / max(fps, 1e-6)
            end_sec = end_frame / max(fps, 1e-6)

            clip = extract_window_clip(
                cap,
                start_frame=start_frame,
                end_frame=end_frame,
                num_frames=32,
                img_size=112,
            )
            if clip is None:
                break

            clip = clip.unsqueeze(0).to(device)  # (1, T, C, H, W)
            pred = model(clip)                   # (1, 2)
            v, a = pred.squeeze(0).tolist()

            writer.writerow([window_idx, start_sec, end_sec, float(v), float(a)])

            if window_idx % 25 == 0:
                log(
                    f"Window {window_idx}: [{start_sec:.1f}, {end_sec:.1f}] "
                    f"-> V={v:.4f}, A={a:.4f}"
                )

            window_idx += 1

    f.close()
    cap.release()
    log(f"Done. Saved {window_idx} windows to {out_csv}")


if __name__ == "__main__":
    main()
