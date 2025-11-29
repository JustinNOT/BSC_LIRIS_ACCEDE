import sys
import os
from pathlib import Path
import csv

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

CKPT_PATH = ROOT / "checkpoints" / "va_resnet_gru_best.pth"  # DISCRETE model
VIDEO_PATH = ROOT / "test" / "after_the_rain" / "After_The_Rain.mp4"
OUT_CSV = ROOT / "test" / "after_the_rain" / "After_The_Rain_discrete_1s.csv"


def load_checkpoint(device):
    print(f"[INFO] Loading discrete checkpoint: {CKPT_PATH}")
    obj = torch.load(CKPT_PATH, map_location=device)
    print(f"[INFO] Checkpoint type: {type(obj)}")
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"[INFO] First 10 keys: {keys[:10]}")
    return obj


def auto_import_va_resnet_gru():
    for path in ROOT.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "class VAResNetGRU" in text:
            module_dir = str(path.parent)
            module_name = path.stem
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            try:
                module = __import__(module_name)
                cls = getattr(module, "VAResNetGRU", None)
                if cls is not None:
                    print(f"[INFO] Found VAResNetGRU in {path}")
                    return cls
            except Exception as e:
                print(f"[WARN] Failed importing VAResNetGRU from {path}: {e}")
                continue
    print("[ERROR] Could not find VAResNetGRU class in repo.")
    return None


def build_model_from_checkpoint(obj, device):
    if hasattr(obj, "forward"):
        model = obj
        model.to(device)
        model.eval()
        return model

    if not isinstance(obj, dict):
        print("[ERROR] Unknown checkpoint format; expected state_dict.")
        return None

    state_dict = obj.get("model", obj)

    VAResNetGRU = auto_import_va_resnet_gru()
    if VAResNetGRU is None:
        return None

    model = VAResNetGRU()  # add ctor args if needed
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("[INFO] Discrete model built and weights loaded.")
    return model


def grab_window_around_time(cap, t_sec, fps, num_frames=16, resize=224):
    if fps <= 0:
        return None

    center_frame = int(t_sec * fps)
    start = center_frame - num_frames // 2
    if start < 0:
        start = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    while len(frames) < num_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resize, resize))
        frames.append(frame)

    if not frames:
        return None

    return np.stack(frames, axis=0)


def main():
    print("[INFO] dump_discrete_1s_after_the_rain.py starting")

    if not VIDEO_PATH.exists():
        print(f"[ERROR] Video not found at {VIDEO_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    obj = load_checkpoint(device)
    model = build_model_from_checkpoint(obj, device)
    if model is None:
        print("[ERROR] Could not build model; aborting.")
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0.0

    print(f"[INFO] Video: {VIDEO_PATH.name}")
    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Frames: {frame_count}")
    print(f"[INFO] Duration: {duration:.2f}s")

    times_1s = np.arange(0.0, max(0.0, duration), 1.0)
    print(f"[INFO] Sampling {len(times_1s)} timestamps at 1s intervals.")

    rows = []
    n_features = None

    with torch.no_grad():
        for idx, t in enumerate(times_1s):
            if idx % 10 == 0:
                print(f"[INFO] t={t:.2f}s ({idx+1}/{len(times_1s)})")

            clip = grab_window_around_time(cap, float(t), fps, num_frames=16, resize=224)
            if clip is None:
                continue

            x = torch.from_numpy(clip).float() / 255.0
            x = x.permute(0, 3, 1, 2).unsqueeze(0).to(device)

            out = model(x)
            if not isinstance(out, torch.Tensor):
                continue

            vec = out.detach().cpu().numpy().reshape(-1)
            if n_features is None:
                n_features = vec.shape[0]

            pred_idx = int(vec.argmax())
            row = [float(t), pred_idx] + [float(v) for v in vec]
            rows.append(row)

    cap.release()

    if not rows:
        print("[ERROR] No rows produced; nothing to save.")
        return

    header = ["time_sec", "pred_class"] + [f"logit_{i}" for i in range(n_features)]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[INFO] Wrote {len(rows)} rows to {OUT_CSV}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
