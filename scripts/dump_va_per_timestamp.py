import sys
import os
from pathlib import Path
import csv

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CKPT_PATH = ROOT / "checkpoints" / "va_resnet_gru_continuous_best.pth"
ANNOT_ROOT = ROOT / "data" / "liris_continuous" / "annotations"
VIDEO_ROOT = ROOT / "data" / "liris_continuous" / "movies"
OUT_CSV = ROOT / "va_eval_after_the_rain.csv"


def find_video(root):
    exts = {".mp4", ".mkv", ".avi", ".mov"}
    chosen = None
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = Path(dirpath) / f
            if p.suffix.lower() in exts:
                if "after_the_rain" in p.stem.lower():
                    return p
                if chosen is None:
                    chosen = p
    return chosen


def find_valence_arousal_files(video_path):
    stem = video_path.stem.lower()
    val_path = None
    aro_path = None

    for dirpath, _, files in os.walk(ANNOT_ROOT):
        for f in files:
            p = Path(dirpath) / f
            name = p.stem.lower()
            if stem in name or name in stem:
                if "valence" in name and val_path is None:
                    val_path = p
                if "arousal" in name and aro_path is None:
                    aro_path = p
    return val_path, aro_path


def load_series(path):
    print(f"[INFO] Loading series from: {path}")
    arr = np.loadtxt(path, skiprows=1)
    if arr.ndim == 1:
        arr = arr[None, :]

    times = arr[:, 0].astype(float)
    vals = arr[:, 1].astype(float)
    print(f"[INFO] Loaded {len(times)} points from {path.name}")
    return times, vals


def load_checkpoint(device):
    print(f"[INFO] Loading checkpoint: {CKPT_PATH}")
    obj = torch.load(CKPT_PATH, map_location=device)
    print(f"[INFO] Checkpoint type: {type(obj)}")
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
            module = __import__(module_name)
            cls = getattr(module, "VAResNetGRU", None)
            if cls is not None:
                print(f"[INFO] Found VAResNetGRU in {path}")
                return cls
    print("[WARN] Could not find VAResNetGRU definition in repo.")
    return None


def build_model_from_checkpoint(obj, device):
    if hasattr(obj, "forward"):
        model = obj
        model.to(device)
        model.eval()
        return model

    if not isinstance(obj, dict):
        print("[WARN] Unknown checkpoint format; cannot build model.")
        return None

    state_dict = obj.get("model", obj)
    VAResNetGRU = auto_import_va_resnet_gru()
    if VAResNetGRU is None:
        return None

    model = VAResNetGRU()  # add ctor args if needed
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("[INFO] Model built and weights loaded.")
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

    if len(frames) == 0:
        return None

    return np.stack(frames, axis=0)  # [T, H, W, 3]


def main():
    print("[INFO] dump_va_per_timestamp.py starting")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    obj = load_checkpoint(device)
    model = build_model_from_checkpoint(obj, device)
    if model is None:
        print("[ERROR] Model could not be built; aborting.")
        return

    video = find_video(VIDEO_ROOT)
    if video is None:
        print(f"[ERROR] No video found under {VIDEO_ROOT}")
        return

    val_path, aro_path = find_valence_arousal_files(video)
    if val_path is None or aro_path is None:
        print("[ERROR] Could not find both valence and arousal annotation files.")
        print("       valence:", val_path)
        print("       arousal:", aro_path)
        return

    t_val, gt_val = load_series(val_path)
    t_aro, gt_aro = load_series(aro_path)

    if not np.allclose(t_val, t_aro):
        print("[WARN] Time axes differ; using intersection and interpolating.")
        t_common = np.intersect1d(t_val, t_aro)

        def interp(t_src, v_src, t_dst):
            return np.interp(t_dst, t_src, v_src)

        gt_val = interp(t_val, gt_val, t_common)
        gt_aro = interp(t_aro, gt_aro, t_common)
        times = t_common
    else:
        times = t_val

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Using video: {video}")
    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Total annotation points: {len(times)}")

    out_rows = []

    with torch.no_grad():
        for idx, (t, v_gt, a_gt) in enumerate(zip(times, gt_val, gt_aro)):
            clip = grab_window_around_time(cap, t, fps, num_frames=16, resize=224)
            if clip is None:
                continue

            x = torch.from_numpy(clip).float() / 255.0
            x = x.permute(0, 3, 1, 2).unsqueeze(0).to(device)  # [1, T, C, H, W]

            out = model(x)
            if not isinstance(out, torch.Tensor):
                continue

            out_np = out.squeeze(0).cpu().numpy()
            if out_np.ndim == 1 and out_np.shape[0] == 2:
                pv, pa = float(out_np[0]), float(out_np[1])
            else:
                pv, pa = float(out_np[0]), float(out_np[1])

            # print per-timestamp comparison
            print(
                f"[ROW {idx+1}] t={t:.3f}s "
                f"GT_V={v_gt:.3f} GT_A={a_gt:.3f} "
                f"PRED_V={pv:.3f} PRED_A={pa:.3f}"
            )

            out_rows.append([t, v_gt, a_gt, pv, pa])

    cap.release()

    if not out_rows:
        print("[ERROR] No rows produced; aborting.")
        return

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_sec", "gt_valence", "gt_arousal", "pred_valence", "pred_arousal"])
        writer.writerows(out_rows)

    print(f"[INFO] Wrote {len(out_rows)} rows to {OUT_CSV}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
