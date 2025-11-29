import sys
import os
from pathlib import Path

import torch
import numpy as np
import cv2


ROOT = Path(__file__).resolve().parents[1]
CKPT_PATH = ROOT / "checkpoints" / "va_resnet_gru_continuous_best.pth"


def find_any_video(root: Path):
    exts = {".mp4", ".mkv", ".avi", ".mov"}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if Path(f).suffix.lower() in exts:
                return Path(dirpath) / f
    return None


def load_checkpoint(device):
    print(f"[INFO] Loading checkpoint: {CKPT_PATH}")
    obj = torch.load(CKPT_PATH, map_location=device)
    print(f"[INFO] Checkpoint type: {type(obj)}")
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"[INFO] Checkpoint keys (first 10): {keys[:10]}")
    return obj


def auto_import_va_resnet_gru():
    """
    Search the repo for a file that defines `class VAResNetGRU` and import it.
    This avoids you having to specify the exact module path.
    """
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
        print("[WARN] VAResNetGRU not imported; cannot run forward pass.")
        return None

    model = VAResNetGRU()  # if your constructor needs args, add them here
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("[INFO] Model built and weights loaded.")
    return model


def grab_short_clip(video_path: Path, num_frames=16, resize=224):
    print(f"[INFO] Using video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    while len(frames) < num_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resize, resize))
        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError("No frames read from video.")

    arr = np.stack(frames, axis=0)  # [T, H, W, 3]
    print(f"[INFO] Grabbed {arr.shape[0]} frames of size {arr.shape[1:3]}")
    return arr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    obj = load_checkpoint(device)
    model = build_model_from_checkpoint(obj, device)

    video_path = find_any_video(ROOT)
    if video_path is None:
        print("[WARN] No video files (.mp4/.mkv/.avi/.mov) found under repo.")
        return

    clip = grab_short_clip(video_path)  # [T, H, W, 3]

    x = torch.from_numpy(clip).float() / 255.0  # [T, H, W, 3]
    x = x.permute(0, 3, 1, 2).unsqueeze(0)      # [1, T, C, H, W]
    x = x.to(device)

    if model is None:
        print("[INFO] Checkpoint loaded and video read, but no model forward run.")
        return

    with torch.no_grad():
        out = model(x)

    print("[INFO] Model output type:", type(out))
    if isinstance(out, torch.Tensor):
        print("[INFO] Model output shape:", tuple(out.shape))
        print("[INFO] First output row:", out[0].detach().cpu().numpy())
    else:
        print("[INFO] Model output:", out)


if __name__ == "__main__":
    main()
