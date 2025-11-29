import os
import csv
from typing import List, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class LirisContinuousWindowDataset(Dataset):
    """
    Dataset for the LIRIS-ACCEDE continuous subset.

    Each sample corresponds to one row in data/liris_continuous/labels_windows.csv:
        movie_id, filename, t_start, t_end, valence, arousal, split

    We sample `num_frames` frames uniformly between [t_start, t_end].
    """

    def __init__(
        self,
        labels_csv: str,
        movies_root: str,
        split: str,
        num_frames: int = 16,
        img_size: int = 96,
    ):
        super().__init__()
        self.labels_csv = labels_csv
        self.movies_root = movies_root
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size

        self.samples: List[Dict] = []
        with open(labels_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.samples.append(row)

        if not self.samples:
            raise RuntimeError(f"No samples found in {labels_csv} for split='{split}'")

        # basic transform: BGR numpy -> PIL -> resized tensor
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def _load_clip_frames(self, video_path: str, t_start: float, t_end: float):
        """
        Load `num_frames` frames between [t_start, t_end] seconds from the video.

        Returns tensor of shape (num_frames, 3, H, W).
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps <= 0 or frame_count <= 0:
            duration = max(t_end, t_start + 1.0)
        else:
            duration = frame_count / fps

        # Clamp window into video duration
        t_start = max(0.0, t_start)
        t_end = min(duration, t_end)
        if t_end <= t_start:
            t_end = min(duration, t_start + 1.0)

        # Uniform times in [t_start, t_end]
        times = np.linspace(t_start, t_end, self.num_frames, endpoint=False)

        frames = []
        last_frame = None

        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, frame = cap.read()
            if not ok:
                if last_frame is not None:
                    frame = last_frame.copy()
                else:
                    break
            last_frame = frame

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)  # (3, H, W)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"No frames read from {video_path}")

        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())

        clip = torch.stack(frames, dim=0)  # (T, 3, H, W)
        return clip

    def __getitem__(self, idx):
        row = self.samples[idx]
        rel_path = row["filename"]  # e.g. "continuous-movies\\After_The_Rain.mp4"
        video_path = os.path.join(self.movies_root, rel_path)
        video_path = os.path.normpath(video_path)

        t_start = float(row["t_start"])
        t_end = float(row["t_end"])
        val = float(row["valence"])
        aro = float(row["arousal"])

        clip = self._load_clip_frames(video_path, t_start, t_end)
        target = torch.tensor([val, aro], dtype=torch.float32)
        return clip, target
