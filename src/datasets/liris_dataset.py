import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class LirisVASequenceDataset(Dataset):
    def __init__(self, labels_csv, video_root, split="train",
                 num_frames=32, img_size=112):
        """
        labels_csv: path to data/liris_discrete/labels.csv
        video_root: folder containing all ACCEDExxxxx.mp4
        split: "train", "val", or "test"
        """
        self.df = pd.read_csv(labels_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.video_root = video_root
        self.num_frames = num_frames
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def _load_video_frames(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Video has 0 frames: {path}")

        # sample up to num_frames evenly across the clip
        n = min(self.num_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, n, dtype=int)
        frames = []
        idx_set = set(indices.tolist())
        cur_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if cur_idx in idx_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            cur_idx += 1

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"No frames read from: {path}")

        frames = np.stack(frames, axis=0)  # (T, H, W, C)
        frames = np.transpose(frames, (0, 3, 1, 2))  # (T, C, H, W)
        return torch.from_numpy(frames)  # float32

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(self.video_root, row["filename"])
        video = self._load_video_frames(video_path)

        target = torch.tensor(
            [row["valence"], row["arousal"]],
            dtype=torch.float32
        )

        return video, target
