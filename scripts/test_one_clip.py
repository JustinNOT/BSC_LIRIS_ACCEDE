import os
import sys
import random
import torch

# --- Make sure repo root is on sys.path ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.datasets.liris_dataset import LirisVASequenceDataset
from src.models.va_resnet_gru import VAResNetGRU


def main():
    labels_csv = "data/liris_discrete/labels.csv"
    video_root = "data/liris_discrete/raw_videos/data"
    ckpt_path = "checkpoints/va_resnet_gru_best.pth"

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Build test dataset
    test_ds = LirisVASequenceDataset(
        labels_csv,
        video_root,
        split="test",
        num_frames=32,
        img_size=112,
    )
    print("Test set size:", len(test_ds))

    # Pick random test index
    idx = random.randint(0, len(test_ds) - 1)
    print(f"Sampling test index: {idx}")

    video, target = test_ds[idx]        # video: (T, C, H, W), target: (2,)
    video = video.unsqueeze(0).to(device)  # (1, T, C, H, W)
    target = target.to(device)

    # Load model
    model = VAResNetGRU(hidden_dim=256, pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        pred = model(video)[0]          # (2,)

    gt = target.cpu().tolist()
    pr = pred.cpu().tolist()

    print("\nGround truth [valence, arousal]:", gt)
    print("Predicted    [valence, arousal]:", pr)
    print("Diff (pred - gt):",
          [pr[0] - gt[0], pr[1] - gt[1]])


if __name__ == "__main__":
    main()
