import os
import pandas as pd

# NOTE: videos are under raw_videos/data/ according to your folder tree
RANKING_PATH = os.path.join(
    "data", "raw", "LIRIS-ACCEDE-annotations",
    "LIRIS-ACCEDE-annotations", "annotations", "ACCEDEranking.txt"
)
VIDEO_DIR = os.path.join("data", "liris_discrete", "raw_videos", "data")
OUT_CSV = os.path.join("data", "liris_discrete", "labels.csv")


def main():
    if not os.path.isfile(RANKING_PATH):
        raise FileNotFoundError(f"Ranking file not found at {RANKING_PATH}")

    print("Reading ranking file:", RANKING_PATH)
    print("Looking for videos under:", os.path.abspath(VIDEO_DIR))

    # The file is tab-separated with header:
    # id, name, valenceRank, arousalRank, valenceValue, arousalValue, ...
    df = pd.read_csv(RANKING_PATH, sep=r"\s+", engine="python")

    # Extract clip_id (ACCEDE00000) and filename (ACCEDE00000.mp4)
    df["filename"] = df["name"]
    df["clip_id"] = df["name"].apply(lambda x: os.path.splitext(x)[0])

    # Use valenceValue and arousalValue as our continuous labels
    df = df.rename(
        columns={
            "valenceValue": "valence",
            "arousalValue": "arousal",
        }
    )

    # Keep only needed columns
    df = df[["clip_id", "filename", "valence", "arousal"]]

    # Filter to clips that we actually have videos for
    def has_video(row):
        return os.path.isfile(os.path.join(VIDEO_DIR, row["filename"]))

    df["has_video"] = df.apply(has_video, axis=1)
    missing = df[~df["has_video"]]
    if not missing.empty:
        print(f"WARNING: {len(missing)} rows without matching video. They will be dropped (showing first 5):")
        print(missing.head())

    df = df[df["has_video"]].drop(columns=["has_video"]).reset_index(drop=True)

    # Shuffle and split 70/15/15
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(df)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    if n == 0:
        raise RuntimeError("No rows left after filtering for existing videos. Check VIDEO_DIR path.")

    df.loc[:n_train - 1, "split"] = "train"
    df.loc[n_train:n_train + n_val - 1, "split"] = "val"
    df.loc[n_train + n_val:, "split"] = "test"

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()
