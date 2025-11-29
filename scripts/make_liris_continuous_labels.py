import os
import csv
import random
import numpy as np

# Aggregated (movie-level) annotation files:
ANNOT_ROOT = "data/liris_continuous/annotations/continuous-annotations"

# Root where ALL the continuous movie files live (any subfolders):
MOVIE_ROOT = "data/liris_continuous/movies"

OUT_CSV    = "data/liris_continuous/labels_windows.csv"

# window config (can tweak)
WINDOW_SEC = 8.0
HOP_SEC    = 2.0


def norm_key(s: str) -> str:
    """Lowercase and keep only alphanumeric; used to match movie IDs to filenames."""
    return "".join(c for c in s.lower() if c.isalnum())


def find_annotation_pairs():
    """
    Find (movie_id, valence_file, arousal_file) pairs.

    We use aggregated movie-level files like:
        After_The_Rain_Arousal.txt
        After_The_Rain_Valence.txt
    """
    all_entries = os.listdir(ANNOT_ROOT)
    files = [
        f for f in all_entries
        if os.path.isfile(os.path.join(ANNOT_ROOT, f)) and f.lower().endswith(".txt")
    ]

    val_files = [f for f in files if "valence" in f.lower()]
    aru_files = [f for f in files if "arousal" in f.lower()]

    def strip_suffix(name: str, key: str) -> str:
        # remove "_Valence" / "_Arousal" etc and extension, but keep the base movie name
        base = name
        base = base.replace(".txt", "").replace(".csv", "")
        idx = base.lower().rfind("_" + key.lower())
        if idx != -1:
            base = base[:idx]
        return base  # e.g. "After_The_Rain"

    pairs = []
    for vf in val_files:
        movie_id = strip_suffix(vf, "valence")
        match = None
        for af in aru_files:
            if strip_suffix(af, "arousal") == movie_id:
                match = af
                break
        if match:
            pairs.append(
                (movie_id,
                 os.path.join(ANNOT_ROOT, vf),
                 os.path.join(ANNOT_ROOT, match))
            )

    print(f"Found {len(pairs)} valence/arousal annotation pairs")
    return pairs


def load_time_series(path):
    """
    Load (times, values) from an annotation file.

    Tries to be robust:
      - If 1 column -> assume 1 Hz, t=0,1,2,...
      - If 2+ columns (comma/semicolon/tab) -> first=t, second=value.
    """
    times = []
    vals = []

    with open(path, "r") as f:
        first = f.readline()
        maybe_header = first.strip()

        def parse_line(line):
            line = line.strip()
            if not line:
                return None
            line = line.replace(";", ",").replace("\t", ",")
            parts = [p for p in line.split(",") if p != ""]
            if len(parts) == 1:
                return ("value_only", float(parts[0]))
            else:
                return ("time_val", (float(parts[0]), float(parts[1])))

        first_parsed = None
        try:
            first_parsed = parse_line(maybe_header)
        except Exception:
            first_parsed = None

        if first_parsed is None or first_parsed[0] == "time_val":
            # treat first line as data (t, v)
            if first_parsed is not None:
                t, v = first_parsed[1]
                times.append(t)
                vals.append(v)
        else:
            # header or value-only first line; skip as header
            pass

        idx = 0 if len(times) == 0 else int(times[-1]) + 1
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            kind, val = parsed
            if kind == "time_val":
                t, v = val
                times.append(t)
                vals.append(v)
            else:  # value_only
                v = val
                t = float(idx)
                idx += 1
                times.append(t)
                vals.append(v)

    if not times:
        raise RuntimeError(f"No data parsed from {path}")

    return np.array(times, dtype=float), np.array(vals, dtype=float)


def collect_movie_files():
    """Walk MOVIE_ROOT and collect all .mp4 files with their dirs."""
    movie_files = []
    for root, _, files in os.walk(MOVIE_ROOT):
        for f in files:
            if f.lower().endswith(".mp4"):
                movie_files.append((root, f))
    if not movie_files:
        print(f"WARNING: no .mp4 files found under {MOVIE_ROOT}")
    else:
        print(f"Found {len(movie_files)} movie files under {MOVIE_ROOT}")
    return movie_files


def find_movie_for_id(movie_id, movie_files):
    """
    Try to find the best-matching movie filename for a given annotation movie_id.
    """
    key = norm_key(movie_id)
    if not key:
        return None

    candidates = []
    for root, fname in movie_files:
        fk = norm_key(fname)
        if key in fk or fk in key:
            candidates.append((root, fname))

    if not candidates:
        return None

    # If multiple, just pick the first (they should be unique in this dataset)
    return candidates[0]


def build_windows():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    pairs = find_annotation_pairs()
    movie_files = collect_movie_files()
    rows = []
    movie_ids = []

    for movie_id, val_path, aru_path in pairs:
        print(f"\nProcessing {movie_id}")
        tv, vv = load_time_series(val_path)
        ta, av = load_time_series(aru_path)

        # ---- Relaxed alignment: truncate to common length ----
        n = min(len(tv), len(ta))
        if n == 0:
            print(f"  WARNING: empty time series for {movie_id}, skipping")
            continue
        tv = tv[:n]
        vv = vv[:n]
        ta = ta[:n]
        av = av[:n]

        # Use valence times as reference
        times = tv
        duration = float(times[-1]) if len(times) > 0 else 0.0

        movie_match = find_movie_for_id(movie_id, movie_files)
        if movie_match is None:
            print(f"  WARNING: no movie file found for id '{movie_id}', skipping")
            continue
        movie_root, filename = movie_match
        rel_path = os.path.join(os.path.relpath(movie_root, MOVIE_ROOT), filename)
        rel_path = rel_path.lstrip("./\\")
        print(f"  Using movie file: {rel_path} (duration approx {duration:.1f}s from annotations)")

        t = 0.0
        while t + WINDOW_SEC <= duration:
            t_start = t
            t_end = t + WINDOW_SEC

            mask = (times >= t_start) & (times < t_end)
            if not np.any(mask):
                t += HOP_SEC
                continue

            v_mean = float(vv[mask].mean())
            a_mean = float(av[mask].mean())

            rows.append({
                "movie_id": movie_id,
                "filename": rel_path,
                "t_start": t_start,
                "t_end": t_end,
                "valence": v_mean,
                "arousal": a_mean,
            })
            t += HOP_SEC

        movie_ids.append(movie_id)

    # split by movie
    random.seed(42)
    uniq_movies = sorted(set(movie_ids))
    random.shuffle(uniq_movies)
    n = len(uniq_movies)
    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)

    train_ids = set(uniq_movies[:n_train])
    val_ids   = set(uniq_movies[n_train:n_train+n_val])
    test_ids  = set(uniq_movies[n_train+n_val:])

    for r in rows:
        mid = r["movie_id"]
        if mid in train_ids:
            r["split"] = "train"
        elif mid in val_ids:
            r["split"] = "val"
        else:
            r["split"] = "test"

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["movie_id", "filename", "t_start", "t_end",
                        "valence", "arousal", "split"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nSaved {len(rows)} windows to {OUT_CSV}")

if __name__ == "__main__":
    build_windows()
