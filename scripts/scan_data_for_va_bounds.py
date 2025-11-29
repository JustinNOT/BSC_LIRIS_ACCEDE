from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
data_root = root / "data"

print(f"[INFO] Scanning for possible VA annotation files under: {data_root}")
if not data_root.exists():
    print("[ERROR] data/ directory not found.")
    raise SystemExit(1)

# Collect all txt/csv files
files = list(data_root.rglob("*.txt")) + list(data_root.rglob("*.csv"))
print(f"[INFO] Found {len(files)} candidate files")

vals = []
aros = []

def try_load(path: Path):
    """Attempt to load a numeric array from path with different delimiters/header skips."""
    for skip in (0, 1):              # try with/without a single header line
        for delim in (None, ",", ";", "\t"):
            try:
                arr = np.loadtxt(path, delimiter=delim, skiprows=skip)
                if arr.size == 0:
                    continue
                # Ensure 2D
                if arr.ndim == 1:
                    # If it's a long 1D, we can't interpret as VA; skip
                    if arr.size < 2:
                        continue
                    # Not reshaping; most VA files should be 2D already
                    return None
                return arr
            except Exception:
                continue
    return None

for f in files:
    arr = try_load(f)
    if arr is None:
        continue

    # Ignore tiny or 1-column arrays
    if arr.shape[0] < 5 or arr.shape[1] < 2:
        continue

    # Decide which columns are VA:
    # - If >=3 columns and col0 looks like time (monotonic, span > 10), treat cols 1 & 2 as VA.
    # - Else, treat cols 0 & 1 as VA.
    use_cols_12 = False
    if arr.shape[1] >= 3:
        col0 = arr[:, 0]
        # crude "time-like" heuristic
        if np.all(np.diff(col0) >= 0) and (col0.max() - col0.min()) > 10:
            use_cols_12 = True

    if use_cols_12:
        v = arr[:, 1]
        a = arr[:, 2]
    else:
        v = arr[:, 0]
        a = arr[:, 1]

    # Filter out obviously non-VA stuff (e.g., pure indices or huge values)
    # Keep only files where values are in a plausible small range.
    v_min, v_max = float(v.min()), float(v.max())
    a_min, a_max = float(a.min()), float(a.max())

    # Heuristic: VA ratings usually live somewhere in [-10, 10] or [0, 10].
    if any(abs(x) > 10 for x in (v_min, v_max, a_min, a_max)):
        # probably not VA labels, skip
        continue

    print(f"[INFO] Using file: {f}")
    print(f"       Valence min/max: {v_min:.3f}/{v_max:.3f}, Arousal min/max: {a_min:.3f}/{a_max:.3f}")

    vals.extend(v.tolist())
    aros.extend(a.tolist())

if not vals:
    print("\n[ERROR] No plausible VA-like numeric data collected from any file.")
    raise SystemExit(1)

vals = np.array(vals, dtype=float)
aros = np.array(aros, dtype=float)

print("\n[RESULT] APPROXIMATE GLOBAL VALENCE BOUNDS ACROSS DATA/:")
print(f"  min = {vals.min():.4f}, max = {vals.max():.4f}, mean = {vals.mean():.4f}")

print("\n[RESULT] APPROXIMATE GLOBAL AROUSAL BOUNDS ACROSS DATA/:")
print(f"  min = {aros.min():.4f}, max = {aros.max():.4f}, mean = {aros.mean():.4f}")
