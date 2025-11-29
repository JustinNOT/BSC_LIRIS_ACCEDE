from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
base = root / "data" / "liris_discrete"

print(f"[INFO] Looking for discrete annotations under: {base}")
if not base.exists():
    print("[ERROR] data/liris_discrete not found.")
    raise SystemExit(1)

files = list(base.rglob("*.txt")) + list(base.rglob("*.csv"))
print(f"[INFO] Found {len(files)} candidate files")

vals = []
aros = []

def try_load(path: Path):
    for skip in (0, 1):          # try with / without header row
        for delim in (None, ",", ";", "\t"):
            try:
                arr = np.loadtxt(path, delimiter=delim, skiprows=skip)
                if arr.ndim == 1:
                    if arr.size < 2:
                        continue
                    arr = arr.reshape(1, -1)
                if arr.shape[1] >= 2:
                    return arr
            except Exception:
                continue
    return None

for f in files:
    arr = try_load(f)
    if arr is None:
        continue

    # If first column looks like a time axis (monotonic, large span), use cols 1 & 2
    col0 = arr[:, 0]
    use_cols_12 = False
    if arr.shape[1] >= 3:
        if (col0.max() - col0.min()) > 10:  # crude "time" heuristic
            use_cols_12 = True

    if use_cols_12:
        v = arr[:, 1]
        a = arr[:, 2]
    else:
        v = arr[:, 0]
        a = arr[:, 1]

    vals.extend(v.tolist())
    aros.extend(a.tolist())

if not vals:
    print("[ERROR] No numeric valence/arousal data collected.")
    raise SystemExit(1)

vals = np.array(vals, dtype=float)
aros = np.array(aros, dtype=float)

print("\n[RESULT] Valence bounds:")
print(f"  min = {vals.min():.4f}, max = {vals.max():.4f}, mean = {vals.mean():.4f}")

print("\n[RESULT] Arousal bounds:")
print(f"  min = {aros.min():.4f}, max = {aros.max():.4f}, mean = {aros.mean():.4f}")
