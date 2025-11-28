import os

ANNOT_ROOT = "data/raw/LIRIS-ACCEDE-annotations"

print("Walking", os.path.abspath(ANNOT_ROOT))
for root, dirs, files in os.walk(ANNOT_ROOT):
    # skip macOS junk
    dirs[:] = [d for d in dirs if not d.startswith("__MACOSX")]
    print("\nDIR:", root)
    for f in files:
        print("   ", f)
