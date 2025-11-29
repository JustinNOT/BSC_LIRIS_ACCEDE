import os

ROOT = "data/liris_continuous/annotations"

for root, dirs, files in os.walk(ROOT):
    rel = os.path.relpath(root, ".")
    print(f"\nDIR: {rel}")
    for f in sorted(files)[:50]:
        print("   ", f)
