import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

def dir_size_mb(path: str) -> float:
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            try:
                total += os.path.getsize(fpath)
            except Exception:
                pass
    return total / (1024 * 1024)

print("=== Model Directory Sizes ===")
for d in ["models/joblib", "tuned_params"]:  # lstm/meta archived in Phase 0
    if os.path.exists(d):
        size = dir_size_mb(d)
        count = sum(len(files) for _, _, files in os.walk(d))
        print(f"  {d:25s}: {size:.1f} MB ({count} files)")

total_model_size = sum(
    dir_size_mb(d) for d in ["models/joblib"]
    if os.path.exists(d)
)
print(f"\nTotal model size:  {total_model_size:.1f} MB")
print(f"Render free RAM:   512 MB")
print(f"Estimated headroom:{512 - total_model_size - 150:.0f} MB (after ~150MB for Python/FastAPI)")
