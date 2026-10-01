import random
import shutil
from pathlib import Path

random.seed(42)

raw = Path("data/raw")
out = Path("data/split")

classes = {
    "marked": raw / "marked" / "marked",
    "unmarked": raw / "unmarked" / "unmarked",
}

for cls, folder in classes.items():
    files = sorted(folder.glob("*.png"))
    random.shuffle(files)
    n = len(files)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }
    for split, items in splits.items():
        dest = out / split / cls
        dest.mkdir(parents=True, exist_ok=True)
        for f in items:
            shutil.copy2(f, dest / f.name)
        print(cls, split, len(items))