import os
import shutil
import random
import argparse

TARGET_CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Septoria_leaf_spot",
    "Tomato___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
]

CAP_PER_CLASS = 1000
SEED = 42


def prepare_splits(source_dir: str, output_dir: str) -> None:
    random.seed(SEED)
    for split in ["train", "val", "test"]:
        for cls in TARGET_CLASSES:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    for cls in TARGET_CLASSES:
        cls_dir = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] Folder not found: {cls_dir}")
            continue

        files = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        files.sort()
        random.shuffle(files)
        sampled = files[:CAP_PER_CLASS]

        n_total = len(sampled)
        n_train = int(n_total * 0.70)
        n_val = round(n_total * 0.15)

        splits = {
            "train": sampled[:n_train],
            "val": sampled[n_train:n_train + n_val],
            "test": sampled[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            for fname in split_files:
                src = os.path.join(cls_dir, fname)
                dst = os.path.join(output_dir, split_name, cls, fname)
                shutil.copy2(src, dst)

        print(
            f"{cls}: total={n_total} -> train={len(splits['train'])}, "
            f"val={len(splits['val'])}, test={len(splits['test'])}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to PlantVillage folder")
    parser.add_argument("--output", default="data/splits", help="Output split directory")
    args = parser.parse_args()
    prepare_splits(args.source, args.output)
