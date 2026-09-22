"""
Extraction de features pour le dataset de fleurs : redimensionnement de
chaque image en 16x16 couleur, aplatie en vecteur de 768 valeurs (16*16*3)
normalisees dans [0, 1], stockees avec le label de classe (index entier).
"""

import csv
import json
from pathlib import Path

from PIL import Image

RAW_DIR = Path(__file__).resolve().parent / "raw"
OUT_DIR = Path(__file__).resolve().parent / "processed"
OUT_DIR.mkdir(exist_ok=True)

TARGET_SIZE = (16, 16)


def list_classes():
    return sorted(d.name for d in RAW_DIR.iterdir() if d.is_dir() and any(d.glob("*.jpg")))


def extract_features(image_path):
    with Image.open(image_path) as im:
        im = im.convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
        pixels = list(im.getdata())
    flat = []
    for r, g, b in pixels:
        flat.extend([r / 255.0, g / 255.0, b / 255.0])
    return flat


def main():
    classes = list_classes()
    class_to_idx = {name: i for i, name in enumerate(classes)}
    (OUT_DIR / "classes.json").write_text(json.dumps(class_to_idx, indent=2))

    n_features = TARGET_SIZE[0] * TARGET_SIZE[1] * 3
    header = [f"px{i}" for i in range(n_features)] + ["label"]

    with open(OUT_DIR / "features.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for class_name in classes:
            class_dir = RAW_DIR / class_name
            idx = class_to_idx[class_name]
            count = 0
            for img_path in sorted(class_dir.glob("*.jpg")):
                try:
                    features = extract_features(img_path)
                except Exception as e:
                    print(f"skip {img_path}: {e}")
                    continue
                writer.writerow(features + [idx])
                count += 1
            print(f"{class_name} (label={idx}): {count} images -> features extraites")

    print(f"\nTermine. {n_features} features par image. Fichier : {OUT_DIR / 'features.csv'}")


if __name__ == "__main__":
    main()
