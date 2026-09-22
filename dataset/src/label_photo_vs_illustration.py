"""
Heuristique de labellisation photo (1) vs illustration/clipart (-1) sur les
images jasmin, utilisee comme tache pretexte pour demontrer le Modele
Lineaire et le PMC sur un extrait reel du dataset (une seule vraie classe de
fleur etant disponible pour l'instant, une classification fleur/fleur n'est
pas encore possible).

Heuristique (sur l'image originale, avant redimensionnement en features) :
- fraction de pixels quasi blancs (fond uni typique des illustrations)
- nombre de couleurs distinctes (quantifiees) : les illustrations utilisent
  des aplats de couleur, donc beaucoup moins de couleurs qu'une photo.
"""

import csv
import json
from pathlib import Path

from PIL import Image

RAW_DIR = Path(__file__).resolve().parent / "raw" / "jasmin"
OUT_DIR = Path(__file__).resolve().parent / "processed"
OUT_DIR.mkdir(exist_ok=True)

FEATURE_SIZE = (16, 16)
WHITE_THRESHOLD = 240
WHITE_FRACTION_LIMIT = 0.35
UNIQUE_COLORS_LIMIT = 900


def is_photo(image_path) -> bool:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        pixels = list(im.getdata())

    white_count = sum(1 for r, g, b in pixels if r > WHITE_THRESHOLD and g > WHITE_THRESHOLD and b > WHITE_THRESHOLD)
    white_fraction = white_count / len(pixels)

    quantized = {(r // 8, g // 8, b // 8) for r, g, b in pixels}
    n_unique = len(quantized)

    is_illustration = white_fraction > WHITE_FRACTION_LIMIT or n_unique < UNIQUE_COLORS_LIMIT
    return not is_illustration


def extract_features(image_path):
    with Image.open(image_path) as im:
        im = im.convert("RGB").resize(FEATURE_SIZE, Image.LANCZOS)
        pixels = list(im.getdata())
    flat = []
    for r, g, b in pixels:
        flat.extend([r / 255.0, g / 255.0, b / 255.0])
    return flat


def main():
    rows = []
    n_photo = 0
    n_illustration = 0
    for img_path in sorted(RAW_DIR.glob("*.jpg")):
        photo = is_photo(img_path)
        label = 1 if photo else -1
        n_photo += photo
        n_illustration += not photo
        rows.append((img_path.name, extract_features(img_path), label))

    n_features = FEATURE_SIZE[0] * FEATURE_SIZE[1] * 3
    header = ["filename"] + [f"px{i}" for i in range(n_features)] + ["label"]
    with open(OUT_DIR / "photo_vs_illustration.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for filename, features, label in rows:
            writer.writerow([filename] + features + [label])

    summary = {"n_photo": n_photo, "n_illustration": n_illustration, "total": len(rows)}
    (OUT_DIR / "photo_vs_illustration_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"photo={n_photo} illustration={n_illustration} total={len(rows)}")
    print(f"Fichier : {OUT_DIR / 'photo_vs_illustration.csv'}")


if __name__ == "__main__":
    main()
