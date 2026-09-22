"""
Scraping d'images de fleurs (jasmin) depuis Bing Images.
Usage:
    python scrape_flowers.py                     # toutes les classes, cible par defaut
    python scrape_flowers.py --classes jasmin     # une seule classe
    python scrape_flowers.py --target 500          # cible personnalisee
    python scrape_flowers.py --no-headless         # voir le navigateur tourner
"""

import argparse
import json
import random
import time
from io import BytesIO
from pathlib import Path

import imagehash
import requests
from bs4 import BeautifulSoup
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

OUT_DIR = Path(__file__).resolve().parent.parent / "dataset" / "raw"
IMG_SIZE = (224, 224)
MIN_SIDE = 100
TARGET_PER_CLASS_DEFAULT = 2000
MAX_SCROLLS_PER_QUERY = 40
STALE_SCROLL_LIMIT = 4
MAX_CANDIDATES_PER_QUERY = 400
REQUEST_TIMEOUT = 8
MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024

DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

QUERIES = {
    "jasmin": [
        "jasmine flower",
        "jasmine flower macro",
        "jasmine flower close up",
        "fleur de jasmin",
        "jasmine plant blossom",
        "jasmine flower garden",
        "white jasmine flower",
        "jasmine vine flower",
        "arabian jasmine flower",
        "jasmine flower photography",
        "jasmine flower field",
        "jasmine bush blossom",
        "jasmine flower branch",
        "night blooming jasmine flower",
        "jasmine flower petals",
        "jasminum flower",
    ],
}


def make_driver(headless: bool):
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1600,1200")
    options.add_argument("--disable-gpu")
    options.add_argument("--log-level=3")
    options.add_argument(f"user-agent={DOWNLOAD_HEADERS['User-Agent']}")
    return webdriver.Chrome(options=options)


def collect_urls_for_query(driver, query: str) -> set[str]:
    search_url = f"https://www.bing.com/images/search?q={requests.utils.quote(query)}&form=HDRSC2"
    driver.get(search_url)
    time.sleep(1.5)

    found: set[str] = set()
    stale_rounds = 0
    for _ in range(MAX_SCROLLS_PER_QUERY):
        soup = BeautifulSoup(driver.page_source, "lxml")
        before = len(found)
        for tag in soup.select("a.iusc"):
            m_attr = tag.get("m")
            if not m_attr:
                continue
            try:
                data = json.loads(m_attr)
            except json.JSONDecodeError:
                continue
            murl = data.get("murl")
            if murl:
                found.add(murl)

        if len(found) >= MAX_CANDIDATES_PER_QUERY:
            break

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(0.8, 1.5))

        try:
            more_button = driver.find_element(By.CSS_SELECTOR, "a.btn_seemore")
            if more_button.is_displayed():
                more_button.click()
                time.sleep(1.0)
        except Exception:
            pass

        if len(found) == before:
            stale_rounds += 1
            if stale_rounds >= STALE_SCROLL_LIMIT:
                break
        else:
            stale_rounds = 0

    return found


def load_or_build_hash_index(class_dir: Path) -> dict[str, str]:
    hash_file = class_dir / ".hashes.json"
    if hash_file.exists():
        return json.loads(hash_file.read_text())

    index: dict[str, str] = {}
    for img_path in class_dir.glob("*.jpg"):
        try:
            with Image.open(img_path) as im:
                index[str(imagehash.phash(im))] = img_path.name
        except Exception:
            continue
    save_hash_index(class_dir, index)
    return index


def save_hash_index(class_dir: Path, index: dict[str, str]) -> None:
    (class_dir / ".hashes.json").write_text(json.dumps(index))


def download_and_save(url: str, class_dir: Path, class_name: str, counter: int,
                       hash_index: dict[str, str]) -> bool:
    try:
        resp = requests.get(url, headers=DOWNLOAD_HEADERS, timeout=REQUEST_TIMEOUT, stream=True)
        if resp.status_code != 200:
            return False
        content_type = resp.headers.get("Content-Type", "")
        if not content_type.startswith("image"):
            return False

        content = resp.raw.read(MAX_DOWNLOAD_BYTES, decode_content=True)
        if not content:
            return False

        image = Image.open(BytesIO(content))
        image.load()
        if min(image.size) < MIN_SIDE:
            return False

        image = image.convert("RGB")
        phash = str(imagehash.phash(image))
        if phash in hash_index:
            return False

        resized = image.resize(IMG_SIZE, Image.LANCZOS)
        filename = f"{class_name}_{counter:05d}.jpg"
        resized.save(class_dir / filename, "JPEG", quality=90)

        hash_index[phash] = filename
        return True
    except Exception:
        return False


def scrape_class(driver, class_name: str, target: int) -> None:
    class_dir = OUT_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    hash_index = load_or_build_hash_index(class_dir)
    counter = len(list(class_dir.glob("*.jpg")))

    print(f"\n=== {class_name} : {counter}/{target} images deja presentes ===")
    if counter >= target:
        print(f"[{class_name}] cible deja atteinte, on passe.")
        return

    queries = QUERIES[class_name]
    for query in queries:
        if counter >= target:
            break
        print(f"[{class_name}] requete: '{query}'")
        candidates = collect_urls_for_query(driver, query)
        print(f"[{class_name}]   {len(candidates)} URLs candidates trouvees")

        new_this_query = 0
        for url in candidates:
            if counter >= target:
                break
            ok = download_and_save(url, class_dir, class_name, counter, hash_index)
            if ok:
                counter += 1
                new_this_query += 1
                if counter % 50 == 0:
                    save_hash_index(class_dir, hash_index)
                    print(f"[{class_name}]   progression: {counter}/{target}")
            time.sleep(random.uniform(0.2, 0.6))

        save_hash_index(class_dir, hash_index)
        print(f"[{class_name}]   +{new_this_query} images retenues sur cette requete "
              f"(total {counter}/{target})")
        time.sleep(random.uniform(2, 5))

    print(f"=== {class_name} termine : {counter}/{target} ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=list(QUERIES.keys()),
                         choices=list(QUERIES.keys()))
    parser.add_argument("--target", type=int, default=TARGET_PER_CLASS_DEFAULT)
    parser.add_argument("--no-headless", action="store_true")
    args = parser.parse_args()

    driver = make_driver(headless=not args.no_headless)
    try:
        for class_name in args.classes:
            scrape_class(driver, class_name, args.target)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
