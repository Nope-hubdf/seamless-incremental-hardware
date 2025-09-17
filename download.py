import os
import json
from tqdm import tqdm
import requests
import cairosvg
from concurrent.futures import ThreadPoolExecutor

# INPUT_DATA = {
#     "data": {
#         "2615": {
#             "combinations": {
#                 "2615": [{
#                     "gStaticUrl": "https://www.gstatic.com/android/keyboard/emojikitchen/20201001/u2615/u2615_u2615.png",
#                     "leftEmojiCodepoint": "2615",
#                     "rightEmojiCodepoint": "2615",
#                     "isLatest": True
#                 }],
#                 "2753": [{
#                     "gStaticUrl": "https://www.gstatic.com/android/keyboard/emojikitchen/20250130/u2753/u2753_u2615.png",
#                     "leftEmojiCodepoint": "2753",
#                     "rightEmojiCodepoint": "2615",
#                     "isLatest": True
#                 }],
#                 "2757": [{
#                     "gStaticUrl": "https://www.gstatic.com/android/keyboard/emojikitchen/20240530/u2757/u2757_u2615.png",
#                     "leftEmojiCodepoint": "2757",
#                     "rightEmojiCodepoint": "2615",
#                     "isLatest": True
#                 }]
#             }
#         }
#     }
# }
INPUT_DATA = json.load(open("metadata.json", "r", encoding="utf-8"))


def download_file(http_session, url, dest_path, timeout=10):
    try:
        with http_session.get(url, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        return False


dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

latest_pairs = {}
for item in INPUT_DATA.get("data", {}).values():
    for combos in item.get("combinations", {}).values():
        for combo in combos:
            if combo.get("isLatest"):
                left = str(combo.get("leftEmojiCodepoint") or "")
                right = str(combo.get("rightEmojiCodepoint") or "")
                url = combo.get("gStaticUrl")
                if left and right and url:
                    latest_pairs[(left, right)] = url

if not latest_pairs:
    tqdm.write("No latest pairs found")
    raise SystemExit

def download_worker(kv):
    (left, right), image_url = kv
    out_name = f"{left}_{right}.png"
    out_path = os.path.join(dataset_dir, out_name)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return f"SKIP {out_name} <- {image_url}"
    with requests.Session() as s:
        ok = download_file(s, image_url, out_path)
    return f"{"OK" if ok else "ERR"} {out_name} <- {image_url}"

max_workers = min(16, (os.cpu_count() or 4) * 4)
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    results = ex.map(download_worker, latest_pairs.items())
    for res in tqdm(results, total=len(latest_pairs), desc="Downloading combinations", unit="pair"):
        tqdm.write(res)

emoji_codes = {code for pair in latest_pairs for code in pair}

def convert_worker(code):
    png_path = os.path.join(dataset_dir, f"{code}.png")
    if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
        return f"SKIP {code}.png"
    svg_url = f"https://fonts.gstatic.com/s/e/notoemoji/latest/{code}/emoji.svg"
    with requests.Session() as s:
        try:
            resp = s.get(svg_url, timeout=10)
            resp.raise_for_status()
            try:
                cairosvg.svg2png(bytestring=resp.content, write_to=png_path, output_width=536, output_height=536)
                return f"CONV {code}.svg -> {code}.png"
            except Exception:
                return f"CONV ERR {code}.svg"
        except Exception:
            return f"ERR {code}.svg <- {svg_url}"

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    codes_sorted = sorted(emoji_codes)
    results = ex.map(convert_worker, codes_sorted)
    for res in tqdm(results, total=len(codes_sorted), desc="SVG conversions (one per emoji)", unit="svg"):
        tqdm.write(res)

tqdm.write("Done. Files in " + dataset_dir)


