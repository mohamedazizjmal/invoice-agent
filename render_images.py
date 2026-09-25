"""
Render synthetic invoice records (from generator.py) to PNG images that look
like a plain scanned/photographed invoice, so a real OCR engine (Tesseract)
can be run on them.

This is the bridge between the pure-text generator (label + clean text) and
the actual OCR pipeline step in the project: we need *images* to measure a
realistic OCR error rate, not just clean text.

Arabic rendering uses Pillow's raqm text layout engine (HarfBuzz + FriBidi
under the hood), which handles Arabic contextual letter-shaping and bidi
reordering automatically -- including tricky mixed-direction lines like
"رقم الفاتورة: FAC-2023-1723" where Latin digits/codes sit inside RTL text.

Font choice matters more than expected here: NotoSansArabic-Regular (the
"correct-looking" dedicated Arabic font) turned out to have NO glyphs for
uppercase Latin letters, '(', ')', '%', '|', or '/' -- confirmed via font
cmap inspection -- which corrupted every line mixing Arabic labels with
Latin invoice codes (tofu boxes). DejaVuSans, by contrast, ships ~165
Arabic codepoints alongside full Latin coverage and renders mixed-script
lines correctly in one pass, so it is used as the single font for both
scripts below.
"""

import os
import random
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

_ARABIC_RANGE = range(0x0600, 0x06FF)


def _has_arabic(s):
    return any(ord(ch) in _ARABIC_RANGE for ch in s)


def shape_line(line):
    """Return (text, direction) for drawing. With raqm, no manual
    reshaping is needed -- we just tell PIL the base direction and it
    handles shaping/bidi itself."""
    return line, ("rtl" if _has_arabic(line) else "ltr")


def render_invoice_image(ocr_text, out_path, noise=True, seed=None):
    if seed is not None:
        random.seed(seed)

    lines = ocr_text.split("\n")
    rendered_lines = []  # text actually drawn -- fair OCR-comparison ground truth
    font_size = 22
    font = ImageFont.truetype(FONT_PATH, font_size, layout_engine=ImageFont.Layout.RAQM)

    line_height = int(font_size * 1.6)
    width = 900
    height = line_height * (len(lines) + 4) + 80

    img = Image.new("L", (width, height), color=255)  # grayscale, white bg
    draw = ImageDraw.Draw(img)

    y = 40
    for line in lines:
        text, direction = shape_line(line)
        rendered_lines.append(text)
        if direction == "rtl":
            bbox = draw.textbbox((0, 0), text, font=font, direction="rtl")
            text_w = bbox[2] - bbox[0]
            x = width - 50 - text_w
            draw.text((x, y), text, font=font, fill=0, direction="rtl")
        else:
            x = 50
            draw.text((x, y), text, font=font, fill=0)
        y += line_height

    if noise:
        img = _add_scan_noise(img, seed=seed)

    img.save(out_path)
    return out_path, "\n".join(rendered_lines)


def _add_scan_noise(img, seed=None):
    """Light, realistic degradation: slight rotation (skew), gaussian-ish
    speckle noise, and a mild blur — simulates a phone photo / low-quality
    scan rather than a pristine rendered page."""
    import numpy as np
    if seed is not None:
        np.random.seed(seed)

    angle = np.random.uniform(-1.2, 1.2)
    img = img.rotate(angle, expand=True, fillcolor=255)

    arr = np.array(img).astype(np.int16)
    speckle = np.random.normal(0, 6, arr.shape).astype(np.int16)
    arr = np.clip(arr + speckle, 0, 255).astype("uint8")
    img = Image.fromarray(arr)

    from PIL import ImageFilter
    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    return img


def render_dataset_images(jsonl_path, out_dir, limit=None, noise=True):
    import json
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rec = json.loads(line)
            rid = rec["labels"]["record_id"]
            img_path = os.path.join(out_dir, f"{rid}.png")
            render_invoice_image(rec["ocr_text"], img_path, noise=noise, seed=i)
            manifest.append({
                "record_id": rid,
                "image_path": img_path,
                "language": rec["labels"]["language"],
                "ground_truth_text": rec["ocr_text"],
                "labels": rec["labels"],
            })
    return manifest


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--no-noise", action="store_true")
    args = p.parse_args()

    manifest = render_dataset_images(args.jsonl, args.out, limit=args.limit, noise=not args.no_noise)
    print(f"Rendered {len(manifest)} images -> {args.out}/")
