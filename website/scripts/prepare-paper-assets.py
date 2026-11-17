"""Copy the supplied PDF and render its figures for the project page.

Only reads the supplied paper. Does not recompile it or touch research files.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
if len(sys.argv) != 2:
    raise SystemExit("Usage: python scripts/prepare-paper-assets.py /path/to/ICDE2027.pdf")
SOURCE = Path(sys.argv[1])
PUBLIC = ROOT / "public"
ASSETS = ROOT / "src/assets/paper"
PUBLIC.mkdir(exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)
pdf = fitz.open(SOURCE)
assert len(pdf) == 14
first = pdf[0].get_text()
assert "From Rules to Rows" in first
for name in ["Zhili Wang", "Heitor Murilo Gomes", "Xingquan Zhu", "Yi He"]:
    assert name in first

shutil.copyfile(SOURCE, PUBLIC / "ICDE2027.pdf")
assert SOURCE.read_bytes() == (PUBLIC / "ICDE2027.pdf").read_bytes()

# PDF page coordinates in points. Captions are reproduced as accessible HTML.
crops = {
    "method.png": {"page": 7, "rect": [49, 48, 564, 242], "figure": "6"},
    "validity.png": {"page": 10, "rect": [49, 51, 300, 151], "figure": "8, upper panel"},
    "quality.png": {"page": 10, "rect": [312, 214, 564, 312], "figure": "9"},
    "adaptability.png": {"page": 12, "rect": [49, 220, 564, 341], "figure": "11"},
}
for name, crop in crops.items():
    page = pdf[crop["page"] - 1]
    image = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=fitz.Rect(crop["rect"]), alpha=False)
    image.save(str(ASSETS / name))
    crop["rendered_pixels"] = [image.width, image.height]
    crop["sha256"] = hashlib.sha256((ASSETS / name).read_bytes()).hexdigest()

record = {
    "authoritative_input": "User-supplied ICDE2027.pdf",
    "pdf_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "pdf_pages": len(pdf),
    "authors": ["Zhili Wang", "Heitor Murilo Gomes", "Xingquan Zhu", "Yi He"],
    "figures": crops,
    "notes": "Direct PDF renders; no regenerated experimental graphics or modified scientific values.",
}
(ROOT / "PAPER_PROVENANCE.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"pdf_pages": len(pdf), "copied_pdf_sha256": record["pdf_sha256"], "figures": list(crops)}, ensure_ascii=True))
