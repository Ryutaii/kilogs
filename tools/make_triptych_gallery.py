#!/usr/bin/env python3
from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageOps

DEFAULT_OUTDIR = Path(
    "results/lego/feat_t_full/runs/teacher_full_rehab_masked_white/renders/step_001000/rgba_white/debug_diffs"
)


def find_triptychs(outdir: Path) -> List[Path]:
    return sorted(p for p in outdir.glob("*_triptych.png") if p.is_file())


def write_gallery_html(outdir: Path, images: Iterable[Path]) -> Path:
    html_path = outdir / "index.html"
    with html_path.open("w", encoding="utf-8") as fp:
        fp.write(
            "<!doctype html><meta charset='utf-8'><title>Triptychs</title>"
            "<style>body{font-family:sans-serif} .grid{display:grid;"
            "grid-template-columns:repeat(auto-fill,minmax(320px,1fr));"
            "gap:12px} figure{margin:0}</style>"
            "<h1>Worst-N Triptychs</h1><div class='grid'>"
        )
        for path in images:
            escaped = html.escape(path.name)
            fp.write(
                f"<figure><img src='{escaped}' style='width:100%'><figcaption>{escaped}</figcaption></figure>"
            )
        fp.write("</div>")
    return html_path


def build_contact_sheet(images: Iterable[Path], outdir: Path, columns: int = 4) -> Path:
    thumbs = []
    for path in images:
        image = Image.open(path).convert("RGB")
        thumb_height = 240
        scale = thumb_height / image.height
        image = image.resize((int(image.width * scale), thumb_height), Image.BICUBIC)
        image = ImageOps.contain(image, (960, thumb_height))
        thumbs.append(image)

    if not thumbs:
        raise SystemExit("No thumbnails available for contact sheet generation")

    cell_w = max(thumb.width for thumb in thumbs)
    cell_h = max(thumb.height for thumb in thumbs)
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), (20, 20, 20))

    for idx, thumb in enumerate(thumbs):
        row = idx // columns
        col = idx % columns
        x = col * cell_w + (cell_w - thumb.width) // 2
        y = row * cell_h + (cell_h - thumb.height) // 2
        sheet.paste(thumb, (x, y))

    sheet_path = outdir / "contact_sheet.png"
    sheet.save(sheet_path)
    return sheet_path


def main(outdir: Path) -> None:
    images = find_triptychs(outdir)
    if not images:
        raise SystemExit(f"[ERR] no triptychs in {outdir}")

    html_path = write_gallery_html(outdir, images)
    sheet_path = build_contact_sheet(images, outdir)
    print(f"[done] wrote {html_path} and {sheet_path}")


if __name__ == "__main__":
    main(DEFAULT_OUTDIR)
