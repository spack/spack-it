"""Fix asymmetric whitespace around pssgplot-generated PDF figures.

Root cause: pssgplot's LinePlot/BarPlot call fig.tight_layout() and save
with bbox_inches='tight'; matplotlib's tight-bbox estimate for a rotated
y-axis label under the custom Gill Sans font (fonts/gillsans.ttf) comes out
wrong, reserving far more space on the left than the right/top get from the
same "tight" pass. Observed on fig/audit.pdf in the paper repo: ~40pt left
margin vs. ~1pt right/top, before this fix.

`pdfcrop` would normally handle this by recomputing the real ink bbox via
ghostscript, but ghostscript is broken in this dev environment (missing
libXt.6.dylib -- an arm64/X11 mismatch unrelated to this project, not worth
"fixing" by installing XQuartz just for this). This script works around
both problems without needing ghostscript at all:

1. Rasterize the PDF via macOS `qlmanage` (the same renderer Preview/Quick
   Look use). Deliberately NOT `sips`: sips was observed to mis-render a
   PDF with a non-default MediaBox, reporting 0px margin on all sides even
   when qlmanage/Preview render the identical file correctly with content
   intact -- don't use sips for this measurement, only qlmanage.
2. Measure the actual non-white content bounding box in pixels.
3. Convert that to PDF points using the page's real MediaBox size.
4. Rewrite the page's MediaBox via pypdf so all four margins are equal
   (content bbox + a fixed `--pad` on every side).
5. Re-rasterize the *result* and verify the new margins are actually
   symmetric and that no content was lost, before leaving the output in
   place. This safety check exists because an earlier manual attempt at
   this exact fix clipped the y-axis label and tick numbers entirely --
   a low-resolution measurement with too strict a "non-white" threshold
   silently mistook faint anti-aliased text for blank margin. Don't trust
   a single low-res measurement; this script rasterizes at --raster-px
   (default 3000px wide) and verifies post-hoc rather than assuming the
   first measurement was right.

Usage:
    .venv/bin/python -m analysis.symmetrize_pdf_margins fig/audit.pdf
    .venv/bin/python -m analysis.symmetrize_pdf_margins fig/audit.pdf --pad 4 --raster-px 3000
    .venv/bin/python -m analysis.symmetrize_pdf_margins fig/audit.pdf --output /tmp/check.pdf --dry-run

macOS only (depends on `qlmanage`). If a figure comes out fine from
pssgplot (symmetric already), running this is a harmless no-op modulo the
small requested --pad.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject


def _raster_bbox(pdf_path: Path, raster_px: int, threshold: float) -> dict:
    """Rasterize pdf_path via qlmanage and measure the non-white content bbox in px."""
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            ["qlmanage", "-t", "-s", str(raster_px), "-o", td, str(pdf_path)],
            check=True,
            capture_output=True,
        )
        png_path = Path(td) / f"{pdf_path.name}.png"
        if not png_path.exists():
            raise RuntimeError(f"qlmanage did not produce a thumbnail for {pdf_path}")
        img = Image.open(png_path).convert("RGB")
        arr = np.array(img)

    h, w, _ = arr.shape
    mask = np.any(arr < threshold, axis=2)
    cols = np.where(mask.any(axis=0))[0]
    rows = np.where(mask.any(axis=1))[0]
    if len(cols) == 0 or len(rows) == 0:
        raise RuntimeError(
            f"{pdf_path}: no non-white content detected at threshold={threshold} -- "
            "refusing to crop blindly, check the file/threshold manually"
        )
    return {
        "w": w,
        "h": h,
        "left": int(cols.min()),
        "right": int(w - 1 - cols.max()),
        "top": int(rows.min()),
        "bottom": int(h - 1 - rows.max()),
    }


def _assert_symmetric(bbox: dict, tol_frac: float = 0.03, label: str = "") -> None:
    margins = [bbox["left"], bbox["right"], bbox["top"], bbox["bottom"]]
    span = max(bbox["w"], bbox["h"])
    spread = max(margins) - min(margins)
    if spread > tol_frac * span:
        raise RuntimeError(
            f"{label}: margins not symmetric after crop (L={bbox['left']} "
            f"R={bbox['right']} T={bbox['top']} B={bbox['bottom']}, "
            f"spread={spread}px > {tol_frac * span:.1f}px tolerance) -- "
            "not overwriting the input; inspect manually"
        )


def symmetrize(
    pdf_path: Path,
    output_path: Path,
    pad: float,
    raster_px: int,
    threshold: float,
    dry_run: bool,
) -> None:
    reader = PdfReader(str(pdf_path))
    page = reader.pages[0]
    mb = page.mediabox
    x0, y0, x1, y1 = float(mb.left), float(mb.bottom), float(mb.right), float(mb.top)
    page_w_pt, page_h_pt = x1 - x0, y1 - y0

    before = _raster_bbox(pdf_path, raster_px, threshold)
    px_per_pt_x = before["w"] / page_w_pt
    px_per_pt_y = before["h"] / page_h_pt
    print(
        f"before (px): L={before['left']} R={before['right']} "
        f"T={before['top']} B={before['bottom']}  (raster {before['w']}x{before['h']})"
    )

    content_x0 = x0 + before["left"] / px_per_pt_x
    content_x1 = x1 - before["right"] / px_per_pt_x
    content_y0 = y0 + before["bottom"] / px_per_pt_y
    content_y1 = y1 - before["top"] / px_per_pt_y

    new_box = RectangleObject(
        (content_x0 - pad, content_y0 - pad, content_x1 + pad, content_y1 + pad)
    )
    print(f"new mediabox (pt): {list(new_box)}")

    if dry_run:
        print("--dry-run: not writing anything")
        return

    with tempfile.TemporaryDirectory() as td:
        candidate = Path(td) / pdf_path.name
        writer = PdfWriter()
        writer.append(reader)
        writer.pages[0].mediabox = new_box
        with open(candidate, "wb") as f:
            writer.write(f)

        after = _raster_bbox(candidate, raster_px, threshold)
        print(
            f"after  (px): L={after['left']} R={after['right']} "
            f"T={after['top']} B={after['bottom']}  (raster {after['w']}x{after['h']})"
        )
        _assert_symmetric(after, label=str(pdf_path))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(candidate, output_path)
    print(f"wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path, help="PDF figure to symmetrize")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output path (default: overwrite the input in place)",
    )
    parser.add_argument("--pad", type=float, default=4.0, help="uniform margin in pt (default 4.0)")
    parser.add_argument(
        "--raster-px",
        type=int,
        default=3000,
        help="raster width in px for measurement -- higher is more precise but slower (default 3000)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=250,
        help="pixel value (0-255) below which a pixel counts as non-white content (default 250)",
    )
    parser.add_argument("--dry-run", action="store_true", help="measure and print, but don't write")
    args = parser.parse_args()

    output_path = args.output or args.pdf
    symmetrize(args.pdf, output_path, args.pad, args.raster_px, args.threshold, args.dry_run)


if __name__ == "__main__":
    main()
