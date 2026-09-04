#!/usr/bin/env python3
"""Generate placeholder PNG figures for development/review purposes.

Useful for:
- Testing document layout before all figures are ready
- Creating stub figures with labels to see where they'll appear
- Verifying the image injection mechanism works end-to-end

Usage:
    python make_placeholder_figures.py [fig_names ...]
    python make_placeholder_figures.py          # all missing figures
    python make_placeholder_figures.py 01-structure-before-search 05-band-discovery
"""

import os
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: Pillow not installed. Install with: pip install Pillow")
    sys.exit(1)

HERE = Path(__file__).parent
FIG_DIR = HERE / "prose" / "fig"

# All figures referenced in the prose, by chapter
FIGURES = {
    "01-structure-before-search": "Ch 1: Structure before search",
    "01-pipeline-roadmap": "Ch 1: Pipeline roadmap",
    "02-fis-components": "Ch 2: FIS components",
    "02-tsk-inference": "Ch 2: TSK inference",
    "02-norm-surfaces": "Ch 2: norm/conorm surfaces",
    "02-vat-rdi": "Ch 2: VAT RDI",
    "02-minimax-ultrametric": "Ch 2: minimax distance",
    "02-persistence": "Ch 2: Persistence",
    "03-pvat-reorder": "Ch 3: pVAT reorder",
    "03-inplace-permutation": "Ch 3: in-place permutation",
    "03-stitch": "Ch 3: stitch",
    "03-complexity-fit": "Ch 3: Complexity fit [GENERATED]",
    "03-three-arm-seconds": "Ch 3: three arms in seconds",
    "03-memory-ceiling": "Ch 3: memory ceiling",
    "04-mog-classification": "Ch 4: MoG classification",
    "04-output-partitioning": "Ch 4: output partitioning",
    "04-rule-count": "Ch 4: rule count",
    "04-anomaly-geometry": "Ch 4: anomaly geometry",
    "04-speedup": "Ch 4: training-time bars",
    "04-anomaly-sweep": "Ch 4: Anomaly sweep",
    "04-mf-dedup-sweep": "Ch 4: dedup sweep",
    "04-rtiot-confusion": "Ch 4: RT-IoT confusion",
    "05-minimax-transform": "Ch 5: Minimax transform",
    "05-band-discovery": "Ch 5: Band discovery",
    "05-persistence-ramp": "Ch 5: Persistence ramp",
    "06-fuzzy-tree": "Ch 6: Fuzzy tree",
    "06-hme-structure": "Ch 6: HME structure",
    "06-mimo-rollout": "Ch 6: MIMO rollout",
}


def make_placeholder(name: str, title: str, path: Path) -> bool:
    """Create a minimal placeholder PNG for a figure.

    Returns True if created, False if already exists.
    """
    if path.exists():
        return False

    # Create a simple placeholder image
    width, height = 640, 480
    img = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fall back to default
    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/liberation/LiberationSans-Bold.ttf", 24
        )
        label_font = ImageFont.truetype(
            "/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 14
        )
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    # Draw border
    draw.rectangle(
        [(10, 10), (width - 10, height - 10)], outline=(200, 200, 200), width=2
    )

    # Draw title
    draw.text((width // 2, 40), title, fill=(80, 80, 80), font=title_font, anchor="mm")

    # Draw label
    draw.text(
        (width // 2, height // 2),
        "[Placeholder figure]",
        fill=(150, 150, 150),
        font=label_font,
        anchor="mm",
    )
    draw.text(
        (width // 2, height // 2 + 30),
        f"File: {path.name}",
        fill=(180, 180, 180),
        font=label_font,
        anchor="mm",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG")
    return True


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which figures to create
    if len(sys.argv) > 1:
        requested = set(sys.argv[1:])
        to_create = {k: v for k, v in FIGURES.items() if k in requested}
        unknown = requested - set(FIGURES.keys())
        if unknown:
            print(f"Unknown figures: {', '.join(sorted(unknown))}")
            return 1
    else:
        # Create all missing figures
        to_create = FIGURES

    print(f"Creating placeholders in {FIG_DIR} ...\n")
    created = 0
    skipped = 0

    for name, title in sorted(to_create.items()):
        png_path = FIG_DIR / f"{name}.png"
        eps_path = FIG_DIR / f"{name}.eps"

        # Skip if already exists
        if png_path.exists():
            print(f"  skip {name} (already exists)")
            skipped += 1
            continue

        if make_placeholder(name, title, png_path):
            print(f"  ✓ {name}")
            created += 1

            # Also create a dummy EPS (the harness would generate vector versions)
            # For now, just create a minimal text-based EPS
            if not eps_path.exists():
                with open(eps_path, "w") as f:
                    f.write(f"%!PS-Adobe-3.0 EPSF-3.0\n")
                    f.write(f"%%BoundingBox: 0 0 640 480\n")
                    f.write(f"/Courier findfont 12 scalefont setfont\n")
                    f.write(f"100 400 moveto\n")
                    f.write(f"({title} — placeholder) show\n")
                    f.write(f"showpage\n")

    print(f"\nCreated {created}, skipped {skipped}")
    if created > 0:
        print(
            f"\nRun 'python research/proposal-defense/build_pdf.py' to rebuild the PDF"
        )
        print("and check the image injection diagnostics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
