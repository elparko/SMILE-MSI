#!/usr/bin/env python3
"""Build the macOS app icon ("SMILE MSI.icns") from the SMILE MSI logo.

Composites the wide logo onto a white, rounded-corner tile (the modern macOS
"floating squircle" look) so it reads well at every size, then emits the full
.iconset and runs iconutil. Re-run after changing scripts/assets/smile_msi_logo.png:

    ./.venv/bin/python scripts/make_icon.py

The output ("scripts/SMILE MSI.icns") is picked up automatically by
scripts/make_macos_app.sh.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SRC = HERE / "assets" / "smile_msi_logo.png"
OUT = HERE / "SMILE MSI.icns"

MASTER = 1024          # canvas edge for the master icon
TILE_INSET = 0.08      # transparent margin around the white tile (fraction of edge)
TILE_RADIUS = 0.22     # corner radius of the white tile (fraction of tile edge)
LOGO_PAD = 0.10        # padding between the logo and the tile edge (fraction of tile)


def build_master() -> Image.Image:
    logo = Image.open(SRC).convert("RGBA")
    canvas = Image.new("RGBA", (MASTER, MASTER), (0, 0, 0, 0))

    inset = int(MASTER * TILE_INSET)
    tile_box = (inset, inset, MASTER - inset, MASTER - inset)
    tile_w = tile_box[2] - tile_box[0]
    radius = int(tile_w * TILE_RADIUS)

    # White rounded tile.
    tile = Image.new("RGBA", (tile_w, tile_w), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    d.rounded_rectangle((0, 0, tile_w - 1, tile_w - 1), radius=radius,
                        fill=(255, 255, 255, 255))
    canvas.paste(tile, (tile_box[0], tile_box[1]), tile)

    # Scale the logo to fit inside the padded tile, keeping aspect ratio.
    avail = int(tile_w * (1 - 2 * LOGO_PAD))
    lw, lh = logo.size
    scale = min(avail / lw, avail / lh)
    new = (max(1, int(lw * scale)), max(1, int(lh * scale)))
    logo = logo.resize(new, Image.LANCZOS)

    # Center the logo within the tile.
    cx = tile_box[0] + tile_w // 2
    cy = tile_box[1] + tile_w // 2
    canvas.paste(logo, (cx - new[0] // 2, cy - new[1] // 2), logo)
    return canvas


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing source logo: {SRC}")
    master = build_master()

    sizes = [16, 32, 64, 128, 256, 512, 1024]
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "SMILE MSI.iconset"
        iconset.mkdir()
        for px in sizes:
            img = master.resize((px, px), Image.LANCZOS)
            if px in (32, 64, 256, 512):           # @2x of the half size
                img.save(iconset / f"icon_{px // 2}x{px // 2}@2x.png")
            if px <= 512:                           # 1x entries up to 512
                img.save(iconset / f"icon_{px}x{px}.png")
            if px == 1024:
                img.save(iconset / "icon_512x512@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(OUT)],
                       check=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
