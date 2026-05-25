#!/usr/bin/env python
"""
run_inference.py — batch CLIPSeg-debris inference for a DesignSafe HPC job.

Runs the **official** model (cloned by job_infer.sh into ./CLIPSeg-debris) on the
staged GeoTIFF tiles (``inputs/grid-*-imagery.tif``) and writes georeferenced
3-class mask GeoTIFFs + colorized PNGs + a per-tile summary CSV to ``outputs/``.

Imports the same thin helpers used in the notebook (``clipseg_official`` +
``debris_common``), which are staged alongside this script.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # debris_common.py + clipseg_official.py live here

import clipseg_official as co       # noqa: E402
import debris_common as dc          # noqa: E402


def find_weights(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    env = os.environ.get("WEIGHTS_FILENAME")
    wdir = HERE / "weights"
    if env and (wdir / env).exists():
        return wdir / env
    for pat in ("*.safetensors", "*.ckpt"):
        hits = sorted(wdir.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No weights in {wdir} (or pass --weights).")


def main():
    ap = argparse.ArgumentParser(description="CLIPSeg-debris HPC inference")
    ap.add_argument("--input-dir", default="inputs")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--repo", default=str(HERE / "CLIPSeg-debris"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out = Path(args.output_dir)
    (out / "mask_tif").mkdir(parents=True, exist_ok=True)
    (out / "mask_png").mkdir(parents=True, exist_ok=True)

    tiles = sorted(Path(args.input_dir).glob("grid-*-imagery.tif"))
    if not tiles:
        sys.exit(f"No 'grid-*-imagery.tif' tiles in {args.input_dir}")
    print(f"[run] {len(tiles)} tiles; repo={args.repo}")

    repo = co.resolve_repo(Path(args.repo).parent, Path(args.repo).name)
    model = co.load_model(repo, find_weights(args.weights), device=args.device)
    device = co.get_device() if args.device is None else args.device

    rgbs, profiles, names = [], [], []
    for t in tiles:
        with rasterio.open(t) as src:
            rgbs.append(src.read([1, 2, 3]).transpose(1, 2, 0))
            profiles.append(src.profile.copy())
        names.append(t.stem.replace("-imagery", ""))

    t0 = time.time()
    masks = co.predict_images(model, rgbs, device=device, progress=True)
    print(f"[run] {len(masks)} tiles in {time.time()-t0:.1f}s")

    rows = []
    for name, mask, prof in zip(names, masks, profiles):
        dc.save_mask_geotiff(mask, prof, out / "mask_tif" / f"{name}-mask.tif")
        Image.fromarray(dc.colorize(mask)).save(out / "mask_png" / f"{name}-mask.png")
        rows.append(dict(tile=name, **dc.class_fractions(mask)))

    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[run] done -> {out}")


if __name__ == "__main__":
    main()
