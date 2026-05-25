"""
debris_common.py
================
Small, dependency-light helpers shared across the demo: the 3-class definition,
the model's normalization constants, and image/mask utilities (colorize,
overlay, per-class stats, GeoTIFF I/O). **No torch and no model code here** — so
this module imports anywhere (it is also safe to import on a machine that only
has the geospatial stack).

The constants below mirror the *official* CLIPSeg-debris dataset/inference
(`src/data/components/debris_one_hot.py`, `src/models/clipseg/lightning_modules.py`):
3 densities prompted as ``"a photo of {density}"`` and a fixed RGB normalization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

# --------------------------------------------------------------------------- #
# The 3 debris classes (order defines class indices 0, 1, 2)
# --------------------------------------------------------------------------- #

#: Density words used by the official dataset (`text_prompts`).
DENSITIES = ("no debris", "debris at low density", "debris at high density")

#: Prompts actually fed to CLIPSeg-debris — the official template is
#: ``"a photo of {density}"`` (see CLIPSegLitModule.predict_step ->
#: model.sample_prompts(..., prompt_list=('a photo of {}',))).
PROMPTS = tuple(f"a photo of {d}" for d in DENSITIES)

#: Short, human-readable class names.
CLASS_NAMES = ("no debris", "low-density debris", "high-density debris")

#: Official RGB normalization (NOT ImageNet) from the debris dataset transform.
NORM_MEAN = np.array([0.57784108, 0.5724125, 0.5619426], dtype=np.float32)
NORM_STD = np.array([0.24724819, 0.24302182, 0.23344601], dtype=np.float32)

#: Display palette: black / amber / red (no / low / high debris).
PALETTE = np.array([[0, 0, 0], [255, 204, 0], [255, 0, 0]], dtype=np.uint8)

#: Native model input size (official datamodule resizes to 256x256).
MODEL_INPUT_SIZE = 256


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #

def ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Coerce an array to HxWx3 uint8 RGB."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = (arr * 255 if arr.max() <= 1.0 else arr).clip(0, 255).astype(np.uint8)
    return arr


def resize_rgb(img: np.ndarray, size: int = MODEL_INPUT_SIZE) -> np.ndarray:
    """Resize HxWx3 uint8 to (size, size) bilinearly (cv2, matching the dataset)."""
    import cv2
    img = ensure_rgb_uint8(img)
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def normalize_chw(img: np.ndarray, size: int = MODEL_INPUT_SIZE) -> np.ndarray:
    """RGB uint8 (any size) -> normalized CHW float32 at (size, size).

    Matches the official transform: resize -> /255 -> (x-mean)/std -> CHW.
    """
    rgb = resize_rgb(img, size).astype(np.float32) / 255.0
    rgb = (rgb - NORM_MEAN) / NORM_STD
    return rgb.transpose(2, 0, 1)


def load_rgb(path: Union[str, Path], size: Optional[int] = None) -> np.ndarray:
    """Load PNG/JPG/GeoTIFF as HxWx3 uint8 RGB (GeoTIFF via rasterio, else PIL)."""
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        import rasterio
        with rasterio.open(path) as src:
            n = min(3, src.count)
            arr = src.read(list(range(1, n + 1))).transpose(1, 2, 0)
        rgb = ensure_rgb_uint8(arr)
    else:
        from PIL import Image
        rgb = ensure_rgb_uint8(np.asarray(Image.open(path).convert("RGB")))
    return resize_rgb(rgb, size) if size else rgb


# --------------------------------------------------------------------------- #
# Mask helpers
# --------------------------------------------------------------------------- #

def colorize(mask: np.ndarray) -> np.ndarray:
    """3-class mask -> RGB image using the debris PALETTE."""
    return PALETTE[np.asarray(mask).astype(np.int64)]


def overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend the colorized mask over the RGB image (debris pixels only)."""
    rgb = ensure_rgb_uint8(rgb)
    mask = np.asarray(mask)
    if mask.shape[:2] != rgb.shape[:2]:
        import cv2
        mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    color = colorize(mask).astype(np.float32)
    out = rgb.astype(np.float32).copy()
    debris = mask > 0
    out[debris] = (1 - alpha) * out[debris] + alpha * color[debris]
    return out.clip(0, 255).astype(np.uint8)


def class_fractions(mask: np.ndarray) -> Dict[str, float]:
    """Fraction of pixels in each class, keyed by CLASS_NAMES."""
    mask = np.asarray(mask)
    total = mask.size
    return {name: float((mask == i).sum()) / total for i, name in enumerate(CLASS_NAMES)}


def save_mask_geotiff(mask: np.ndarray, src_profile: dict, out_path: Union[str, Path]) -> Path:
    """Write a 3-class mask as a single-band uint8 GeoTIFF, copying geo metadata
    from ``src_profile`` (e.g. the imagery tile's profile)."""
    import rasterio
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(src_profile)
    profile.update(count=1, dtype="uint8", nodata=255, compress="deflate",
                   height=mask.shape[0], width=mask.shape[1])
    profile.pop("photometric", None)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)
    return out_path
