"""
viz.py
======
Visualization helpers for the CLIPSeg-debris DesignSafe demo. Matplotlib only
(plus optional contextily basemaps); no torch required.

Functions
---------
legend_handles()             debris-class color legend handles (optionally a subset).
show_overlay_row()           up to N debris overlays in a single row.
show_predictions()           grid of [input | mask | overlay] rows.
show_grid_on_basemap()       grid cells over an aerial/web basemap.
show_region_mosaic()         regional imagery + debris-overlay side by side.
plot_debris_choropleth()     per-cell debris-area choropleth on the grid.
class_area_bar()             bar chart of total area per debris class.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

try:                                    # colorize/overlay/palette (no torch)
    from . import debris_common as _dc
except ImportError:                     # flat import fallback
    import debris_common as _dc

PALETTE = _dc.PALETTE
CLASS_NAMES = _dc.CLASS_NAMES


# --------------------------------------------------------------------------- #
def legend_handles(classes=None):
    """Matplotlib patch handles for the debris classes. ``classes`` selects which
    class indices to include (default: all). Pass ``(1, 2)`` for debris-only."""
    from matplotlib.patches import Patch
    idx = range(len(CLASS_NAMES)) if classes is None else classes
    return [Patch(facecolor=np.array(PALETTE[i]) / 255.0, edgecolor="k",
                  label=CLASS_NAMES[i]) for i in idx]


def show_overlay_row(images: Sequence[np.ndarray], masks: Sequence[np.ndarray],
                     titles: Optional[Sequence[str]] = None, n: int = 4,
                     alpha: float = 0.5, panel: float = 3.0):
    """Show up to ``n`` debris overlays (imagery + predicted mask) in a single row of
    equal-size panels, with a debris-only (low/high) legend below."""
    import matplotlib.pyplot as plt

    k = min(len(images), n)
    fig, axes = plt.subplots(1, k, figsize=(panel * k, panel))
    if k == 1:
        axes = [axes]
    for i in range(k):
        rgb = _dc.ensure_rgb_uint8(images[i])
        axes[i].imshow(_dc.overlay(rgb, masks[i], alpha=alpha))
        axes[i].set_xticks([]); axes[i].set_yticks([])
        if titles is not None:
            axes[i].set_title(titles[i], fontsize=9)
    fig.legend(handles=legend_handles(classes=(1, 2)), loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return fig


def show_predictions(images: Sequence[np.ndarray], masks: Sequence[np.ndarray],
                     titles: Optional[Sequence[str]] = None, max_n: int = 8,
                     alpha: float = 0.45, figsize_scale: float = 3.0):
    """Show up to ``max_n`` rows of [input | predicted mask | overlay]."""
    import matplotlib.pyplot as plt

    n = min(len(images), max_n)
    fig, axes = plt.subplots(n, 3, figsize=(3 * figsize_scale, n * figsize_scale))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        rgb = _dc.ensure_rgb_uint8(images[i])
        mask = masks[i]
        axes[i, 0].imshow(rgb)
        axes[i, 1].imshow(_dc.colorize(mask))
        axes[i, 2].imshow(_dc.overlay(rgb, mask, alpha=alpha))
        for j, t in enumerate(("input", "debris mask", "overlay")):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(t)
        label = titles[i] if titles is not None else f"#{i}"
        axes[i, 0].set_ylabel(label, fontsize=9)
    fig.legend(handles=legend_handles(), loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    return fig


def show_grid_on_basemap(grid_gdf, center_lonlat=None, ax=None, basemap: bool = True,
                         title: str = "Grid cells over aerial basemap"):
    """Plot grid-cell outlines over a web/aerial basemap (needs contextily)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    g = grid_gdf.to_crs("EPSG:3857")
    g.boundary.plot(ax=ax, edgecolor="cyan", linewidth=1.2)
    if center_lonlat is not None:
        import geopandas as gpd
        from shapely.geometry import Point
        c = gpd.GeoSeries([Point(center_lonlat[1], center_lonlat[0])],
                          crs="EPSG:4326").to_crs("EPSG:3857")
        c.plot(ax=ax, color="red", marker="*", markersize=180, zorder=5)
    if basemap:
        try:
            import contextily as ctx
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery,
                            crs="EPSG:3857", attribution_size=6)
        except Exception as exc:  # pragma: no cover
            print(f"[viz] basemap unavailable ({exc}); showing outlines only.")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def show_region_mosaic(imagery_path, mask_path=None, alpha: float = 0.5,
                       title: str = "Estero Island — regional debris map"):
    """Show the regional imagery mosaic and (optionally) the debris overlay."""
    import matplotlib.pyplot as plt
    import rasterio

    with rasterio.open(imagery_path) as src:
        img = src.read([1, 2, 3]).transpose(1, 2, 0)
    img = _dc.ensure_rgb_uint8(img)

    if mask_path is None:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img); ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        return fig

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    mask = np.where(mask == 255, 0, mask)  # nodata -> no-debris for display

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(img); axes[0].set_title("NOAA ERI imagery mosaic")
    axes[1].imshow(_dc.overlay(img, mask, alpha=alpha))
    axes[1].set_title("CLIPSeg-debris overlay")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    # Debris-only legend (low/high), placed below the panels so it doesn't overlap.
    fig.legend(handles=legend_handles(classes=(1, 2)), loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04))
    return fig


def plot_debris_choropleth(demo, summary_df, column: str = "debris_area_m2",
                           title: Optional[str] = None):
    """Choropleth of a per-cell metric on the grid, over an aerial basemap."""
    import matplotlib.pyplot as plt

    grid = demo.build_grid().merge(summary_df, on="grid_id", how="left")
    g = grid.to_crs("EPSG:3857")
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    g.plot(ax=ax, column=column, cmap="YlOrRd", alpha=0.65,
           edgecolor="white", linewidth=0.4, legend=True,
           legend_kwds={"label": column, "shrink": 0.6})
    try:
        import contextily as ctx
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery,
                        crs="EPSG:3857", attribution_size=6)
    except Exception:
        pass
    ax.set_title(title or f"Per-cell {column}")
    ax.set_xticks([]); ax.set_yticks([])
    return fig


def class_area_bar(summary_df, cell_area_m2: float = 2500.0,
                   title: str = "Total mapped debris area by class"):
    """Bar chart of total low/high debris area across all cells."""
    import matplotlib.pyplot as plt

    low = float(summary_df["frac_low"].sum() * cell_area_m2)
    high = float(summary_df["frac_high"].sum() * cell_area_m2)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["low-density", "high-density"], [low, high],
           color=[np.array(PALETTE[1]) / 255.0, np.array(PALETTE[2]) / 255.0],
           edgecolor="k")
    ax.set_ylabel("area (m$^2$)")
    ax.set_title(title)
    for i, v in enumerate([low, high]):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig
