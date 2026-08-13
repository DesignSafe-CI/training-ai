"""
regional.py
===========
A compact, self-contained "regional pipeline in a class" for the DesignSafe
CLIPSeg-debris demo.

``RegionalDebrisDemo`` reproduces — in a single readable class — the parts of
the production NSF debris regional pipeline that matter for a small live demo:

  1. Build an ``n x n`` block of square grid cells (default 10x10 of 50 m)
     centered on a lat/lon, in a projected CRS (UTM).
  2. Pull the matching post-event NOAA **Emergency Response Imagery (ERI)** for
     a hurricane straight from the public NOAA S3 bucket, using *windowed*
     ``/vsicurl`` reads (only the bytes overlapping each cell are fetched — no
     whole-tile downloads), reprojected/resampled to 256x256 RGB GeoTIFFs.
  3. (Optionally) run CLIPSeg-debris on each cell tile.
  4. Mosaic the per-cell imagery / masks back into regional GeoTIFFs.

The default region is **Estero Island, FL (Fort Myers Beach)** after
**Hurricane Ian (2022)** — the model's training region — using the NOAA ERI
flight ``20220930d_RGB``.

Only the geospatial stack (rasterio / geopandas / shapely / pyproj) is needed
for steps 1, 2 and 4; torch + CLIPSeg are imported lazily only in step 3.
"""

from __future__ import annotations

import os
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject
from shapely.geometry import box, mapping


# --------------------------------------------------------------------------- #
# NOAA Emergency Response Imagery (ERI) sources  (public S3, no credentials)
# --------------------------------------------------------------------------- #

@dataclass
class ERISource:
    """A NOAA ERI flight: a tile-index archive + a base URL for the COG tiles."""
    name: str
    tile_index_url: str           # .tar / .zip holding the tile-index shapefile
    tif_base_url: str             # prefix; basename of each index row is appended
    crs_hint: str = "EPSG:4326"   # ERI tiles are delivered in WGS84 lat/lon
    url_column: Optional[str] = None  # override auto-detection of the filename column


#: Hurricane Ian (2022) ERI flight covering Estero Island / Fort Myers Beach.
ESTERO_IAN_ERI = ERISource(
    name="20220930d_RGB",
    tile_index_url=(
        "https://noaa-eri-pds.s3.amazonaws.com/2022_Hurricane_Ian/"
        "20220930d_RGB/tile_index_20220930d_RGB.tar"
    ),
    tif_base_url=(
        "https://noaa-eri-pds.s3.amazonaws.com/2022_Hurricane_Ian/20220930d_RGB"
    ),
    crs_hint="EPSG:4326",
)

#: A couple of other public ERI flights, handy for "try another region" demos.
ERI_SOURCES = {
    "estero_ian_2022": ESTERO_IAN_ERI,
    "treasure_island_milton_2024": ERISource(
        name="20241011d_RGB",
        tile_index_url=(
            "https://noaa-eri-pds.s3.amazonaws.com/2024_Hurricane_Milton/"
            "20241011d_RGB/tile_index_20241011d_RGB.tar"
        ),
        tif_base_url=(
            "https://noaa-eri-pds.s3.amazonaws.com/2024_Hurricane_Milton/20241011d_RGB"
        ),
    ),
}

#: NAD83 / UTM 17N — the projected CRS for west-central Florida (matches the
#: production pipeline's ``target_crs``).
DEFAULT_TARGET_CRS = "EPSG:26917"


# --------------------------------------------------------------------------- #
# The demo class
# --------------------------------------------------------------------------- #

@dataclass
class RegionalDebrisDemo:
    center_lat: float = 26.456753          # Estero Island, FL
    center_lon: float = -81.959124
    n_cells: int = 10                      # 10 x 10 block ...
    cell_size_m: float = 50.0              # ... of 50 m cells
    target_pixels: int = 256               # output px per cell (model input size)
    target_crs: str = DEFAULT_TARGET_CRS
    source: ERISource = field(default_factory=lambda: ESTERO_IAN_ERI)
    workdir: str = "experiments/estero_regional"
    start_id: int = 1

    # --- internal caches (not constructor args) ---
    _grid: Optional[gpd.GeoDataFrame] = field(default=None, init=False, repr=False)
    _tiles: Optional[gpd.GeoDataFrame] = field(default=None, init=False, repr=False)
    _src_cache: dict = field(default_factory=dict, init=False, repr=False)
    _imagery_paths: dict = field(default_factory=dict, init=False, repr=False)
    _coverage: dict = field(default_factory=dict, init=False, repr=False)
    _masks: dict = field(default_factory=dict, init=False, repr=False)

    # ---- directory layout -------------------------------------------------- #
    @property
    def root(self) -> Path:
        p = Path(self.workdir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cache_dir(self) -> Path:
        p = self.root / "cache"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def imagery_dir(self) -> Path:
        p = self.root / "imagery"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def mask_dir(self) -> Path:
        p = self.root / "masks"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---- 1. grid ----------------------------------------------------------- #
    @property
    def center_utm(self) -> Tuple[float, float]:
        tf = Transformer.from_crs("EPSG:4326", self.target_crs, always_xy=True)
        return tf.transform(self.center_lon, self.center_lat)

    def build_grid(self, force: bool = False) -> gpd.GeoDataFrame:
        """Return a GeoDataFrame of grid polygons (columns grid_id, row, col,
        geometry) in ``target_crs``, centered on (center_lat, center_lon).

        Row 0 is the northernmost row; ids run row-major (N->S, W->E), matching
        the production ``generate_grids`` ordering.
        """
        if self._grid is not None and not force:
            return self._grid

        cx, cy = self.center_utm
        size = self.cell_size_m
        half = (self.n_cells * size) / 2.0
        x_min = cx - half
        y_max = cy + half

        records = []
        gid = self.start_id
        for r in range(self.n_cells):                 # 0 = north
            y_hi = y_max - r * size
            y_lo = y_hi - size
            for c in range(self.n_cells):             # 0 = west
                x_lo = x_min + c * size
                x_hi = x_lo + size
                records.append(dict(grid_id=gid, row=r, col=c,
                                    geometry=box(x_lo, y_lo, x_hi, y_hi)))
                gid += 1
        self._grid = gpd.GeoDataFrame(records, geometry="geometry", crs=self.target_crs)
        return self._grid

    def grid_wgs84(self) -> gpd.GeoDataFrame:
        return self.build_grid().to_crs("EPSG:4326")

    def region_bounds(self) -> Tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) of the whole grid block, in target_crs."""
        return tuple(self.build_grid().total_bounds)

    # ---- 2. NOAA ERI ------------------------------------------------------- #
    def _download(self, url: str, dest: Path) -> Path:
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        print(f"[regional] downloading {url}")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        return dest

    def load_tile_index(self, force: bool = False) -> gpd.GeoDataFrame:
        """Download + read the NOAA ERI tile index, reprojected to target_crs,
        with a ``tile_url`` column giving each tile's full COG URL."""
        if self._tiles is not None and not force:
            return self._tiles

        src = self.source
        archive_name = src.tile_index_url.rsplit("/", 1)[-1]
        archive = self._download(src.tile_index_url, self.cache_dir / archive_name)

        extract_dir = self.cache_dir / archive_name.rsplit(".", 1)[0]
        extract_dir.mkdir(parents=True, exist_ok=True)
        if not any(extract_dir.glob("*.shp")):
            if archive.suffix.lower() == ".zip":
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(extract_dir)
            else:  # .tar
                with tarfile.open(archive) as tf:
                    tf.extractall(extract_dir)

        shp = next(iter(extract_dir.rglob("*.shp")), None)
        if shp is None:
            raise FileNotFoundError(f"No .shp inside {archive}")
        tiles = gpd.read_file(shp)
        if tiles.crs is None:
            tiles.set_crs(src.crs_hint, inplace=True)

        tiles["tile_url"] = self._resolve_tile_urls(tiles)
        tiles = tiles.to_crs(self.target_crs)
        self._tiles = tiles
        print(f"[regional] tile index '{src.name}': {len(tiles)} tiles "
              f"({shp.name})")
        return tiles

    def _resolve_tile_urls(self, tiles: gpd.GeoDataFrame) -> List[str]:
        """Find the column holding tile filenames and build absolute COG URLs."""
        col = self.source.url_column
        if col is None:
            preferred = ["location", "url", "filename", "name", "tile", "path",
                         "Location", "URL", "FILENAME", "NAME", "TILE", "PATH"]
            cand = [c for c in preferred if c in tiles.columns]
            cand += [c for c in tiles.columns if c not in cand and c != "geometry"]
            for c in cand:
                vals = tiles[c].astype(str)
                if (vals.str.lower().str.contains(r"\.tif").mean()) > 0.5:
                    col = c
                    break
        if col is None:
            raise ValueError(
                f"Could not find a tile-filename column in {list(tiles.columns)}; "
                "set ERISource.url_column explicitly.")

        base = self.source.tif_base_url.rstrip("/")
        urls = []
        for v in tiles[col].astype(str):
            if v.startswith(("http://", "https://", "/vsicurl/")):
                urls.append(v)
            else:
                urls.append(f"{base}/{os.path.basename(v)}")
        return urls

    def _open_src(self, url: str):
        """Open (and cache) a remote NOAA ERI COG via GDAL's /vsicurl/."""
        if url not in self._src_cache:
            vsi = url if url.startswith("/vsicurl/") else f"/vsicurl/{url}"
            self._src_cache[url] = rasterio.open(vsi)
        return self._src_cache[url]

    def _tiles_for_bounds(self, bounds) -> gpd.GeoDataFrame:
        tiles = self.load_tile_index()
        return tiles[tiles.intersects(box(*bounds))]

    def _fetch_cell(self, geom, grid_id: int, overwrite: bool) -> Tuple[Path, float]:
        out_path = self.imagery_dir / f"grid-{grid_id:06d}-imagery.tif"
        bounds = geom.bounds
        px = self.target_pixels
        dst_transform = transform_from_bounds(*bounds, px, px)

        if out_path.exists() and not overwrite:
            with rasterio.open(out_path) as ds:
                cov = float((ds.read().sum(axis=0) > 0).mean())
            return out_path, cov

        dst = np.zeros((3, px, px), dtype=np.uint8)
        filled = np.zeros((px, px), dtype=bool)
        for url in self._tiles_for_bounds(bounds)["tile_url"]:
            try:
                src = self._open_src(url)
                # Warp directly from the source COG (EPSG:4326) into the cell's
                # UTM grid at target_pixels; GDAL reads only the needed region.
                tmp = np.zeros((3, px, px), dtype=np.uint8)
                reproject(
                    source=rasterio.band(src, (1, 2, 3)),
                    destination=tmp,
                    dst_transform=dst_transform,
                    dst_crs=self.target_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=0,
                )
            except Exception as exc:  # noqa: BLE001 — skip unreadable tile
                print(f"[regional]   warn: {url} -> {exc}")
                continue
            valid = (tmp.sum(axis=0) > 0) & (~filled)
            for b in range(3):
                dst[b][valid] = tmp[b][valid]
            filled |= valid
            if filled.all():
                break

        coverage = float(filled.mean())
        profile = dict(driver="GTiff", height=px, width=px, count=3,
                       dtype="uint8", crs=self.target_crs,
                       transform=dst_transform, compress="lzw", tiled=True)
        with rasterio.open(out_path, "w", **profile) as d:
            d.write(dst)
        return out_path, coverage

    def fetch_imagery(self, overwrite: bool = False, progress: bool = True
                      ) -> Dict[int, Path]:
        """Fetch a NOAA ERI RGB tile for every grid cell. Returns {grid_id: path}."""
        grid = self.build_grid()
        self.load_tile_index()
        rows = list(grid.itertuples(index=False))
        if progress:
            try:
                from tqdm.auto import tqdm
                rows = tqdm(rows, desc="NOAA ERI tiles")
            except Exception:
                pass
        for rec in rows:
            path, cov = self._fetch_cell(rec.geometry, rec.grid_id, overwrite)
            self._imagery_paths[rec.grid_id] = path
            self._coverage[rec.grid_id] = cov
        n_cov = sum(c > 0.5 for c in self._coverage.values())
        print(f"[regional] fetched {len(self._imagery_paths)} cells; "
              f"{n_cov} with >50% imagery coverage.")
        self._close_sources()
        return self._imagery_paths

    def _close_sources(self):
        for src in self._src_cache.values():
            try:
                src.close()
            except Exception:
                pass
        self._src_cache.clear()

    # ---- 3. inference with the OFFICIAL model (lazy torch) ---------------- #
    def run_inference(self, repo_root=None, weights=None, model=None,
                      device=None, write_masks: bool = True, progress: bool = True
                      ) -> Dict[int, np.ndarray]:
        """Run the **official** CLIPSeg-debris model on every fetched cell and
        write georeferenced 3-class mask GeoTIFFs. Returns {grid_id: mask}.

        Pass either a preloaded official ``model`` (from
        ``clipseg_official.load_model``) or both ``repo_root`` and ``weights``.
        """
        from . import clipseg_official as co
        from . import debris_common as dc

        if not self._imagery_paths:
            self.fetch_imagery()
        if model is None:
            if repo_root is None or weights is None:
                raise ValueError("Provide a preloaded official `model`, or both "
                                 "`repo_root` (resolved CLIPSeg-debris repo) and `weights`.")
            model = co.load_model(repo_root, weights, device=device)

        ids = sorted(self._imagery_paths)
        rgbs, profiles = [], []
        for gid in ids:
            with rasterio.open(self._imagery_paths[gid]) as src:
                rgbs.append(src.read([1, 2, 3]).transpose(1, 2, 0))
                profiles.append(src.profile.copy())

        masks = co.predict_images(model, rgbs, device=device, progress=progress)
        for gid, mask, prof in zip(ids, masks, profiles):
            if write_masks:
                dc.save_mask_geotiff(mask, prof,
                                     self.mask_dir / f"grid-{gid:06d}-mask.tif")
            self._masks[gid] = mask
        return self._masks

    def export_tiles_png(self, out_dir=None) -> Path:
        """Write each imagery tile as ``post-rgb-<gid>_50m.png`` (the filename
        pattern the official ``DebrisPredictionDataset`` expects). Useful for the
        official ``src.eval.evaluate`` path or for staging an HPC job."""
        from PIL import Image
        out_dir = Path(out_dir) if out_dir else (self.root / "tiles_png")
        out_dir.mkdir(parents=True, exist_ok=True)
        for gid, p in sorted(self._imagery_paths.items()):
            with rasterio.open(p) as src:
                rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
            Image.fromarray(rgb.astype(np.uint8)).save(
                out_dir / f"post-rgb-{gid:06d}_50m.png")
        return out_dir

    def load_masks_from_disk(self, mask_dir=None) -> Dict[int, np.ndarray]:
        """Populate ``self._masks`` from mask GeoTIFFs on disk (e.g. results
        downloaded from an HPC job). Returns {grid_id: mask}."""
        mdir = Path(mask_dir) if mask_dir is not None else self.mask_dir
        for mp in sorted(mdir.glob("grid-*-mask.tif")):
            gid = int(mp.stem.split("-")[1])
            with rasterio.open(mp) as src:
                self._masks[gid] = src.read(1)
        print(f"[regional] loaded {len(self._masks)} masks from {mdir}")
        return self._masks

    # ---- 4. mosaic + stats ------------------------------------------------- #
    def mosaic(self, kind: str = "imagery") -> Path:
        """Merge the per-cell tiles into one regional GeoTIFF. kind: imagery|mask."""
        from rasterio.merge import merge

        if kind == "imagery":
            files = sorted(self.imagery_dir.glob("grid-*-imagery.tif"))
            out = self.root / "mosaic_imagery.tif"
            nodata = 0
        elif kind == "mask":
            files = sorted(self.mask_dir.glob("grid-*-mask.tif"))
            out = self.root / "mosaic_mask.tif"
            nodata = 255
        else:
            raise ValueError("kind must be 'imagery' or 'mask'")
        if not files:
            raise FileNotFoundError(f"No {kind} tiles found — run the earlier step first.")

        datasets = [rasterio.open(f) for f in files]
        try:
            arr, transform = merge(datasets, method="first", nodata=nodata)
            profile = datasets[0].profile.copy()
        finally:
            for d in datasets:
                d.close()
        profile.update(height=arr.shape[1], width=arr.shape[2],
                       transform=transform, count=arr.shape[0])
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(arr)
        print(f"[regional] wrote {kind} mosaic -> {out}")
        return out

    def summary_table(self) -> pd.DataFrame:
        """Per-cell debris statistics (areas in m^2). Requires masks in memory."""
        from . import debris_common as dc

        if not self._masks:
            raise RuntimeError("No masks yet — call run_inference() first.")
        cell_area = self.cell_size_m ** 2
        grid = self.build_grid().set_index("grid_id")
        rows = []
        for gid, mask in self._masks.items():
            fr = dc.class_fractions(mask)
            low = fr["low-density debris"]
            high = fr["high-density debris"]
            rows.append(dict(
                grid_id=gid,
                row=int(grid.loc[gid, "row"]), col=int(grid.loc[gid, "col"]),
                coverage=self._coverage.get(gid, np.nan),
                frac_no_debris=fr["no debris"],
                frac_low=low, frac_high=high,
                debris_area_m2=(low + high) * cell_area,
                high_debris_area_m2=high * cell_area,
            ))
        df = pd.DataFrame(rows).sort_values("grid_id").reset_index(drop=True)
        return df

    # ---- misc -------------------------------------------------------------- #
    def describe(self) -> str:
        b = self.region_bounds()
        return (
            f"RegionalDebrisDemo(center=({self.center_lat}, {self.center_lon}), "
            f"{self.n_cells}x{self.n_cells} cells of {self.cell_size_m:.0f} m "
            f"-> {self.n_cells*self.cell_size_m:.0f} m square)\n"
            f"  CRS={self.target_crs}  bounds={tuple(round(v,1) for v in b)}\n"
            f"  ERI source={self.source.name}\n  workdir={self.root}"
        )
