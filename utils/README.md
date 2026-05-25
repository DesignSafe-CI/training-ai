# `utils/` — helper modules

These keep the notebook short: the notebook holds the *story* and the headline
calls (including the official `src.eval.evaluate` usage); the long/heavy functions
live here. **The model is not vendored** — `clipseg_official` loads the official
repository and uses its model + inference recipe.

| Module | What it does | Heavy deps |
|---|---|---|
| `debris_common.py` | 3-class definition, official normalization, colorize/overlay/stats, GeoTIFF I/O. | numpy (cv2/rasterio lazy) |
| `clipseg_official.py` | Obtain the official repo (PRJ-6225 / GitHub), load `CLIPDensePredT`, run the official inference recipe. | torch, openai-clip (lazy) |
| `regional.py` | `RegionalDebrisDemo`: build grid → pull NOAA ERI → 256×256 tiles → official inference → mosaic. | rasterio, geopandas, shapely, pyproj |
| `dapi_helpers.py` | DesignSafe API: auth, files up/down, apps/systems, HPC inference job, **Vista training job**, wandb embed. | dapi (lazy) |
| `viz.py` | Prediction panels, basemap grid, mosaic overlay, choropleth, bars. | matplotlib, contextily |

## Official usage (no vendoring)

`clipseg_official.resolve_repo()` obtains the **official CLIPSeg-debris** software
and puts it on `sys.path`:

1. an existing checkout, else
2. the DesignSafe published software **PRJ-6225** (DOI `10.17603/ds2-yt43-hw55`), else
3. `git clone` of <https://github.com/Way-Yuhao/CLIPSeg-debris> (`v1.0.1`).

`load_model()` then builds `src.models.clipseg.clipseg.CLIPDensePredT` with the
exact `configs/model/clip_seg.yaml` hyper-parameters and loads the trained
weights; `predict_image()` reproduces `CLIPSegLitModule.predict_step` (prompt with
`"a photo of {density}"` for the 3 densities, stack, forward, `argmax`).

The published-dataset showcase in the notebook calls the repo's own
`src.eval.evaluate` (Hydra) directly — the standard official entry point.

## The 3 debris classes

| class | density prompt | color |
|---|---|---|
| 0 | `a photo of no debris` | black |
| 1 | `a photo of debris at low density` | amber |
| 2 | `a photo of debris at high density` | red |

Weights (`clipseg_debris.safetensors` / `clipseg_debris_weight.ckpt`) are on
Hugging Face (`YuhaoL/CLIPSeg-debris`); see `../weights/README.md`.
