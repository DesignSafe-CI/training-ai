"""
utils — helper package for the 2026 NHERI Computational Symposium
CLIPSeg-debris on DesignSafe demo.

Submodules (import what you need; heavy deps load lazily):
    debris_common    3-class definition, normalization, image/mask helpers (no torch)
    clipseg_official obtain + load the OFFICIAL CLIPSeg-debris repo and run inference
    regional         RegionalDebrisDemo: grid -> NOAA ERI -> tiles -> mosaic
    finetune_data    prepare/inspect/verify the published dataset for fine-tuning
    dapi_helpers     DesignSafe API (auth, files, apps/systems, HPC inference + Vista training)
    viz              matplotlib visualization helpers

The model itself is NOT vendored here — `clipseg_official` clones/loads the
official repository (DesignSafe PRJ-6225 / github.com/Way-Yuhao/CLIPSeg-debris)
and uses its `CLIPDensePredT` and inference recipe.
"""

__all__ = ["debris_common", "clipseg_official", "regional", "finetune_data",
           "dapi_helpers", "viz"]
__version__ = "2.1.0"
