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

# --- Compatibility shim for the regular DesignSafe Jupyter session ----------- #
# Its Docker container starts the kernel under a numeric UID that is NOT listed
# in /etc/passwd. Recent PyTorch (>=2.6) runs `getpass.getuser()` at the module
# load time of `torch._dynamo` (to pick a default Inductor cache dir), and that
# call falls through to `pwd.getpwuid(os.getuid())` → `KeyError: getpwuid()`,
# which aborts `import torch._dynamo` and therefore `import torchvision`.
# `getpass.getuser()` consults LOGNAME/USER/LNAME/USERNAME *before* the pwd
# database, so seeding `USER` here makes the lookup short-circuit and never hit
# `pwd`. No-op on systems where USER is already set (Vista, regular shells).
import os
os.environ.setdefault("USER", "jupyter")

__all__ = ["debris_common", "clipseg_official", "regional", "finetune_data",
           "dapi_helpers", "viz"]
__version__ = "2.1.0"
