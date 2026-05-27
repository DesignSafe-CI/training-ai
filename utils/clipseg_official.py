"""
clipseg_official.py
===================
Use the **official** CLIPSeg-debris research software for inference — no vendored
model code. This module:

  * locates/obtains the official repo (DesignSafe published software **PRJ-6225**,
    else clones github.com/Way-Yuhao/CLIPSeg-debris) and puts it on ``sys.path``;
  * builds ``CLIPDensePredT`` straight from the repo (exact ``configs/model/clip_seg.yaml``
    hyper-parameters) and loads the trained weights;
  * runs the **same inference recipe as the repo's** ``CLIPSegLitModule.predict_step``:
    prompt each image with ``"a photo of {density}"`` for the 3 densities, stack,
    forward, and ``argmax`` over the prompt axis.

The full Hydra ``src.eval.evaluate`` entry point (for the published-dataset
showcase) is called directly in the notebook, exactly like the original webinar
notebook. This module is the thin, reusable layer the regional pipeline needs to
get **georeferenced class masks** out of the official model.

torch / clip are imported lazily so the geospatial helpers stay importable
without them.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

try:                                    # package import (utils.clipseg_official)
    from .debris_common import PROMPTS, MODEL_INPUT_SIZE, normalize_chw
except ImportError:                     # flat import (HPC job bundle)
    from debris_common import PROMPTS, MODEL_INPUT_SIZE, normalize_chw

# --------------------------------------------------------------------------- #
# Where to get the official software
# --------------------------------------------------------------------------- #

GITHUB_URL = "https://github.com/Way-Yuhao/CLIPSeg-debris.git"
GITHUB_TAG = "v1.0.1"

#: DesignSafe published-software ZIP (PRJ-6225) on the JupyterHub mount.
DESIGNSAFE_PRJ6225_ZIP = Path(
    "/home/jupyter/NHERI-Published/published-data/PRJ-6225/"
    "Project--debris-segmentation-model-using-post-hurricane-aerial-imagery-clipseg-debris/"
    "data/CLIPSeg-debris_v1.0.1.zip"
)

#: Architecture hyper-parameters — copied from configs/model/clip_seg.yaml.
_ARCH_KWARGS = dict(version="ViT-B/16", reduce_dim=64, complex_trans_conv=True,
                    extract_layers=(3, 7, 9), fix_shift=False)


# --------------------------------------------------------------------------- #
# Obtain + register the official repo
# --------------------------------------------------------------------------- #

def resolve_repo(project_root: Union[str, Path], repo_name: str = "CLIPSeg-debris",
                 verbose: bool = True) -> Path:
    """Return the path to the official CLIPSeg-debris repo, obtaining it if needed,
    and add it to ``sys.path``.

    Order (same idea as the original webinar notebook):
      1. an existing checkout at ``project_root/CLIPSeg-debris``;
      2. the DesignSafe published-software ZIP (PRJ-6225), extracted;
      3. ``git clone`` of the GitHub release.
    """
    project_root = Path(project_root)
    repo_root = project_root / repo_name

    if repo_root.exists():
        if verbose:
            print(f"[clipseg] using existing repo: {repo_root}")
    elif DESIGNSAFE_PRJ6225_ZIP.exists():
        if verbose:
            print(f"[clipseg] extracting DesignSafe PRJ-6225 software:\n  {DESIGNSAFE_PRJ6225_ZIP}")
        tmp = project_root / "_clipseg_extract_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        with zipfile.ZipFile(DESIGNSAFE_PRJ6225_ZIP) as zf:
            zf.extractall(tmp)
        top = [p for p in tmp.iterdir() if p.is_dir()]
        shutil.move(str(top[0]), str(repo_root))
        shutil.rmtree(tmp, ignore_errors=True)
        if verbose:
            print(f"[clipseg] installed official repo at {repo_root}")
    else:
        if verbose:
            print(f"[clipseg] cloning {GITHUB_URL} ({GITHUB_TAG}) ...")
        subprocess.check_call(["git", "clone", "--branch", GITHUB_TAG, "--depth", "1",
                               GITHUB_URL, str(repo_root)])

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


#: Runtime deps the official eval/train pipeline needs, as (pip_name, import_name).
#: Curated + loosely pinned on purpose: the repo's own requirements.txt HARD-pins
#: (e.g. ``lightning==2.4.0``, ``packaging==25``) and is unresolvable against newer
#: torch environments such as TACC Vista (CUDA torch 2.6). We install only what's
#: MISSING, so the preinstalled CUDA torch/torchvision are never touched.
_OFFICIAL_RUNTIME_DEPS = (
    ("hydra-core", "hydra"), ("omegaconf", "omegaconf"),
    ("hydra-colorlog", "hydra_colorlog"), ("colorlog", "colorlog"), ("rich", "rich"),
    ("lightning", "lightning"), ("lightning_utilities", "lightning_utilities"),
    ("openai-clip", "clip"), ("general_utils", "general_utils"),
    ("rootutils", "rootutils"), ("natsort", "natsort"),
    ("scikit-learn", "sklearn"), ("wandb", "wandb"), ("torchvision", "torchvision"),
)


# --------------------------------------------------------------------------- #
# CLIP backbone cache.
# Some Jupyter containers (e.g. the regular DesignSafe JupyterHub) ship with a
# read-only ``~/.cache/`` mount point, so OpenAI ``clip``'s default cache root
# (``~/.cache/clip``) raises ``PermissionError`` the first time the ViT-B/16
# backbone is downloaded. The helpers below find a writable directory and wrap
# ``clip.load`` so the model's own internal ``clip.load(...)`` call uses it.
# No-op on Vista and on any normal shell (the default cache path is writable).
# --------------------------------------------------------------------------- #

def _find_writable_clip_cache() -> Path:
    """Return a directory CLIP can use to cache its backbone — the first writable
    option among ``~/.cache/clip``, ``~/MyData/.cache/clip``, ``/tmp/clip_cache``."""
    import tempfile
    candidates = [
        Path.home() / ".cache" / "clip",
        Path.home() / "MyData" / ".cache" / "clip",
        Path(tempfile.gettempdir()) / "clip_cache",
    ]
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            probe = c / ".w"
            probe.write_text("x"); probe.unlink()
            return c
        except (PermissionError, OSError):
            continue
    raise RuntimeError("no writable CLIP cache directory available")


def _patch_clip_cache(verbose: bool = False) -> Optional[Path]:
    """Wrap ``clip.load`` so calls without an explicit ``download_root`` use a
    writable cache. Idempotent. Returns the chosen cache directory, or ``None``
    if ``clip`` isn't importable yet (e.g. before ``install_requirements``)."""
    try:
        import clip
    except ImportError:
        return None
    if getattr(clip.load, "_clip_cache_patched", False):
        return Path(clip.load._clip_cache_dir)
    cache_dir = _find_writable_clip_cache()
    _orig = clip.load

    def _wrapped(name, device="cpu", jit=False, download_root=None):
        return _orig(name, device=device, jit=jit,
                     download_root=download_root or str(cache_dir))

    _wrapped._clip_cache_patched = True
    _wrapped._clip_cache_dir = str(cache_dir)
    clip.load = _wrapped
    if verbose:
        print(f"[clipseg] clip cache -> {cache_dir}")
    return cache_dir


def _patch_torch_load_weights_only(verbose: bool = False) -> bool:
    """Make ``torch.load`` default to ``weights_only=False`` when the caller
    hasn't explicitly set it.

    PyTorch 2.6 flipped the default of ``weights_only`` from ``False`` to
    ``True`` for safety. The official CLIPSeg-debris Lightning ``.ckpt``
    contains ``functools.partial`` (and other non-tensor objects) that the
    safe unpickler refuses to load, so Lightning's
    ``trainer.predict(ckpt_path=...)`` (called from ``src.eval.evaluate``)
    aborts with ``UnpicklingError: Weights only load failed``. This restores
    the pre-2.6 behavior for the published checkpoint, which we trust.
    Idempotent. No-op if torch isn't importable yet."""
    try:
        import torch
    except ImportError:
        return False
    if getattr(torch.load, "_weights_only_patched", False):
        return True
    _orig = torch.load

    def _wrapped(f, *args, **kwargs):
        # Treat "not passed" and "explicit None" the same — Lightning passes
        # weights_only=None up the chain, which torch 2.6 still resolves to True.
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return _orig(f, *args, **kwargs)

    _wrapped._weights_only_patched = True
    torch.load = _wrapped
    if verbose:
        print("[clipseg] torch.load defaults to weights_only=False (PyTorch 2.6+ compat)")
    return True


def install_requirements(repo_root: Union[str, Path], extra: Sequence[str] = (),
                         use_repo_requirements: bool = False, verbose: bool = True) -> None:
    """Install the official model's runtime dependencies (Hydra, Lightning,
    openai-clip, …) into the current environment (``--user``).

    By default installs a **curated, loosely-pinned** set and **only the missing
    packages** — this avoids the repo ``requirements.txt`` hard-pin conflicts and
    leaves a preinstalled CUDA ``torch``/``torchvision`` untouched. Pass
    ``use_repo_requirements=True`` to force ``pip install -r requirements.txt``.
    """
    import importlib, site
    cmd = [sys.executable, "-m", "pip", "install", "--user", "-q", "--no-warn-script-location"]

    if use_repo_requirements:
        req = Path(repo_root) / "requirements.txt"
        pkgs = (["-r", str(req)] if req.exists() else []) + list(extra)
    else:
        missing = []
        for pip_name, import_name in _OFFICIAL_RUNTIME_DEPS:
            try:
                importlib.import_module(import_name)
            except ImportError:
                missing.append(pip_name)
        pkgs = missing + list(extra)

    if not pkgs:
        if verbose:
            print("[clipseg] official runtime deps already present")
        return
    if verbose:
        print("[clipseg] installing official runtime deps:",
              " ".join(p for p in pkgs if not p.startswith("-")))
    subprocess.run(cmd + pkgs, check=False)

    # make freshly --user-installed packages importable without a kernel restart
    us = site.getusersitepackages()
    if us not in sys.path:
        sys.path.append(us)
    importlib.invalidate_caches()

    # OpenAI CLIP defaults to ~/.cache/clip, which is read-only on some DesignSafe
    # JupyterHub containers — pick a writable spot and wrap clip.load (no-op on Vista).
    _patch_clip_cache(verbose=verbose)
    # PyTorch 2.6 made torch.load default to weights_only=True, which rejects the
    # official Lightning .ckpt (contains functools.partial). Restore the pre-2.6
    # behavior for the trusted published checkpoint.
    _patch_torch_load_weights_only(verbose=verbose)


# --------------------------------------------------------------------------- #
# Build + load the official model
# --------------------------------------------------------------------------- #

def get_device(prefer: str = "cuda") -> str:
    import torch
    return "cuda" if (prefer.startswith("cuda") and torch.cuda.is_available()) else "cpu"


def disable_slurm_env() -> list:
    """Stop PyTorch-Lightning from auto-detecting the Jupyter session's SLURM
    allocation. On HPC-Native (Vista) the notebook runs *inside* a SLURM job, so
    Lightning's Trainer raises e.g. ``You set --ntasks=72 ...``. Clearing the
    ``SLURM_*`` vars makes Lightning use a single-process environment — correct for
    in-session single-GPU inference. The GPU/CUDA is unaffected. Returns the vars
    cleared. (No effect off-SLURM.)"""
    import os
    cleared = [k for k in list(os.environ) if k.startswith("SLURM_")]
    for k in cleared:
        os.environ.pop(k, None)
    return cleared


def import_model_class(repo_root: Union[str, Path]):
    """Import the official ``CLIPDensePredT`` from the repo's ``clipseg.py`` **by
    file path** (importlib), bypassing ``src/models/__init__.py`` so we don't pull
    in Lightning/Hydra just to get the model class. This is exactly how the
    production data-generation pipeline loads it."""
    import importlib.util

    repo_root = Path(repo_root)
    clipseg_py = repo_root / "src" / "models" / "clipseg" / "clipseg.py"
    if not clipseg_py.exists():
        raise FileNotFoundError(
            f"official model file not found: {clipseg_py}\n"
            "Call resolve_repo(...) first to obtain the CLIPSeg-debris repo.")
    spec = importlib.util.spec_from_file_location("clipseg_debris_official_model", clipseg_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CLIPDensePredT


def load_model(repo_root: Union[str, Path], checkpoint_path: Union[str, Path],
               device: Optional[str] = None, verbose: bool = True):
    """Instantiate the official ``CLIPDensePredT`` and load trained weights.

    Accepts either the published ``.safetensors`` (portable) or the Lightning
    ``.ckpt`` (loads with the repo on ``sys.path``). The CLIP ViT-B/16 backbone is
    fetched by the model itself via ``clip.load``; only the trained decoder comes
    from the checkpoint, so we load with ``strict=False`` after stripping the
    Lightning ``model.`` prefix.
    """
    import torch

    repo_root = Path(repo_root)
    if not (repo_root / "src").exists():
        raise FileNotFoundError(
            f"{repo_root} is not the CLIPSeg-debris repo (no src/). "
            "Call resolve_repo(...) first.")
    # keep repo on sys.path so a Lightning .ckpt can unpickle its src.* classes
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    _patch_clip_cache()                              # ensure clip.load uses a writable cache
    _patch_torch_load_weights_only()                 # ensure torch.load can load the Lightning .ckpt
    CLIPDensePredT = import_model_class(repo_root)   # official model, no Lightning needed

    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = get_device()
    model = CLIPDensePredT(**_ARCH_KWARGS)

    if checkpoint_path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(str(checkpoint_path))
    else:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    if any(k.startswith("model.") for k in state_dict):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()
                      if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    if verbose:
        n = sum(p.numel() for p in model.parameters())
        print(f"[clipseg] official CLIPDensePredT loaded ({n/1e6:.1f}M params) "
              f"from {checkpoint_path.name}; device={device}")
    return model


# --------------------------------------------------------------------------- #
# Inference — faithful to CLIPSegLitModule.predict_step
# --------------------------------------------------------------------------- #

def predict_image(model, rgb: np.ndarray, device: Optional[str] = None,
                  size: int = MODEL_INPUT_SIZE) -> np.ndarray:
    """Segment one RGB image -> HxW uint8 mask in {0,1,2}, exactly as the repo's
    ``predict_step`` does (3 prompts, stack, forward, argmax over prompts)."""
    import torch
    if device is None:
        device = next(model.parameters()).device.type
    x = torch.from_numpy(normalize_chw(rgb, size)).unsqueeze(0)        # [1,3,H,W]
    stacked = x.repeat(len(PROMPTS), 1, 1, 1).to(device, torch.float32)  # [3,3,H,W]
    with torch.no_grad():
        pred = model(stacked, list(PROMPTS), return_features=True)[0]  # [3,1,H,W]
        mask = torch.argmax(pred, dim=0).squeeze().to(torch.uint8).cpu().numpy()
    return mask


def predict_images(model, rgbs: Sequence[np.ndarray], device: Optional[str] = None,
                   size: int = MODEL_INPUT_SIZE, progress: bool = False) -> List[np.ndarray]:
    """Segment a list of RGB images (one at a time, like the official predict)."""
    it = rgbs
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(rgbs, desc="CLIPSeg-debris")
        except Exception:
            pass
    return [predict_image(model, r, device=device, size=size) for r in it]
