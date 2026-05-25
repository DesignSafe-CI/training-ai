"""
finetune_data.py
================
Prepare the **published** debris dataset (DesignSafe PRJ-6029) for the OFFICIAL
CLIPSeg-debris fine-tuning datamodule (`DebrisOneHotDataset`), and provide
diagnostics so problems are easy to see in the notebook output.

Published PRJ-6029 layout (the README the user provided):
    data/
      original/          all crops            (post-rgb-<id>_merged_50m.png)
      annotations/       508 consensus masks  (single-channel: 0=no,1=low,2=high)
      annotations_vis/   colored overlays     (ignored for training)
      prompts_vis/{no,low,high}/   engineered visual prompts

What the official `DebrisOneHotDataset` expects (src/data/components/debris_one_hot.py):
    <dataset_dir>/
      original/             ONLY debris-positive images (one per annotation)
      segmentation_merged/  3-channel one-hot masks, BGR = (no, low, high) in {0,255}
      vis_prompts/{no,low,high}/   visual prompts

This module converts the former into the latter (single-channel -> 3-channel
one-hot, folder rename, debris-positive filtering, consistent naming), so the
official `src/train.py experiment=clipseg_finetune data=debris_one_hot` can run.
Only numpy/opencv are needed (no torch).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _id(name) -> Optional[str]:
    """Numeric image id (zero-padded to 6) from a filename — robust to
    `post-rgb-000363_merged_50m.png`, `000363.png`, `363.png`."""
    digits = re.findall(r"\d+", Path(name).stem)
    if not digits:
        return None
    return f"{int(max(digits, key=len)):06d}"   # longest digit run wins (ignores '50m')


def _find_subdir(parent: Path, names) -> Optional[Path]:
    """First existing subdir whose name matches any of `names` (case-insensitive,
    substring ok). Also searches one level down (zips often nest a folder)."""
    parent = Path(parent)
    if not parent.exists():
        return None
    cands = [parent] + [p for p in parent.iterdir() if p.is_dir()]
    for base in cands:
        for child in sorted(base.iterdir()) if base.is_dir() else []:
            if child.is_dir() and any(n.lower() == child.name.lower() for n in names):
                return child
        for child in sorted(base.iterdir()) if base.is_dir() else []:
            if child.is_dir() and any(n.lower() in child.name.lower() for n in names):
                return child
    return None


def _pngs(d: Optional[Path]) -> List[Path]:
    if d is None or not d.exists():
        return []
    return [f for f in sorted(d.glob("*.png")) if not f.name.startswith("._")]


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #

def inspect_published_dataset(published_dir) -> Dict:
    """Print what's in the published dataset (folders, counts, sample names, and
    the annotation encoding) so we can map it correctly. Returns a summary dict."""
    import cv2

    published_dir = Path(published_dir)
    print(f"[finetune-data] inspecting: {published_dir}")
    print("  exists:", published_dir.exists())
    if published_dir.exists():
        print("  entries:", [p.name for p in sorted(published_dir.iterdir())][:20])

    orig = _find_subdir(published_dir, ["original"])
    anns = _find_subdir(published_dir, ["annotations", "annotation", "segmentation", "masks"])
    prom = _find_subdir(published_dir, ["prompts_vis", "vis_prompts", "prompts"])
    info = {"original": orig, "annotations": anns, "prompts": prom}

    for label, d in info.items():
        if d and label == "prompts":     # prompts live in no/low/high subfolders
            files = [f for f in sorted(d.rglob("*.png")) if not f.name.startswith("._")]
        else:
            files = _pngs(d)
        sample = files[0].name if files else "-"
        print(f"  {label:12s}: {str(d):55s}  {len(files)} png  e.g. {sample}")

    # annotation encoding
    ann_files = _pngs(anns)
    if ann_files:
        a = cv2.imread(str(ann_files[0]), cv2.IMREAD_UNCHANGED)
        if a is not None:
            uniq = np.unique(a if a.ndim == 2 else a[..., 0])
            print(f"  annotation[0] shape={a.shape} dtype={a.dtype} "
                  f"unique(first ch)={uniq[:8].tolist()}{'...' if len(uniq) > 8 else ''}")
    # prompts subfolders
    if prom:
        subs = {s.name: len(_pngs(s)) for s in sorted(prom.iterdir()) if s.is_dir()}
        print("  prompt subfolders:", subs)
    info["n_original"] = len(_pngs(orig))
    info["n_annotations"] = len(ann_files)
    return info


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #

def _to_one_hot_bgr(ann_path: Path, size=None) -> np.ndarray:
    """Single-channel class mask (0=no,1=low,2=high) -> 3-channel uint8 image with
    BGR = (no, low, high) binary masks, matching what DebrisOneHotDataset reads
    (cv2 IMREAD_COLOR; channel idx == density idx: no=0, low=1, high=2)."""
    import cv2

    a = cv2.imread(str(ann_path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise ValueError(f"could not read annotation {ann_path}")
    if a.ndim == 3:                      # collapse to a class-index plane
        a = a[..., 0]
    a = a.astype(np.int32)
    # If values look scaled (e.g. 0/127/255), remap sorted unique -> 0,1,2.
    uniq = sorted(int(v) for v in np.unique(a))
    if uniq and max(uniq) > 2:
        remap = {v: i for i, v in enumerate(uniq[:3])}
        a = np.vectorize(lambda v: remap.get(int(v), 0))(a).astype(np.int32)
    h, w = a.shape[:2]
    out = np.zeros((h, w, 3), np.uint8)
    out[..., 0] = (a == 0) * 255         # B = no debris
    out[..., 1] = (a == 1) * 255         # G = low density
    out[..., 2] = (a == 2) * 255         # R = high density
    if size:
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_NEAREST)
    return out


def prepare_finetune_dataset(published_dir, out_dir, limit: Optional[int] = None,
                             prompt_limit: Optional[int] = 80, verbose: bool = True) -> Dict:
    """Convert published PRJ-6029 -> the official DebrisOneHotDataset layout.

    Produces ``out_dir/{original, segmentation_merged, vis_prompts/{no,low,high}}``
    using only debris-positive images (those that have an annotation). ``limit``
    subsamples the debris-positive set for a quick demo (keeps it small enough to
    bundle into the HPC job). Returns a summary dict.
    """
    import cv2

    published_dir, out = Path(published_dir), Path(out_dir)
    orig_d = _find_subdir(published_dir, ["original"])
    ann_d = _find_subdir(published_dir, ["annotations", "annotation", "segmentation", "masks"])
    prom_d = _find_subdir(published_dir, ["prompts_vis", "vis_prompts", "prompts"])
    if not orig_d or not ann_d:
        raise FileNotFoundError(
            f"Could not locate original/ and annotations/ under {published_dir}. "
            "Run inspect_published_dataset() to see the layout.")

    orig = {_id(f): f for f in _pngs(orig_d) if _id(f)}
    anns = {_id(f): f for f in _pngs(ann_d) if _id(f)}
    ids = sorted(set(orig) & set(anns))           # debris-positive (have a mask)
    if limit:
        ids = ids[:limit]
    if not ids:
        raise RuntimeError("No image/annotation id matches — check naming via "
                           "inspect_published_dataset().")

    (out / "original").mkdir(parents=True, exist_ok=True)
    (out / "segmentation_merged").mkdir(parents=True, exist_ok=True)
    for i in ids:
        name = f"post-rgb-{i}_merged_50m.png"     # consistent, hyphen-free-friendly id at split('-')[2]
        shutil.copy(orig[i], out / "original" / name)
        cv2.imwrite(str(out / "segmentation_merged" / name), _to_one_hot_bgr(anns[i]))

    n_prompts = {}
    for d in ("no", "low", "high"):
        srcp = _find_subdir(prom_d, [d]) if prom_d else None
        dstp = out / "vis_prompts" / d
        dstp.mkdir(parents=True, exist_ok=True)
        files = _pngs(srcp)
        if prompt_limit:
            files = files[:prompt_limit]      # cap so the job bundle stays small
        for f in files:
            shutil.copy(f, dstp / f.name)
        n_prompts[d] = len(files)

    summary = {"dataset_dir": str(out), "n_debris_positive": len(ids),
               "n_prompts": n_prompts}
    if verbose:
        print(f"[finetune-data] prepared -> {out}")
        print(f"  original/ + segmentation_merged/: {len(ids)} debris-positive images")
        print(f"  vis_prompts/: {n_prompts}")
        if not all(n_prompts.values()):
            print("  ⚠️ some prompt folders are empty — check prompt naming (no/low/high).")
    return summary


# --------------------------------------------------------------------------- #
# Obtain the published dataset on THIS session (Vista) — mount / corral / DAPI
# --------------------------------------------------------------------------- #

def _attr(o, k):
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


def corral_locate_published(verbose: bool = True) -> Optional[Path]:
    """If TACC **corral** is mounted on this node (common on Vista), find the
    PRJ-6029 ``data`` dir directly (fast, no DAPI). Returns it or None."""
    roots = ["/corral/projects/NHERI/published", "/corral-repl/projects/NHERI/published",
             "/corral/main/projects/NHERI/published",
             "/corral-repl/main/projects/NHERI/published"]
    for r in roots:
        rp = Path(r)
        if not rp.exists():
            continue
        for proj in sorted(rp.iterdir()):
            if not proj.is_dir() or "6029" not in proj.name:
                continue
            for cand in [proj] + [p for p in proj.iterdir() if p.is_dir()]:
                if (cand / "original").exists() and (cand / "annotations").exists():
                    if verbose:
                        print(f"  [corral] found data dir: {cand}")
                    return cand
                for sub in (cand.iterdir() if cand.is_dir() else []):
                    if sub.is_dir() and (sub / "original").exists() and (sub / "annotations").exists():
                        if verbose:
                            print(f"  [corral] found data dir: {sub}")
                        return sub
    return None


def dapi_probe_published(ds, prj="PRJ-6029", hint=None, verbose=True):
    """Diagnostic: report which forms of the published path ``designsafe.storage.published``
    accepts (``/PRJ-XXXX``, versioned ``/PRJ-XXXXv2``, and the root). If every form
    errors, the published system is not reachable from this host (→ stage once)."""
    from .dapi_helpers import published_probe
    return published_probe(ds, prj, verbose=verbose)


def fetch_published_dapi(ds, prj, dest_src_dir, limit=120, prompt_limit=80,
                         verbose=True) -> Optional[Path]:
    """Download the NEEDED subset of the published dataset to this session via the files
    API on ``designsafe.storage.published``. Discovers the real project base (handles
    versioned publications, e.g. ``PRJ-6029v2``) and the ``original``/``annotations``/
    ``prompts`` dirs by BFS, then paginates + downloads. Returns the published-format
    dir, or None on failure."""
    from .dapi_helpers import (quiet, published_base_uri, published_find_dirs,
                               _ds_list, _ds_download)

    base = published_base_uri(ds, prj, verbose=verbose)
    if not base:
        return None
    dirs = published_find_dirs(ds, base, {
        "original": {"original"},
        "annotations": {"annotations", "annotation"},
        "prompts": {"prompts_vis", "vis_prompts", "prompts"},
    })
    if verbose:
        sub = f"/{prj}"  # for short display
        print("  [dapi] located:",
              {k: (v.split("/published/")[-1] if v else None) for k, v in dirs.items()})
    if not dirs["original"] or not dirs["annotations"]:
        return None

    def _png_map(dir_uri):
        """Paginate a dir; return {id: (name, url)} for its .png files."""
        out, off = {}, 0
        while off < 8000:
            ents, _ = _ds_list(ds, dir_uri, limit=100, offset=off)
            if not ents:
                break
            for e in ents:
                nm = (_attr(e, "name") or "").split("/")[-1]
                if not nm.endswith(".png") or nm.startswith("._"):
                    continue
                iid = _id(nm)
                if iid and iid not in out:
                    out[iid] = (nm, _attr(e, "url") or f"{dir_uri}/{nm}")
            if len(ents) < 100:
                break
            off += 100
        return out

    orig = _png_map(dirs["original"])
    anns = _png_map(dirs["annotations"])
    ids = sorted(set(orig) & set(anns))
    if limit:
        ids = ids[:limit]
    if not ids:
        return None

    pub = Path(dest_src_dir)
    (pub / "original").mkdir(parents=True, exist_ok=True)
    (pub / "annotations").mkdir(parents=True, exist_ok=True)
    got = 0
    for i in ids:
        for sub_name, (nm, url) in (("original", orig[i]), ("annotations", anns[i])):
            out = pub / sub_name / nm
            try:
                _ds_download(ds, url, out)
                got += out.exists() and out.stat().st_size > 0
            except Exception:
                pass

    if dirs["prompts"]:
        pdirs = published_find_dirs(ds, dirs["prompts"],
                                    {"no": {"no"}, "low": {"low"}, "high": {"high"}}, max_depth=1)
        for d in ("no", "low", "high"):
            duri = pdirs.get(d)
            if not duri:
                continue
            (pub / "prompts_vis" / d).mkdir(parents=True, exist_ok=True)
            ents, _ = _ds_list(ds, duri, limit=max(1, prompt_limit or 100))
            for e in (ents[:prompt_limit] if prompt_limit else ents):
                nm = (_attr(e, "name") or "").split("/")[-1]
                if not nm.endswith(".png"):
                    continue
                try:
                    _ds_download(ds, _attr(e, "url") or f"{duri}/{nm}",
                                 pub / "prompts_vis" / d / nm)
                except Exception:
                    pass
    if verbose:
        print(f"  [dapi] downloaded {len(ids)} image+mask pairs ({got} files) -> {pub}")
    return pub if any((pub / "original").glob("*.png")) else None


def obtain_and_prepare(ds, out_dir, prj="PRJ-6029", mount_dir=None,
                       limit: Optional[int] = 120, prompt_limit: Optional[int] = 80,
                       verbose: bool = True) -> Optional[Path]:
    """Get the published dataset on THIS session and convert it to the trainer
    layout. Tries, in order: (1) NHERI-Published **mount**, (2) TACC **corral**
    filesystem, (3) **DAPI publications** download. Returns the prepared dir or None."""
    out = Path(out_dir)

    if mount_dir and Path(mount_dir).exists() and (Path(mount_dir) / "original").exists():
        print("[finetune-data] source = NHERI-Published mount")
        inspect_published_dataset(mount_dir)
        prepare_finetune_dataset(mount_dir, out, limit=limit, prompt_limit=prompt_limit)
        return out

    cor = corral_locate_published(verbose)
    if cor:
        print("[finetune-data] source = TACC corral filesystem")
        inspect_published_dataset(cor)
        prepare_finetune_dataset(cor, out, limit=limit, prompt_limit=prompt_limit)
        return out

    if ds is not None:
        print("[finetune-data] source = DAPI publications (downloading subset to this session)")
        pub = fetch_published_dapi(ds, prj, out.parent / (out.name + "_src"),
                                   limit=limit, prompt_limit=prompt_limit, verbose=verbose)
        if pub:
            inspect_published_dataset(pub)
            prepare_finetune_dataset(pub, out, limit=limit, prompt_limit=prompt_limit)
            return out
        # DAPI didn't find the data — show exactly what the published system accepts.
        print("[finetune-data] DAPI publications navigation found nothing; probing path forms:")
        try:
            dapi_probe_published(ds, prj)
        except Exception as exc:
            print("   probe failed:", exc)

    print("\n[finetune-data] Could not obtain the published dataset on this session. Options")
    print("  (all one-time; afterwards Vista's 7a auto-detects the MyData copy):")
    print("  A) DesignSafe web -> Data Depot -> Published -> PRJ-6029: select the `data`")
    print("     folder -> Copy -> into My Data/clipseg_finetune_src. Then set:")
    print("       fd.prepare_finetune_dataset('<MyData>/clipseg_finetune_src',")
    print(f"                                   r'{out}')")
    print("  B) Run THIS cell once on the regular JupyterHub (the NHERI-Published mount")
    print("     works there); it writes the prepared set to MyData, shared with Vista.")
    return None


# --------------------------------------------------------------------------- #
# verify (diagnostic): can the OFFICIAL dataset load a sample?
# --------------------------------------------------------------------------- #

def verify_finetune_dataset(repo_root, dataset_dir, n: int = 2) -> bool:
    """Instantiate the official DebrisOneHotDataset on the prepared dir and pull a
    few samples — surfaces any format mismatch with a clear message."""
    import importlib.util
    import sys

    repo_root = Path(repo_root)
    ds_py = repo_root / "src" / "data" / "components" / "debris_one_hot.py"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("debris_one_hot_check", ds_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    DebrisOneHotDataset = mod.DebrisOneHotDataset

    try:
        ds = DebrisOneHotDataset(
            dataset_dir=str(dataset_dir), resize_to=(256, 256),
            text_prompts=["no debris", "debris at low density", "debris at high density"],
            densities=["no", "low", "high"])
        print(f"[finetune-data] DebrisOneHotDataset OK — {len(ds)} samples; "
              f"sample ids {ds.img_ids[:5]}")
        for k in range(min(n, len(ds))):
            data_x, data_y = ds[k]
            print(f"  sample {k}: image {tuple(data_x[0].shape)}, target {tuple(data_y[0].shape)}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[finetune-data] ✗ dataset load failed: {type(exc).__name__}: {exc}")
        return False
