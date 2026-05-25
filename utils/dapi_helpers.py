"""
dapi_helpers.py
===============
Thin convenience wrappers around **dapi** (the DesignSafe API) for the
CLIPSeg-debris demo. Everything degrades gracefully when ``dapi`` is not
installed or you are not on DesignSafe, so the notebook still imports and the
in-session examples still run.

What these helpers cover (all built on the documented dapi surface):
  * authentication                      -> connect()
  * environment detection               -> on_designsafe(), published_mount()
  * path <-> Tapis URI translation      -> to_uri()
  * recursive upload / download         -> upload_dir(), download_dir()
  * GPU system / queue discovery        -> gpu_queues()
  * assembling a self-contained HPC job -> build_job_bundle()
  * generating an inference job request -> generate_inference_job()

See https://designsafe-ci.github.io/dapi/ for the underlying API.
"""

from __future__ import annotations

import contextlib
import io
import shutil
from pathlib import Path
from typing import List, Optional


# --------------------------------------------------------------------------- #
# Internal helpers (clean output + correct job archiving)
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def quiet(enabled: bool = True):
    """Silence dapi's chatty stdout (per-file upload/download lines, the full
    job-request dump, etc.) so notebook output stays clean for a live demo."""
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _username(ds) -> str:
    """Best-effort DesignSafe username (parsed from a MyData URI translation)."""
    try:
        return ds.files.to_uri("/MyData").split("designsafe.storage.default/")[1].split("/")[0]
    except Exception:
        return ""


def _fix_archive(req: dict, ds) -> dict:
    """Point job archiving at the USER's own storage. Some app/dapi defaults point
    the archive dir elsewhere (we saw 'silvia/...'), which makes the job fail to
    archive and denies output listing. Idempotent."""
    user = _username(ds)
    if user:
        req["archiveSystemId"] = "designsafe.storage.default"
        req["archiveSystemDir"] = f"{user}/tapis-jobs-archive/${{JobName}}-${{JobUUID}}"
    return req


def _set_main_program(req: dict, program: str = "bash") -> dict:
    """The designsafe-agnostic-app runs ``<Main Program> <Main Script>``; its
    default Main Program is ``python3``, which fails for a *bash* entry script
    (``set -euo pipefail`` -> SyntaxError). Force it to ``bash``. Idempotent."""
    try:
        for a in req["parameterSet"]["appArgs"]:
            if a.get("name") == "Main Program":
                a["arg"] = program
                return req
        req["parameterSet"]["appArgs"].append({"name": "Main Program", "arg": program})
    except Exception:
        pass
    return req


# --------------------------------------------------------------------------- #
# Auth + environment
# --------------------------------------------------------------------------- #

def connect(verbose: bool = True):
    """Create and return an authenticated ``DSClient``.

    On DesignSafe JupyterHub this is non-interactive (tokens are already in the
    environment). Locally, dapi will prompt for DesignSafe credentials.
    """
    try:
        from dapi import DSClient
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "dapi is not installed. Install it with `pip install dapi` "
            "(pre-installed on DesignSafe JupyterHub)."
        ) from exc
    ds = DSClient()
    if verbose:
        print("[dapi] connected to DesignSafe.")
    return ds


def on_designsafe() -> bool:
    """True if we appear to be on DesignSafe — either the JupyterHub VM
    (``/home/jupyter``) or an HPC-native session like Vista/Stampede3
    (``/data/designsafe``)."""
    return any(Path(p).exists() for p in ("/home/jupyter", "/data/designsafe"))


def published_mount(prj: str, subpath: str = "") -> Optional[Path]:
    """Path to a published dataset on the DesignSafe mount, trying the known
    locations across the JupyterHub VM and the HPC-native (Vista/Stampede3)
    sessions. e.g. ``published_mount("PRJ-6029")``. Returns None if not found —
    in that case access the data with DAPI (``ds.files``) instead."""
    candidates = [
        Path("/home/jupyter/NHERI-Published/published-data") / prj,
        Path("/data/designsafe/published-data") / prj,
        Path("/data/designsafe/NHERI-Published/published-data") / prj,
        Path("/corral-repl/projects/NHERI/published") / prj,
        Path("/corral/projects/NHERI/published") / prj,
    ]
    for base in candidates:
        try:
            if base.exists():
                return base / subpath if subpath else base
        except Exception:
            continue
    return None


def to_uri(ds, path: str, verify_exists: bool = False) -> str:
    """Translate a DesignSafe path (/MyData/..., /NHERI-Published/...,
    /projects/PRJ-XXXX/...) to a Tapis URI."""
    return ds.files.to_uri(path, verify_exists=verify_exists)


def mydata_dir() -> Optional[Path]:
    """Local path to MyData, which maps to designsafe.storage.default and is shared
    across the JupyterHub VM and HPC-Native (Vista) sessions — so files written here
    are visible from both. Returns None if not found."""
    import getpass
    user = getpass.getuser()
    for c in [Path("/data/designsafe/mydata") / user, Path("/home/jupyter/MyData"),
              Path.home() / "MyData"]:
        if c.exists():
            return c
    return None


def _file_attr(f, key):
    """Read an attribute from a Tapis file object or dict (name / url / path)."""
    if isinstance(f, dict):
        return f.get(key)
    return getattr(f, key, None)


PUBLISHED_SYSTEM = "designsafe.storage.published"


def _ds_list(ds, uri, limit: int = 100, offset: int = 0):
    """``ds.files.list`` returning ([], exc) instead of raising."""
    try:
        with quiet():
            return (ds.files.list(uri, limit=limit, offset=offset) or []), None
    except Exception as exc:
        return [], exc


def _parse_tapis_uri(uri: str):
    """``tapis://<system>/<path>`` -> ``(system_id, path)``."""
    rest = str(uri).split("://", 1)[-1]
    sysid, _, path = rest.partition("/")
    return sysid, path


def _ds_download(ds, remote_uri: str, local_path) -> int:
    """Download a Tapis file robustly and return bytes written.

    Uses ``ds.tapis.files.getContents`` directly: dapi 0.5.2's ``files.download``
    calls ``.iter_content()`` on a value that is actually ``bytes`` in this tapipy
    version, so it silently writes 0-byte files. getContents returns the bytes.
    """
    sysid, path = _parse_tapis_uri(remote_uri)
    with quiet():
        data = ds.tapis.files.getContents(systemId=sysid, path=path)
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)
    out = Path(local_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(data)
    return len(data)


def _pick_published_entity(ds, project_uri, verbose: bool = True):
    """A published project dir holds either the data directly (legacy) or one or more
    ``Project--<title>[--V<n>]`` entity folders (newer layout). Return the entity URI
    with the HIGHEST version (the curated/latest publication), or the project_uri
    itself if there are no such entity folders."""
    import re
    ents, _ = _ds_list(ds, project_uri, limit=200)
    cands = []
    for e in ents:
        if _file_attr(e, "type") != "dir":
            continue
        nm = (_file_attr(e, "name") or "").split("/")[-1]
        if nm.lower().startswith("project--"):
            m = re.search(r"--v(\d+)$", nm.lower())
            cands.append(((int(m.group(1)) if m else 1), nm,
                          _file_attr(e, "url") or f"{project_uri}/{nm}"))
    if cands:
        cands.sort()
        if verbose:
            print(f"  [dapi] entity: {cands[-1][1]}")
        return cands[-1][2]
    return project_uri


def published_base_uri(ds, prj, verbose: bool = True):
    """Find the REAL tapis URI base of a published project on
    ``designsafe.storage.published``. Two gotchas dapi's publications API doesn't handle:
    (1) **newer publications live under a ``/published-data/`` prefix** (older ones are
    at ``/PRJ-XXXX`` directly); (2) **versions** — a project dir may hold
    ``Project--...--V2`` entity folders. We try both layouts (+ ``vN`` name variants),
    then descend to the highest-version entity. Last resort: scan the root for the id.
    Returns the entity/base URI, or None."""
    num = "".join(c for c in str(prj) if c.isdigit())
    cands = [f"published-data/{prj}", str(prj),
             f"published-data/{prj}v2", f"{prj}v2",
             f"published-data/{prj}v3", f"{prj}v3"]
    seen = set()
    for name in cands:
        if name in seen:
            continue
        seen.add(name)
        uri = f"tapis://{PUBLISHED_SYSTEM}/{name}"
        ents, _ = _ds_list(ds, uri)
        if ents:
            if verbose:
                print(f"  [dapi] published project: /{name} ({len(ents)} entries)")
            return _pick_published_entity(ds, uri, verbose)
    if verbose:
        print("  [dapi] direct names not found; scanning published root + /published-data ...")
    for root in (f"tapis://{PUBLISHED_SYSTEM}/published-data/", f"tapis://{PUBLISHED_SYSTEM}/"):
        off = 0
        while off < 4000:
            page, err = _ds_list(ds, root, limit=100, offset=off)
            if not page:
                if err and verbose and off == 0:
                    print(f"  [dapi] cannot list {root}: {str(err)[:100]}")
                break
            for e in page:
                nm = (_file_attr(e, "name") or "").split("/")[-1]
                if num and num in nm:
                    uri = f"{root}{nm}"
                    if verbose:
                        print(f"  [dapi] published project (scan): {nm}")
                    return _pick_published_entity(ds, uri, verbose)
            if len(page) < 100:
                break
            off += 100
    return None


def published_find_dirs(ds, base_uri, targets, max_depth: int = 5):
    """BFS a published project from ``base_uri`` using each entry's own ``.url`` (so
    there is no path-prefix guesswork). ``targets`` = {label: set(lowercase names)}.
    Returns {label: dir_uri or None}."""
    from collections import deque
    want = {k: None for k in targets}
    q, seen = deque([(base_uri, 0)]), set()
    while q and not all(want.values()):
        uri, d = q.popleft()
        if d > max_depth or uri in seen:
            continue
        seen.add(uri)
        ents, _ = _ds_list(ds, uri)
        for e in ents:
            if _file_attr(e, "type") != "dir":
                continue
            nm = (_file_attr(e, "name") or "").split("/")[-1]
            low = nm.lower()
            curl = _file_attr(e, "url") or f"{uri}/{nm}"
            for label, names in targets.items():
                if low in names and not want[label]:
                    want[label] = curl
            q.append((curl, d + 1))
    return want


def published_probe(ds, prj, verbose: bool = True):
    """Diagnostic: report which published path forms ``designsafe.storage.published``
    accepts (versioned names + root). Distinguishes a path/version problem (some form
    works) from the system being unreachable on this host (every form errors)."""
    rows = []
    for name in [f"published-data/{prj}", str(prj), f"published-data/{prj}v2", f"{prj}v2"]:
        ents, err = _ds_list(ds, f"tapis://{PUBLISHED_SYSTEM}/{name}")
        rows.append((f"/{name}", len(ents) if not err else None,
                     "" if not err else f"{type(err).__name__}: {str(err)[:80]}"))
    root, rerr = _ds_list(ds, f"tapis://{PUBLISHED_SYSTEM}/")
    rows.append(("/ (root)", len(root) if not rerr else None,
                 "" if not rerr else f"{type(rerr).__name__}: {str(rerr)[:80]}"))
    if verbose:
        print(f"[dapi-probe] {PUBLISHED_SYSTEM}:")
        for p, n, info in rows:
            print(f"   {p:14s} -> {('%d entries' % n) if n is not None else 'ERROR':12s} | {info}")
        if all(n is None for _, n, _ in rows):
            print("   => published system is NOT reachable from this host; stage the dataset")
            print("      once (web Data-Depot copy to MyData, or a JupyterHub run).")
    return rows


def _pub_find_dir(ds, prj, names, max_depth: int = 4):
    """Compat shim: return the tapis URI of the first dir matching ``names``."""
    base = published_base_uri(ds, prj, verbose=False)
    if not base:
        return None
    got = published_find_dirs(ds, base, {"x": {n.lower() for n in names}}, max_depth)
    return got["x"]


def download_published_files(ds, prj, subpaths, filenames, dest_dir, verbose: bool = True) -> int:
    """Download named files from a PUBLISHED project using the **files API** on
    ``designsafe.storage.published``. Discovers the real project base (handles versioned
    publications like ``PRJ-6029v2``, which dapi's publications API can't address) and
    the ``original`` directory, paginating to find the requested files. ``subpaths`` is
    accepted for backward-compatibility but discovery is authoritative. Returns count.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    want = set(filenames)
    base = published_base_uri(ds, prj, verbose=verbose)
    if not base:
        if verbose:
            print(f"  [dapi] could not locate {prj} on {PUBLISHED_SYSTEM}:")
            published_probe(ds, prj)
        return 0
    dirs = published_find_dirs(ds, base, {"original": {"original"}})
    odir = dirs.get("original") or base
    short = odir.split(f"/{PUBLISHED_SYSTEM}/")[-1]

    got, off, left = 0, 0, set(want)
    while left and off < 6000:
        ents, _ = _ds_list(ds, odir, limit=100, offset=off)
        if not ents:
            break
        for e in ents:
            nm = (_file_attr(e, "name") or "").split("/")[-1]
            if nm not in left:
                continue
            out = dest / nm
            try:
                _ds_download(ds, _file_attr(e, "url") or f"{odir}/{nm}", out)
                if out.exists() and out.stat().st_size > 0:
                    got += 1
                    left.discard(nm)
                elif out.exists():
                    out.unlink()
            except Exception as ex:
                if verbose:
                    print(f"  [dapi] download '{nm}' failed: {str(ex)[:90]}")
        if len(ents) < 100:
            break
        off += 100
    if verbose:
        print(f"  [dapi] downloaded {got}/{len(filenames)} from {short}")
    return got


# --------------------------------------------------------------------------- #
# Recursive file transfer (dapi upload/download operate on single files)
# --------------------------------------------------------------------------- #

def upload_dir(ds, local_dir, remote_dir_uri: str, verbose: bool = False) -> int:
    """Upload every file under ``local_dir`` to ``remote_dir_uri`` (a Tapis URI),
    preserving the relative layout. Quiet by default — prints only a summary."""
    local_dir = Path(local_dir)
    base = remote_dir_uri.rstrip("/")
    files = [f for f in sorted(local_dir.rglob("*")) if f.is_file()]
    for f in files:
        rel = f.relative_to(local_dir).as_posix()
        with quiet(not verbose):
            ds.files.upload(str(f), f"{base}/{rel}")
        if verbose:
            print(f"[dapi]  uploaded {rel}")
    print(f"[dapi] uploaded {len(files)} file(s) -> {base}")
    return len(files)


def download_dir(ds, remote_dir_uri: str, local_dir, verbose: bool = False) -> int:
    """Recursively download a Tapis directory URI into ``local_dir`` (quiet)."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    base = remote_dir_uri.rstrip("/")
    n = 0

    def _recurse(uri: str, dest: Path):
        nonlocal n
        with quiet(not verbose):
            entries = list(ds.files.list(uri))
        for entry in entries:
            child_uri = f"{uri.rstrip('/')}/{entry.name}"
            if getattr(entry, "type", "file") == "dir":
                (dest / entry.name).mkdir(parents=True, exist_ok=True)
                _recurse(child_uri, dest / entry.name)
            else:
                _ds_download(ds, child_uri, dest / entry.name)
                n += 1

    _recurse(base, local_dir)
    print(f"[dapi] downloaded {n} file(s) -> {local_dir}")
    return n


# --------------------------------------------------------------------------- #
# Apps / systems discovery
# --------------------------------------------------------------------------- #

def find_apps(ds, query: str = "", verbose: bool = True):
    """List/search Tapis apps (wrapper over ds.apps.find)."""
    return ds.apps.find(query, verbose=verbose)


def gpu_queues(ds, system: str = "ls6"):
    """Return the queue DataFrame for a GPU-capable execution system.

    ls6 (Lonestar6) has NVIDIA A100 GPU queues (e.g. 'gpu-a100', 'gpu-a100-dev');
    stampede3 has Intel PVC GPU nodes. Establishes TMS credentials first.
    """
    try:
        ds.systems.establish_credentials(system)
    except Exception as exc:  # pragma: no cover
        print(f"[dapi] note: could not establish credentials for {system}: {exc}")
    return ds.systems.queues(system)


# --------------------------------------------------------------------------- #
# HPC inference job (scale the regional map to a whole region)
# --------------------------------------------------------------------------- #

#: thin utils that travel with the job; the OFFICIAL repo is cloned on the node.
_BUNDLE_CODE = ("debris_common.py", "clipseg_official.py")


def build_inference_bundle(bundle_dir, utils_dir, job_dir, weights_path,
                           input_tiles_dir, verbose: bool = True) -> Path:
    """Assemble the directory to upload as the inference job input::

        bundle/
          run_inference.py  job_infer.sh        (from designsafe_job/)
          debris_common.py  clipseg_official.py (from utils/)
          weights/<file>
          inputs/grid-*-imagery.tif

    ``job_infer.sh`` clones the **official** CLIPSeg-debris repo on the node and
    runs the official model on the staged GeoTIFF tiles (georeferenced masks out).
    """
    bundle_dir, utils_dir, job_dir = Path(bundle_dir), Path(utils_dir), Path(job_dir)
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    for name in ("run_inference.py", "job_infer.sh"):
        shutil.copy(job_dir / name, bundle_dir / name)
    for name in _BUNDLE_CODE:
        shutil.copy(utils_dir / name, bundle_dir / name)

    (bundle_dir / "weights").mkdir()
    shutil.copy(weights_path, bundle_dir / "weights" / Path(weights_path).name)

    inp = bundle_dir / "inputs"; inp.mkdir()
    n = 0
    for tif in sorted(Path(input_tiles_dir).glob("grid-*-imagery.tif")):
        shutil.copy(tif, inp / tif.name)
        n += 1
    if verbose:
        print(f"[dapi] inference bundle ready at {bundle_dir} ({n} input tiles)")
    return bundle_dir


def generate_inference_job(ds, input_dir_uri: str, allocation: str,
                           app_id: str = "designsafe-agnostic-app",
                           system: str = "ls6", queue: str = "gpu-a100",
                           weights_filename: str = "clipseg_debris.safetensors",
                           max_minutes: int = 60, node_count: int = 1,
                           cores_per_node: int = 1, memory_mb: int = 90000,
                           job_name: str = "clipseg-debris-regional", **extra):
    """Build a Tapis job request that runs the official CLIPSeg-debris on the
    staged tiles via ``job_infer.sh`` on a GPU system/queue.

    Defaults to Lonestar6 A100 (``ls6`` / ``gpu-a100``). ``memory_mb`` is capped to
    stay under per-queue limits (Vista ``gh`` max is 96000). Verify names with
    ``find_apps`` / ``gpu_queues``."""
    with quiet():
        req = ds.jobs.generate(
            app_id=app_id, input_dir_uri=input_dir_uri, script_filename="job_infer.sh",
            allocation=allocation, queue=queue, max_minutes=max_minutes,
            node_count=node_count, cores_per_node=cores_per_node, job_name=job_name,
            extra_env_vars=[{"key": "WEIGHTS_FILENAME", "value": weights_filename}],
            **extra,
        )
    req["execSystemId"] = system          # honor the requested exec system
    req["execSystemLogicalQueue"] = queue
    if memory_mb is not None:
        req["memoryMB"] = int(memory_mb)
    _fix_archive(req, ds)
    _set_main_program(req, "bash")        # run job_infer.sh with bash, not python3
    return req


# --------------------------------------------------------------------------- #
# HPC TRAINING job on Vista (TACC GH200) + wandb progress
# --------------------------------------------------------------------------- #

def build_training_bundle(bundle_dir, job_dir, dataset_dir=None,
                          verbose: bool = True) -> Path:
    """Assemble the training job input: ``train_clipseg.sh`` + the prepared dataset
    (copied into ``bundle/dataset/``). Bundling the dataset is necessary because
    Vista compute nodes do NOT mount MyData — the fine-tune set is small, so it
    travels with the job. The node clones the official repo and runs ``src/train.py``.
    """
    bundle_dir, job_dir = Path(bundle_dir), Path(job_dir)
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    shutil.copy(job_dir / "train_clipseg.sh", bundle_dir / "train_clipseg.sh")
    n = 0
    if dataset_dir:
        dst = bundle_dir / "dataset"
        shutil.copytree(dataset_dir, dst)
        n = sum(1 for f in dst.rglob("*") if f.is_file())
    if verbose:
        print(f"[dapi] training bundle ready at {bundle_dir} "
              f"({n} dataset file(s) staged)")
    return bundle_dir


def generate_training_job(ds, input_dir_uri: str, allocation: str,
                          wandb_project: str = "", wandb_entity: str = "",
                          use_wandb: bool = True,
                          app_id: str = "designsafe-agnostic-app",
                          system: str = "vista", queue: str = "gh-dev",
                          max_minutes: int = 110, max_epochs: int = 10,  # <2h gh-dev cap
                          memory_mb: int = 90000,
                          job_name: str = "clipseg-debris-train-vista", **extra):
    """Build a Tapis job that runs the official ``src/train.py
    experiment=clipseg_finetune`` on **Vista** (GH200). With ``use_wandb=True`` it
    logs to a (public) W&B project; with ``use_wandb=False`` it uses a CSV logger and
    you track progress via DAPI (job status + the job log). The dataset travels in the
    job bundle (``inputDirectory/dataset``); ``train_clipseg.sh`` reads these env vars.

    SECURITY: the W&B API key is **never** placed in the job request — when W&B is
    enabled the Vista node authenticates via your shared ``~/.netrc``.
    ``memory_mb`` stays under Vista ``gh``'s 96000 cap.
    """
    env = [
        {"key": "USE_WANDB", "value": "1" if use_wandb else "0"},
        {"key": "WANDB_PROJECT", "value": wandb_project},
        {"key": "WANDB_ENTITY", "value": wandb_entity},
        {"key": "MAX_EPOCHS", "value": str(max_epochs)},
    ]
    with quiet():
        req = ds.jobs.generate(
            app_id=app_id, input_dir_uri=input_dir_uri, script_filename="train_clipseg.sh",
            allocation=allocation, queue=queue, max_minutes=max_minutes,
            node_count=1, cores_per_node=1, job_name=job_name,
            extra_env_vars=env, **extra,
        )
    req["execSystemId"] = system
    req["execSystemLogicalQueue"] = queue
    if memory_mb is not None:
        req["memoryMB"] = int(memory_mb)
    _fix_archive(req, ds)
    _set_main_program(req, "bash")        # run train_clipseg.sh with bash, not python3
    return req


def wandb_panel(url: str, entity: str = "", project: str = "", run_id: str = None,
                metrics=("train/loss", "val/dice_debris", "val/iou_macro", "val/dice"),
                verbose: bool = True):
    """Show W&B results inline **without an iframe**. wandb.ai sends
    ``X-Frame-Options: DENY``, so an ``<iframe>`` embed fails with "refused to
    connect". Instead this renders:
      1) a clickable button + link to the live dashboard (always works), and
      2) matplotlib curves pulled from the W&B **public API** for the newest run
         (works wherever wandb is authenticated — e.g. the node's ``~/.netrc``).
    Falls back to just the link if the API isn't reachable or no run has logged yet.
    """
    from IPython.display import display, HTML
    display(HTML(
        f'<div style="margin:8px 0;font-family:sans-serif">'
        f'<a href="{url}" target="_blank" rel="noopener" style="display:inline-block;'
        f'padding:9px 16px;background:#FFCC33;color:#111;border-radius:6px;'
        f'font-weight:600;text-decoration:none">&#9654;&nbsp;Open the live W&amp;B dashboard</a>'
        f'&nbsp;&nbsp;<a href="{url}" target="_blank">{url}</a>'
        f'<div style="color:#666;font-size:12px;margin-top:4px">'
        f'(wandb.ai can\'t be embedded in a notebook iframe; curves below are pulled '
        f'live from the W&amp;B API &mdash; re-run this cell to refresh.)</div></div>'))
    if not (entity and project):
        return
    try:
        import math
        import wandb
        import matplotlib.pyplot as plt
        api = wandb.Api(timeout=20)
        if run_id:
            runs = [api.run(f"{entity}/{project}/{run_id}")]
        else:
            runs = list(api.runs(f"{entity}/{project}", order="-created_at"))
        if not runs:
            if verbose:
                print("(no W&B runs in this project yet — re-run this cell once training logs.)")
            return
        run = runs[0]
        hist = run.history(pandas=True)
        cols = [c for c in getattr(hist, "columns", [])]
        chosen = [m for m in metrics if m in cols]
        if not chosen:  # fall back to any logged loss/score columns
            chosen = [c for c in cols
                      if any(k in c.lower() for k in ("loss", "dice", "iou", "f1"))][:4]
        if not chosen:
            if verbose:
                print(f"latest run: {run.name} ({run.state}); metrics not populated yet — "
                      "re-run in a moment.")
            return
        xcol = "epoch" if "epoch" in cols else ("_step" if "_step" in cols else None)
        n = len(chosen); ncol = min(2, n); nrow = math.ceil(n / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 3.1 * nrow), squeeze=False)
        for ax, m in zip(axes.ravel(), chosen):
            sub = hist[[c for c in ([xcol, m] if xcol else [m]) if c]].dropna()
            if sub.empty:
                ax.set_title(f"{m} (no data yet)"); ax.axis("off"); continue
            x = sub[xcol] if xcol else range(len(sub))
            ax.plot(x, sub[m], marker="o", ms=3, lw=1.5)
            ax.set_title(m); ax.set_xlabel(xcol or "step"); ax.grid(alpha=.3)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.suptitle(f"W&B run: {run.name}  ({run.state})", y=1.02, fontweight="bold")
        fig.tight_layout()
        display(fig)
        plt.close(fig)
        if verbose:
            print(f"newest run: {run.name} -> {run.url}")
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"(inline W&B curves unavailable: {str(exc)[:140]} — use the link above.)")


def wandb_embed(url: str, height: int = 720):
    """Deprecated: wandb.ai blocks iframes (X-Frame-Options: DENY). Kept as a
    link-only fallback; prefer ``wandb_panel``."""
    from IPython.display import HTML
    return HTML(f'<a href="{url}" target="_blank" rel="noopener">{url}</a>')


def _archive_uri(job):
    """Base ``tapis://<sys>/<dir>`` of a finished job's archive (dapi exposes this as
    ``job.archive_uri``; fall back to building it from ``job.details``)."""
    uri = getattr(job, "archive_uri", None)
    if uri:
        return str(uri).rstrip("/")
    try:
        d = job.details
        sysid = _file_attr(d, "archiveSystemId")
        sdir = _file_attr(d, "archiveSystemDir")
        if sysid and sdir:
            return f"tapis://{sysid}/{str(sdir).strip('/')}"
    except Exception:
        pass
    return None


def download_job_outputs(ds, job, dest_dir, pattern: str = "*-mask.tif",
                         candidates=None, skip=None, verbose: bool = True,
                         max_entries: int = 4000) -> int:
    """Download a finished job's output files matching ``pattern`` using the regular
    **files API on the archive directory** (``job.archive_uri`` + ``ds.files.list`` /
    ``ds.files.download``).

    Why not ``job.list_outputs(path=...)``? On the agnostic app that wrapper fails to
    list archive SUBdirectories (and a dangling symlink in a parent dir breaks the
    parent's listing entirely). So we list the KNOWN output dirs **directly** — which
    works even when the parent can't be listed — then fall back to a bounded recursive
    files-API walk. ``ds`` is the DSClient.
    """
    import fnmatch
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    skip = skip or {"CLIPSeg-debris", "weights", "inputs", ".git", ".clip_cache", "__pycache__"}
    base = _archive_uri(job)
    if not base:
        if verbose:
            print("[dapi] could not resolve job.archive_uri — cannot download outputs.")
        return 0
    state = {"got": 0, "seen": 0}

    def _list(uri):
        with quiet():
            return ds.files.list(uri) or []

    def _dl(remote_uri, name):
        out = dest / name
        try:
            _ds_download(ds, remote_uri, out)
            if out.exists() and out.stat().st_size > 0:
                state["got"] += 1
                return True
            if out.exists():
                out.unlink()
        except Exception as ex:
            if verbose:
                print(f"  [dapi] download '{name}' failed: {str(ex)[:90]}")
        return False

    def _grab(entries, parent_uri):
        for e in entries:
            nm = (_file_attr(e, "name") or "").split("/")[-1]
            if not nm or _file_attr(e, "type") == "dir":
                continue
            if fnmatch.fnmatch(nm, pattern):
                _dl(_file_attr(e, "url") or f"{parent_uri}/{nm}", nm)

    # 1) Known output dirs, listed DIRECTLY (robust to un-listable parents).
    for c in (candidates or ["inputDirectory/outputs/mask_tif", "outputs/mask_tif",
                             "inputDirectory/outputs", "outputs"]):
        uri = f"{base}/{c}"
        try:
            entries = _list(uri)
        except Exception as ex:
            if verbose:
                print(f"  [dapi] (no {c}: {str(ex)[:70]})")
            continue
        _grab(entries, uri)
        if state["got"]:
            if verbose:
                print(f"[dapi] downloaded {state['got']} file(s) matching '{pattern}' "
                      f"from {c} -> {dest}")
            return state["got"]

    # 2) Bounded recursive walk via the files API (skips big/symlinked dirs).
    def walk(rel, depth):
        if depth > 5 or state["seen"] > max_entries:
            return
        uri = f"{base}/{rel}" if rel else base
        try:
            entries = _list(uri)
        except Exception:
            return
        for e in entries:
            state["seen"] += 1
            nm = (_file_attr(e, "name") or "").split("/")[-1]
            if not nm:
                continue
            if _file_attr(e, "type") == "dir":
                if nm not in skip:
                    walk(f"{rel}/{nm}" if rel else nm, depth + 1)
            elif fnmatch.fnmatch(nm, pattern):
                _dl(_file_attr(e, "url") or f"{uri}/{nm}", nm)

    walk("", 0)
    if verbose:
        print(f"[dapi] downloaded {state['got']} file(s) matching '{pattern}' -> {dest}")
    return state["got"]


def print_job_tree(ds, job, max_depth: int = 4, show_glob: str = "*-mask.tif",
                   max_entries: int = 4000):
    """Print a finished job's archive tree (dirs + files matching ``show_glob``) via the
    files API on ``job.archive_uri`` — so it works even where ``job.list_outputs`` can't
    recurse. Use it to SEE where Tapis put the outputs when a download finds nothing."""
    import fnmatch
    base = _archive_uri(job)
    if not base:
        print("  [could not resolve job.archive_uri]")
        return
    seen = {"n": 0}

    def walk(rel, depth):
        if depth > max_depth or seen["n"] > max_entries:
            return
        uri = f"{base}/{rel}" if rel else base
        try:
            with quiet():
                entries = ds.files.list(uri) or []
        except Exception as ex:
            print(f"{'  ' * depth}[could not list '{rel or '/'}': {str(ex)[:80]}]")
            return
        for e in entries:
            seen["n"] += 1
            nm = (_file_attr(e, "name") or "").split("/")[-1]
            if not nm:
                continue
            if _file_attr(e, "type") == "dir":
                print(f"{'  ' * depth}{nm}/")
                walk(f"{rel}/{nm}" if rel else nm, depth + 1)
            elif fnmatch.fnmatch(nm, show_glob):
                print(f"{'  ' * depth}{nm}")

    walk("", 0)


def show_job_logs(job, max_lines: int = 40):
    """Print the tail of a job's stderr/stdout (to see WHY it failed)."""
    for fname in ("tapisjob.err", "tapisjob.out"):
        try:
            with quiet():
                content = job.get_output_content(fname, max_lines=max_lines, missing_ok=True)
            if content:
                print(f"\n--- {fname} (last {max_lines} lines) ---\n{content}")
        except Exception as exc:
            print(f"[dapi] could not read {fname}: {exc}")


def submit(ds, job_request):
    """Submit a job WITHOUT blocking. Prints the UUID + initial status and returns
    the job. Use this for long/queued jobs (e.g. training) and watch progress via
    wandb or ``job_status`` rather than blocking the notebook."""
    with quiet():
        job = ds.jobs.submit(job_request)
    uuid = getattr(job, "uuid", "?")
    print(f"[dapi] submitted {uuid} -> "
          f"{job_request.get('execSystemId')}/{job_request.get('execSystemLogicalQueue')}")
    try:
        print(f"[dapi] status: {job.get_status()}")
    except Exception:
        pass
    return job


def job_status(ds, job_or_uuid):
    """Re-check a job's status by job object or UUID (e.g. after a queue wait)."""
    job = job_or_uuid
    if isinstance(job_or_uuid, str):
        try:
            job = ds.jobs.get(job_or_uuid)
        except Exception:
            job = ds.jobs.job(job_or_uuid)   # older dapi API
    status = job.get_status()
    print(f"[dapi] {getattr(job, 'uuid', '?')}: {status}")
    if str(status).upper() == "FAILED":
        show_job_logs(job)
    return status


# --------------------------------------------------------------------------- #
def submit_and_monitor(ds, job_request, interval: int = 15, timeout_minutes: int = 120,
                       logs_on_fail: bool = True):
    """Submit a job request and block until it reaches a terminal state, OR until
    the monitor times out. Output is concise; on FAILED, the job's error-log tail
    prints. NOTE: a 'TIMEOUT' result means the *monitor* gave up waiting (often a
    long queue) — the job is still queued/running; re-check with ``job_status``."""
    with quiet():
        job = ds.jobs.submit(job_request)
    print(f"[dapi] submitted {getattr(job, 'uuid', '?')} -> "
          f"{job_request.get('execSystemId')}/{job_request.get('execSystemLogicalQueue')}")
    status = job.monitor(interval=interval, timeout_minutes=timeout_minutes)
    print(f"[dapi] status after monitor: {status}")
    if str(status).upper() == "TIMEOUT":
        print("[dapi] (monitor timed out — job is still queued/running, NOT failed. "
              "Re-check later with dh.job_status(ds, job).)")
    if logs_on_fail and str(status).upper() == "FAILED":
        show_job_logs(job)
    return job
