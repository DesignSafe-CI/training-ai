#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# job_infer.sh — entrypoint executed by the DesignSafe "agnostic" Tapis app
# (run via `bash job_infer.sh`). Runs the OFFICIAL CLIPSeg-debris model on the
# staged GeoTIFF tiles. The job input dir is the working dir and contains:
#   run_inference.py  debris_common.py  clipseg_official.py
#   weights/<ckpt|safetensors>          inputs/grid-*-imagery.tif
# Results land in outputs/ (georeferenced masks), archived back to your storage.
# ---------------------------------------------------------------------------
set -e   # (no -u/pipefail: globs like weights/*.safetensors must not abort the script)
echo "[job] host=$(hostname)  pwd=$(pwd)  date=$(date)"
echo "[job] inputs: $(ls inputs 2>/dev/null | wc -l) tile(s)"

# 1. Get a Python that has PyTorch. The batch node's bare python3 does NOT; the
#    ML stack is in a TACC module (same one the Jupyter session uses), which also
#    exposes your ~/.local packages (hydra, lightning, openai-clip, rasterio, ...).
module load gcc cuda python3 2>/dev/null || true
if ! python3 -c "import torch" 2>/dev/null; then
    # Fallback: the exact module python the Vista session uses (shared /opt/apps).
    export PATH="/opt/apps/gcc14/cuda12/python3/3.11.8/bin:${PATH}"
fi
echo "[job] python3: $(command -v python3)  ($(python3 -V 2>&1))"
python3 -c "import torch; print('[job] torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    || { echo "[job] ERROR: no torch even after module load"; exit 1; }

# 2. Official repo — cached in $HOME (shared between login + compute nodes on TACC)
#    so repeat jobs SKIP the clone. This is the main per-job overhead; the GPU
#    inference itself is only seconds. First job clones (~30-60s); later jobs reuse.
REPO_CACHE="${HOME}/.clipseg_cache/CLIPSeg-debris"
if [ -z "$(ls -A "${REPO_CACHE}" 2>/dev/null)" ]; then
    mkdir -p "${HOME}/.clipseg_cache"
    git clone --branch v1.0.1 --depth 1 \
        https://github.com/Way-Yuhao/CLIPSeg-debris.git "${REPO_CACHE}" \
        || echo "[job] warn: clone to \$HOME cache failed"
fi
# IMPORTANT: use the cached repo IN PLACE via an absolute path. Do NOT symlink it into
# the job dir — Tapis archives the job dir, and a symlink pointing at $HOME becomes a
# DANGLING link on the archive system, which makes `listFiles` on the dir fail (=> the
# outputs/ masks become un-listable / un-downloadable).
if [ -n "$(ls -A "${REPO_CACHE}" 2>/dev/null)" ]; then
    REPO_DIR="${REPO_CACHE}"
    echo "[job] repo (cached, used in place): ${REPO_DIR}"
else
    [ -d CLIPSeg-debris ] || git clone --branch v1.0.1 --depth 1 \
        https://github.com/Way-Yuhao/CLIPSeg-debris.git
    REPO_DIR="${PWD}/CLIPSeg-debris"
fi
# Light deps land in ~/.local (shared) → a fast no-op after the first job.
python3 -m pip install --user -q openai-clip safetensors rasterio pillow numpy tqdm \
    || echo "[job] warn: pip install issues (offline node?) — relying on preinstalled pkgs"
# CLIP's ViT-B/16 backbone caches to ~/.cache/clip (shared $HOME) → downloaded once.

# 3. Run inference (official model on the staged tiles).
WEIGHTS=""
for f in weights/*.safetensors weights/*.ckpt; do
    if [ -f "$f" ]; then WEIGHTS="$f"; break; fi
done
echo "[job] weights: ${WEIGHTS:-<none>}"
python3 run_inference.py --input-dir inputs --output-dir outputs \
    --repo "${REPO_DIR}" --weights "${WEIGHTS}"

echo "[job] finished; outputs:"; ls -R outputs | head -n 40
