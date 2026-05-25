#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# train_clipseg.sh — fine-tune CLIPSeg-debris on TACC Vista (GH200) via the
# DesignSafe "agnostic" Tapis app (run via `bash train_clipseg.sh`), streaming
# metrics to a PUBLIC wandb project. Calls the OFFICIAL src/train.py.
#
# Env vars (set by dapi_helpers.generate_training_job):
#   DATASET_DIR, WANDB_PROJECT, WANDB_ENTITY, MAX_EPOCHS
# The W&B key is NOT passed in the job — wandb reads ~/.netrc from your shared
# TACC home (written by `wandb login`).
# ---------------------------------------------------------------------------
set -e   # (no -u/pipefail so optional globs/env don't abort the script)
echo "[train] host=$(hostname)  date=$(date)"
nvidia-smi || echo "[train] (no nvidia-smi)"
JOB_DIR="$(pwd)"   # staged bundle dir (contains dataset/); capture before any cd

# 1. Python with PyTorch (batch node's bare python3 lacks it); the TACC module
#    also exposes your ~/.local deps (hydra, lightning, wandb, openai-clip, ...).
module load gcc cuda python3 2>/dev/null || true
if ! python3 -c "import torch" 2>/dev/null; then
    export PATH="/opt/apps/gcc14/cuda12/python3/3.11.8/bin:${PATH}"   # Vista module python
fi
echo "[train] python3: $(command -v python3)  ($(python3 -V 2>&1))"
python3 -c "import torch; print('[train] torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    || { echo "[train] ERROR: no torch even after module load"; exit 1; }

# 2. Official repo (training deps already in ~/.local from install_requirements;
#    we do NOT `pip install -r requirements.txt` — its hard pins conflict on Vista).
if [ ! -d CLIPSeg-debris ]; then
    git clone --branch v1.0.1 --depth 1 https://github.com/Way-Yuhao/CLIPSeg-debris.git
fi

# Stage the bundled dataset to a HYPHEN-FREE path: the job dir contains the job
# UUID (with hyphens), which breaks the dataset's id parser (fullpath.split('-')[2]).
SRC_DATA="${DATASET_DIR:-$JOB_DIR/dataset}"
DATA_DIR="/tmp/clipseg_ft_data"
rm -rf "$DATA_DIR"; cp -r "$SRC_DATA" "$DATA_DIR"
echo "[train] dataset: $SRC_DATA -> $DATA_DIR"
echo "[train]   original=$(ls "$DATA_DIR/original" 2>/dev/null | wc -l)  seg=$(ls "$DATA_DIR/segmentation_merged" 2>/dev/null | wc -l)"

cd CLIPSeg-debris
export WANDB_API_KEY="${WANDB_API_KEY:-}"     # empty -> wandb uses ~/.netrc
export CLIP_CACHE="${PWD}/.clip_cache"; mkdir -p "${CLIP_CACHE}"
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE 2>/dev/null || true   # single-GPU, not SLURM-DDP
mkdir -p "${PWD}/training_logs"

# 3. Official training on one GH200 GPU. Two tracking modes:
#    USE_WANDB=1 -> Weights & Biases logger (live loss/IoU/Dice curves). The
#      DebrisWandbLogger normally logs 4 fixed validation images and errors if they
#      aren't in the val split, so a subsampled demo set needs show_val_ids=[] (keeps
#      the curves, skips those fixed thumbnails).
#    USE_WANDB=0 -> CSV logger (no account needed); track via DAPI job status + this
#      log's per-epoch metrics. Drop the W&B callback since there is no W&B run.
#    check_val_every_n_epoch=1: the experiment default is 5, so a short demo would
#      never validate; =1 runs validation each epoch.
if [ "${USE_WANDB:-1}" = "1" ]; then
    LOGGER_ARGS="logger=wandb logger.wandb.project=${WANDB_PROJECT} logger.wandb.entity=${WANDB_ENTITY:-} logger.wandb.offline=False callbacks.debris_wandb_logger.show_val_ids=[]"
else
    LOGGER_ARGS="logger=csv ~callbacks.debris_wandb_logger"
fi
python3 src/train.py \
    experiment=clipseg_finetune \
    trainer=gpu trainer.devices=1 \
    trainer.max_epochs="${MAX_EPOCHS:-10}" \
    trainer.check_val_every_n_epoch=1 \
    data=debris_one_hot \
    data.dataset.dataset_dir="${DATA_DIR}" \
    +local.data_dir="${DATA_DIR}" \
    +local.log_dir="${PWD}/training_logs" \
    ${LOGGER_ARGS} \
    tags="[clipseg-debris,vista,nheri-2026]"

echo "[train] finished; checkpoints under CLIPSeg-debris/logs/ (archived by Tapis)"
