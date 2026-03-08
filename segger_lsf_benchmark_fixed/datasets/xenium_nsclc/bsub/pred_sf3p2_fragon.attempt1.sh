#!/usr/bin/env bash
#BSUB -J segger_xenium_nsclc_pred_sf3p2_fragon_cleanrun_20260305_a1
#BSUB -o /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_nsclc/logs/pred_sf3p2_fragon.attempt1.out
#BSUB -e /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_nsclc/logs/pred_sf3p2_fragon.attempt1.err
#BSUB -L /bin/bash
#BSUB -q gpu-pro
#BSUB -W 12:00
#BSUB -n 8
#BSUB -R "rusage[mem=256G]"
#BSUB -M 256G
#BSUB -gpu "num=1:j_exclusive=yes:gmem=60G"
#BSUB -w "done(segger_xenium_nsclc_baseline_cleanrun_20260305_a1)"

set -euo pipefail

write_segment_status() {
  local stage_status="$1"
  local stage_rc="${2:-0}"
  printf 'segment_status=%s\nsegment_rc=%s\n' "${stage_status}" "${stage_rc}" > /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_nsclc/runs/pred_sf3p2_fragon/.segment_status
}

classify_segment_status() {
  local stage_rc="$1"
  case "${stage_rc}" in
    137|140) printf '%s_oom' predict ;;
    138|143) printf '%s_runlimit' predict ;;
    *) printf '%s_error' predict ;;
  esac
}

on_primary_exit() {
  local rc=$?
  local current_status=""
  trap - EXIT INT TERM HUP
  if [[ -f /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_nsclc/runs/pred_sf3p2_fragon/.segment_status ]]; then
    current_status="$(awk -F'=' '$1 == \"segment_status\" { print $2; exit }' /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_nsclc/runs/pred_sf3p2_fragon/.segment_status 2>/dev/null || true)"
  fi
  if [[ "${rc}" -ne 0 ]] && [[ "${current_status}" != predict_ok ]]; then
    write_segment_status "$(classify_segment_status "${rc}")" "${rc}"
  elif [[ -z "${current_status}" ]]; then
    write_segment_status "pending" "${rc}"
  fi
  exit "${rc}"
}

trap on_primary_exit EXIT INT TERM HUP

cd /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0
set +u
if [[ 0 != "1" ]]; then
    unset PYTHONPATH || true
fi
export MAMBA_NO_ENV_PROMPT=1
export CONDA_PROMPT_MODIFIER=""
export PYTHONNOUSERSITE=1
export MAMBA_EXE=/Users/b450-admin/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/dkfz/cluster/gpu/data/OE0606/elihei/micromamba
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"
fi
unset __mamba_setup
micromamba activate /omics/groups/OE0606/internal/elihei/projects/conda_envs/seggerv2
set -u

echo "[JOB] dataset=xenium_nsclc job=pred_sf3p2_fragon attempt=1"
echo "[JOB] queue=gpu-pro gmem=60G mem=256G wall=12:00"
hostname || true
date || true
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits || true

mkdir -p /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_nsclc/runs/pred_sf3p2_fragon
write_segment_status "running" "0"
segger predict -c /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_nsclc/runs/baseline/checkpoints/last.ckpt -i /dkfz/cluster/gpu/data/OE0606/fengyun/data/xenium_nscls/xenium_fixed -o /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_nsclc/runs/pred_sf3p2_fragon --prediction-scale-factor 3.2 --use-3d checkpoint --fragment-mode
segger_rc=$?
echo "[JOB] segger_rc=${segger_rc}"
if [[ "${segger_rc}" -ne 0 ]]; then
  write_segment_status "$(classify_segment_status "${segger_rc}")" "${segger_rc}"
  exit "${segger_rc}"
fi

if [[ -f /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_nsclc/runs/pred_sf3p2_fragon/segger_segmentation.parquet ]]; then
  ln -sfn segger_segmentation.parquet /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_nsclc/runs/pred_sf3p2_fragon/transcripts.parquet
fi

write_segment_status predict_ok "0"
echo "[JOB] completed"
exit 0
