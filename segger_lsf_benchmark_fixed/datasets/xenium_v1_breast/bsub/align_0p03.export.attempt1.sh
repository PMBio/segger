#!/usr/bin/env bash
#BSUB -J segger_xenium_v1_breast_align_0p03_cleanrun_20260305_a1_export
#BSUB -o /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_v1_breast/logs/align_0p03.export.attempt1.out
#BSUB -e /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_v1_breast/logs/align_0p03.export.attempt1.err
#BSUB -L /bin/bash
#BSUB -q gpu-pro
#BSUB -W 6:00
#BSUB -n 4
#BSUB -R "rusage[mem=128G]"
#BSUB -M 128G
#BSUB -w "done(segger_xenium_v1_breast_align_0p03_cleanrun_20260305_a1)"

set -euo pipefail

write_export_status() {
  local stage_status="$1"
  local stage_rc="${2:-0}"
  local anndata_rc="${3:-0}"
  local xenium_rc="${4:-0}"
  local export_note="${5:-}"
  printf 'export_status=%s\nexport_rc=%s\nanndata_rc=%s\nxenium_rc=%s\nexport_note=%s\n'     "${stage_status}" "${stage_rc}" "${anndata_rc}" "${xenium_rc}" "${export_note}"     > /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_v1_breast/runs/align_0p03/.export_status
}

on_export_exit() {
  local rc=$?
  local current_status=""
  trap - EXIT INT TERM HUP
  if [[ -f /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_v1_breast/runs/align_0p03/.export_status ]]; then
    current_status="$(awk -F'=' '$1 == \"export_status\" { print $2; exit }' /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_v1_breast/runs/align_0p03/.export_status 2>/dev/null || true)"
  fi
  if [[ "${rc}" -ne 0 ]] && [[ "${current_status}" != "export_ok" ]] && [[ "${current_status}" != "export_skipped_xenium_missing_source" ]]; then
    write_export_status "export_error" "${rc}" "0" "0" ""
  elif [[ -z "${current_status}" ]]; then
    write_export_status "pending" "${rc}" "0" "0" ""
  fi
  exit "${rc}"
}

trap on_export_exit EXIT INT TERM HUP

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

echo "[JOB] dataset=xenium_v1_breast job=align_0p03 export attempt=1"
echo "[JOB] queue=gpu-pro mem=128G wall=6:00"
hostname || true
date || true

if [[ 1 != "1" && 1 != "1" ]]; then
  write_export_status "not_requested" "0" "0" "0" ""
  echo "[JOB] export disabled"
  exit 0
fi

if [[ ! -f /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_v1_breast/runs/align_0p03/segger_segmentation.parquet ]]; then
  write_export_status "export_error" "1" "0" "0" "missing_segmentation"
  echo "[JOB] export missing segmentation input"
  exit 1
fi

mkdir -p /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_v1_breast/exports/align_0p03/anndata /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_v1_breast/exports/align_0p03/xenium_explorer
write_export_status "running" "0" "0" "0" ""

anndata_rc=0
xenium_rc=0
overall_rc=0
final_status="export_ok"
export_note=""

if [[ 1 == "1" ]]; then
  segger export -s /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_v1_breast/runs/align_0p03/segger_segmentation.parquet -i /dkfz/cluster/gpu/data/OE0606/fengyun/data/xenium_v1_breast_fixed -o /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_v1_breast/exports/align_0p03/anndata --format anndata || anndata_rc=$?
fi

if [[ 1 == "1" ]]; then
  if [[ 0 == "1" ]]; then
    segger export -s /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_v1_breast/runs/align_0p03/segger_segmentation.parquet -i /dkfz/cluster/gpu/data/OE0606/fengyun/data/xenium_v1_breast_fixed -o /Users/b450-admin/Desktop/Projects/Active/segger/segger-0.2.0/segger_lsf_benchmark_fixed/datasets/xenium_v1_breast/exports/align_0p03/xenium_explorer --format xenium_explorer --boundary-method convex_hull --boundary-voxel-size 5 --num-workers 8 || xenium_rc=$?
  elif [[ 1 == "1" ]]; then
    export_note="xenium_missing_source"
    final_status="export_skipped_xenium_missing_source"
  else
    xenium_rc=1
    export_note="xenium_missing_source_required"
  fi
fi

if [[ "${anndata_rc}" -ne 0 ]]; then
  overall_rc="${anndata_rc}"
  final_status="export_error"
elif [[ "${xenium_rc}" -ne 0 ]]; then
  overall_rc="${xenium_rc}"
  final_status="export_error"
fi

write_export_status "${final_status}" "${overall_rc}" "${anndata_rc}" "${xenium_rc}" "${export_note}"
echo "[JOB] export_status=${final_status} export_rc=${overall_rc}"
if [[ "${overall_rc}" -ne 0 ]]; then
  exit "${overall_rc}"
fi
echo "[JOB] completed"
exit 0
