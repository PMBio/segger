#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Aggregate per-dataset validation tables produced by
build_benchmark_validation_table.sh.

Usage:
  bash scripts/build_benchmark_validation_table_lsf_multi.sh [options]

Options:
  --root <dir>       Top-level LSF benchmark root
                     (default: /dkfz/cluster/gpu/checkpoints/OE0606/elihei/segger_lsf_benchmark_fixed)
  --out-tsv <file>   Output aggregate TSV path
                     (default: <root>/summaries/aggregate_validation_metrics.tsv)
  --no-refresh       Do not run the per-dataset validation script when a table is missing
  -h, --help         Show this help
EOF
}

ROOT="/dkfz/cluster/gpu/checkpoints/OE0606/elihei/segger_lsf_benchmark_fixed"
OUT_TSV=""
REFRESH=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --out-tsv)
      OUT_TSV="$2"
      shift 2
      ;;
    --no-refresh)
      REFRESH=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${OUT_TSV}" ]]; then
  OUT_TSV="${ROOT}/summaries/aggregate_validation_metrics.tsv"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$(dirname "${OUT_TSV}")"

header_written=0
found_any=0
tmp_out="$(mktemp)"
trap 'rm -f "${tmp_out}"' EXIT
: > "${tmp_out}"

for dataset_dir in "${ROOT}"/datasets/*; do
  [[ -d "${dataset_dir}" ]] || continue
  found_any=1
  dataset="$(basename "${dataset_dir}")"
  validation_tsv="${dataset_dir}/summaries/validation_metrics.tsv"
  context_file="${dataset_dir}/dataset_context.env"

  if [[ ! -f "${validation_tsv}" && "${REFRESH}" == "1" && -f "${context_file}" ]]; then
    unset DATASET_KEY INPUT_DIR SCRNA_REFERENCE_PATH TISSUE_TYPE SEGGER_BIN || true
    # shellcheck disable=SC1090
    source "${context_file}"
    if [[ -n "${TISSUE_TYPE:-}" ]]; then
      bash "${SCRIPT_DIR}/build_benchmark_validation_table.sh" \
        --root "${dataset_dir}" \
        --input-dir "${INPUT_DIR}" \
        --tissue-type "${TISSUE_TYPE}" \
        --include-default-10x true \
        >/dev/null || true
    elif [[ -n "${SCRNA_REFERENCE_PATH:-}" ]]; then
      bash "${SCRIPT_DIR}/build_benchmark_validation_table.sh" \
        --root "${dataset_dir}" \
        --input-dir "${INPUT_DIR}" \
        --scrna-reference-path "${SCRNA_REFERENCE_PATH}" \
        --include-default-10x true \
        >/dev/null || true
    else
      bash "${SCRIPT_DIR}/build_benchmark_validation_table.sh" \
        --root "${dataset_dir}" \
        --input-dir "${INPUT_DIR}" \
        --include-default-10x true \
        >/dev/null || true
    fi
  fi

  [[ -f "${validation_tsv}" ]] || continue
  if [[ "${header_written}" == "0" ]]; then
    head -n1 "${validation_tsv}" | awk -F'\t' '{ printf "dataset"; for (i = 1; i <= NF; i++) printf "\t%s", $i; printf "\n" }' > "${tmp_out}"
    header_written=1
  fi
  tail -n +2 "${validation_tsv}" | awk -F'\t' -v dataset="${dataset}" '
    NF > 0 {
      printf "%s", dataset
      for (i = 1; i <= NF; i++) {
        printf "\t%s", $i
      }
      printf "\n"
    }
  ' >> "${tmp_out}"
done

if [[ "${header_written}" == "0" ]]; then
  printf "dataset\n" > "${tmp_out}"
fi

cp "${tmp_out}" "${OUT_TSV}"

if [[ "${found_any}" == "0" ]]; then
  echo "No dataset roots found under ${ROOT}/datasets" >&2
fi

if command -v column >/dev/null 2>&1; then
  column -ts $'\t' "${OUT_TSV}"
else
  cat "${OUT_TSV}"
fi
