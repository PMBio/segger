#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE_EOF'
Aggregate per-dataset benchmark status snapshots produced by
benchmark_status_dashboard.sh.

Usage:
  bash scripts/benchmark_status_dashboard_lsf_multi.sh [options]

Options:
  --root <dir>          Top-level benchmark root, dataset container root,
                        or a single dataset root
                        (default: /omics/groups/OE0606/internal/elihei/projects/segger_lsf_benchmark_fixed)
  --out-tsv <file>      Output aggregate TSV path
                        (default: <root>/summaries/aggregate_status_snapshot.tsv)
  --no-refresh          Do not rerun the per-dataset dashboard script
  --watch [sec]         Refresh every N seconds (default: 20)
  --recent-limit <n>    Show the last N parsed LSF runs (default: 10, 0 = all)
  --no-color            Disable ANSI colors
  -h, --help            Show this help
USAGE_EOF
}

ROOT="/omics/groups/OE0606/internal/elihei/projects/segger_lsf_benchmark_fixed"
OUT_TSV=""
REFRESH=1
WATCH_SEC=0
RECENT_LIMIT=10
NO_COLOR=0

SCRIPT_DIR=""
DATASETS_DIR=""
declare -a DATASET_DIRS=()
declare -a DASH_WARNINGS=()

C_RESET=""
C_BOLD=""
C_BOLD_OFF=""
C_GREEN=""
C_RED=""
C_YELLOW=""
C_BLUE=""
C_CYAN=""

require_value() {
  local opt="$1"
  if [[ $# -lt 2 ]] || [[ -z "${2}" ]] || [[ "${2}" == -* ]]; then
    echo "ERROR: ${opt} requires a value." >&2
    exit 1
  fi
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

draw_progress_bar() {
  local current="$1"
  local total="$2"
  local width=40
  local fill=0
  local pct=0
  local empty left right

  if [[ "${total}" -gt 0 ]]; then
    fill=$((current * width / total))
    pct=$((current * 100 / total))
  fi
  empty=$((width - fill))
  left="$(printf '%*s' "${fill}" '' | tr ' ' '#')"
  right="$(printf '%*s' "${empty}" '' | tr ' ' '-')"
  printf "[%s%s] %d/%d (%d%%)" "${left}" "${right}" "${current}" "${total}" "${pct}"
}

add_warning() {
  local msg="$1"
  DASH_WARNINGS+=("${msg}")
}

configure_colors() {
  if [[ "${NO_COLOR}" == "0" ]] && [[ -t 1 ]]; then
    C_RESET=$'\033[0m'
    C_BOLD=$'\033[1m'
    C_BOLD_OFF=$'\033[22m'
    C_GREEN=$'\033[32m'
    C_RED=$'\033[31m'
    C_YELLOW=$'\033[33m'
    C_BLUE=$'\033[34m'
    C_CYAN=$'\033[36m'
  else
    C_RESET=""
    C_BOLD=""
    C_BOLD_OFF=""
    C_GREEN=""
    C_RED=""
    C_YELLOW=""
    C_BLUE=""
    C_CYAN=""
  fi
}

render_tsv_table() {
  awk -F'\t' '
    function repeat_char(ch, n, out, i) {
      out = ""
      for (i = 0; i < n; i++) out = out ch
      return out
    }
    function visible_len(s, t) {
      t = s
      gsub(/\033\[[0-9;]*m/, "", t)
      return length(t)
    }
    function print_cell(val, width, vis, i) {
      vis = visible_len(val)
      printf(" %s", val)
      for (i = vis; i < width; i++) {
        printf(" ")
      }
      printf(" |")
    }
    {
      rows = NR
      if ($1 == "__ROW_SEP__") {
        row_sep[NR] = 1
        next
      }
      if (NF > ncols) ncols = NF
      for (i = 1; i <= NF; i++) {
        val = $i
        sub(/\r$/, "", val)
        cells[NR, i] = val
        vis = visible_len(val)
        if (vis > widths[i]) widths[i] = vis
      }
    }
    END {
      if (rows == 0) exit

      sep = "+"
      for (i = 1; i <= ncols; i++) {
        sep = sep repeat_char("-", widths[i] + 2) "+"
      }

      print sep
      printf("|")
      for (i = 1; i <= ncols; i++) {
        print_cell(cells[1, i], widths[i])
      }
      printf("\n")
      print sep

      for (r = 2; r <= rows; r++) {
        if (row_sep[r]) {
          print sep
          continue
        }
        printf("|")
        for (i = 1; i <= ncols; i++) {
          print_cell(cells[r, i], widths[i])
        }
        printf("\n")
      }
      print sep
    }
  '
}

colorize_rows_by_state_column() {
  local state_col="$1"
  awk -F'\t' \
    -v state_col="${state_col}" \
    -v c_done="${C_GREEN}" \
    -v c_run="${C_BLUE}" \
    -v c_fail="${C_RED}" \
    -v c_pending="${C_YELLOW}" \
    -v c_partial="${C_CYAN}" \
    -v c_reset="${C_RESET}" \
    '
      function color_for_state(s, lower) {
        lower = tolower(s)
        if (lower == "done") return c_done
        if (lower == "running") return c_run
        if (lower == "failed") return c_fail
        if (lower == "pending") return c_pending
        if (lower == "partial") return c_partial
        if (lower == "reference") return c_partial
        return ""
      }
      NR == 1 { print; next }
      {
        state = (state_col > 0 && state_col <= NF) ? $state_col : ""
        color = color_for_state(state)
        if (color != "") {
          for (i = 1; i <= NF; i++) {
            $i = color $i c_reset
          }
        }
        print
      }
    ' OFS=$'\t'
}

colorize_rows_by_status_column() {
  local status_col="$1"
  awk -F'\t' \
    -v status_col="${status_col}" \
    -v c_done="${C_GREEN}" \
    -v c_run="${C_BLUE}" \
    -v c_fail="${C_RED}" \
    -v c_pending="${C_YELLOW}" \
    -v c_partial="${C_CYAN}" \
    -v c_reset="${C_RESET}" \
    '
      function state_from_status(status, lower) {
        lower = tolower(status)
        if (status == "" || status == "-" || lower == "<none>") return "pending"
        if (lower ~ /running|in_progress/) return "running"
        if (lower == "partial") return "partial"
        if (lower == "ok" || lower == "skipped_existing" || lower == "recovered_predict_ok") return "done"
        if (lower == "pending" || lower == "planned") return "pending"
        if (lower == "reference") return "reference"
        return "failed"
      }
      function color_for_state(s, lower) {
        lower = tolower(s)
        if (lower == "done") return c_done
        if (lower == "running") return c_run
        if (lower == "failed") return c_fail
        if (lower == "pending") return c_pending
        if (lower == "partial") return c_partial
        if (lower == "reference") return c_partial
        return ""
      }
      NR == 1 { print; next }
      {
        status = (status_col > 0 && status_col <= NF) ? $status_col : ""
        state = state_from_status(status)
        color = color_for_state(state)
        if (color != "") {
          for (i = 1; i <= NF; i++) {
            $i = color $i c_reset
          }
        }
        print
      }
    ' OFS=$'\t'
}

normalize_root_layout() {
  local trimmed="${ROOT%/}"
  if [[ -z "${trimmed}" ]]; then
    ROOT="/"
    return
  fi
  if [[ "$(basename "${trimmed}")" == "datasets" ]]; then
    ROOT="$(cd "${trimmed}/.." && pwd)"
  else
    ROOT="${trimmed}"
  fi
}

collect_dataset_dirs() {
  local d
  DATASET_DIRS=()
  DATASETS_DIR=""

  if [[ -d "${ROOT}/datasets" ]]; then
    DATASETS_DIR="${ROOT}/datasets"
    for d in "${DATASETS_DIR}"/*; do
      [[ -d "${d}" ]] || continue
      DATASET_DIRS+=("${d}")
    done
    return
  fi

  if [[ -f "${ROOT}/job_plan.tsv" ]]; then
    DATASETS_DIR="$(dirname "${ROOT}")"
    DATASET_DIRS=("${ROOT}")
    return
  fi

  if [[ -d "${ROOT}" ]]; then
    DATASETS_DIR="${ROOT}"
    for d in "${ROOT}"/*; do
      [[ -d "${d}" ]] || continue
      [[ -f "${d}/job_plan.tsv" ]] || continue
      DATASET_DIRS+=("${d}")
    done
  fi
}

refresh_dataset_snapshot() {
  local dataset_dir="$1"
  bash "${SCRIPT_DIR}/benchmark_status_dashboard.sh" --root "${dataset_dir}" >/dev/null 2>&1
}

print_aggregate_header() {
  printf "dataset\tjob\tgroup\tgpu\tstatus\tstate\trunning\telapsed_s\trun_count\thad_rerun\thad_anc_retry\thad_predict_fallback\thad_recovery_pass\tseg_exists\tanndata_exists\txenium_exists\tseg_dir\tlog_file\tuse_3d\texpansion\ttx_max_k\ttx_max_dist\tn_mid_layers\tn_heads\tcells_min_counts\tmin_qv\talignment_loss\tsegment_status\texport_status\tsegment_job_id\texport_job_id\texport_note\tnote\n"
}

append_normalized_snapshot_rows() {
  local dataset="$1"
  local dataset_dir="$2"
  local snapshot_file="$3"

  awk -F'\t' -v OFS=$'\t' -v dataset="${dataset}" -v plan_file="${dataset_dir}/job_plan.tsv" '
    function state_from_status(status, lower) {
      lower = tolower(status)
      if (status == "" || status == "-" || lower == "<none>") return "pending"
      if (lower ~ /running|in_progress/) return "running"
      if (lower == "partial") return "partial"
      if (lower == "ok" || lower == "skipped_existing" || lower == "recovered_predict_ok") return "done"
      if (lower == "pending" || lower == "planned") return "pending"
      if (lower == "reference") return "reference"
      return "failed"
    }
    function col(name, fallback) {
      if ((name in idx) && idx[name] > 0 && idx[name] <= NF) return $(idx[name])
      return fallback
    }
    function canonical_bool(v, fallback, lower) {
      lower = tolower(v)
      if (lower == "1" || lower == "true" || lower == "yes") return "1"
      if (lower == "0" || lower == "false" || lower == "no") return "0"
      return fallback
    }
    BEGIN {
      plan_line = 0
      while ((getline line < plan_file) > 0) {
        n = split(line, p, "\t")
        if (plan_line == 0) {
          for (i = 1; i <= n; i++) {
            plan_idx[p[i]] = i
          }
          plan_line++
          continue
        }
        pj = (("job" in plan_idx) && plan_idx["job"] > 0 && plan_idx["job"] <= n) ? p[plan_idx["job"]] : ((n >= 1) ? p[1] : "")
        pg = (("group" in plan_idx) && plan_idx["group"] > 0 && plan_idx["group"] <= n) ? p[plan_idx["group"]] : ((n >= 2) ? p[2] : "")
        if (pj != "") group_by_job[pj] = pg
        plan_line++
      }
      close(plan_file)
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        idx[$i] = i
      }
      next
    }
    NF == 0 { next }
    {
      job = col("job", $1)
      if (job == "" || job == "-") next

      group = col("group", "")
      if (group == "") group = group_by_job[job]
      if (group == "") group = "-"

      gpu = col("gpu", "-")

      status = col("status", "")
      if (status == "" || status == "-") status = "pending"

      state = col("state", "")
      if (state == "" || state == "-") state = state_from_status(status)

      running = col("running", "")
      if (running == "" || running == "-") {
        running = (tolower(state) == "running") ? "1" : "0"
      }
      running = canonical_bool(running, "0")

      elapsed_s = col("elapsed_s", col("elapsed", ""))
      if (elapsed_s == "" || elapsed_s == "-") elapsed_s = "0"

      run_count = col("run_count", "")
      if (run_count == "" || run_count == "-") run_count = "0"

      had_rerun = canonical_bool(col("had_rerun", ""), "0")
      had_anc_retry = canonical_bool(col("had_anc_retry", ""), "0")
      had_predict_fallback = canonical_bool(col("had_predict_fallback", ""), "0")
      had_recovery_pass = canonical_bool(col("had_recovery_pass", ""), "0")

      seg_exists = canonical_bool(col("seg_exists", ""), "0")
      anndata_exists = canonical_bool(col("anndata_exists", ""), "0")
      xenium_exists = canonical_bool(col("xenium_exists", ""), "0")

      seg_dir = col("seg_dir", "")
      if (seg_dir == "" && NF >= 6) seg_dir = $6
      if (seg_dir == "") seg_dir = "-"

      log_file = col("log_file", "")
      if (log_file == "" && NF >= 7) log_file = $7
      if (log_file == "") log_file = "-"

      use_3d = col("use_3d", "-")
      expansion = col("expansion", "-")
      tx_max_k = col("tx_max_k", "-")
      tx_max_dist = col("tx_max_dist", "-")
      n_mid_layers = col("n_mid_layers", "-")
      n_heads = col("n_heads", "-")
      cells_min_counts = col("cells_min_counts", "-")
      min_qv = col("min_qv", "-")
      alignment_loss = col("alignment_loss", "-")
      segment_status = col("segment_status", "-")
      export_status = col("export_status", "-")
      segment_job_id = col("segment_job_id", "-")
      export_job_id = col("export_job_id", "-")
      export_note = col("export_note", "-")
      note = col("note", "-")

      if (segment_status == "") segment_status = "-"
      if (export_status == "") export_status = "-"
      if (segment_job_id == "") segment_job_id = "-"
      if (export_job_id == "") export_job_id = "-"
      if (export_note == "") export_note = "-"
      if (note == "") note = "-"

      print dataset, job, group, gpu, status, state, running, elapsed_s, run_count, had_rerun, had_anc_retry,
            had_predict_fallback, had_recovery_pass, seg_exists, anndata_exists, xenium_exists, seg_dir,
            log_file, use_3d, expansion, tx_max_k, tx_max_dist, n_mid_layers, n_heads, cells_min_counts,
            min_qv, alignment_loss, segment_status, export_status, segment_job_id, export_job_id, export_note, note
    }
  ' "${snapshot_file}" >> "${OUT_TSV}"
}

build_aggregate_snapshot() {
  local dataset_dir dataset snapshot all_jobs found_any=0

  DASH_WARNINGS=()
  print_aggregate_header > "${OUT_TSV}"

  for dataset_dir in "${DATASET_DIRS[@]}"; do
    [[ -d "${dataset_dir}" ]] || continue
    found_any=1
    dataset="$(basename "${dataset_dir}")"

    if [[ "${REFRESH}" == "1" ]]; then
      if ! refresh_dataset_snapshot "${dataset_dir}"; then
        add_warning "refresh failed for dataset ${dataset}"
      fi
    fi

    snapshot="${dataset_dir}/summaries/status_snapshot.tsv"
    all_jobs="${dataset_dir}/summaries/all_jobs.tsv"

    if [[ -f "${snapshot}" ]]; then
      append_normalized_snapshot_rows "${dataset}" "${dataset_dir}" "${snapshot}"
    elif [[ -f "${all_jobs}" ]]; then
      add_warning "using all_jobs.tsv fallback for dataset ${dataset}"
      append_normalized_snapshot_rows "${dataset}" "${dataset_dir}" "${all_jobs}"
    else
      add_warning "no snapshot/all_jobs found for dataset ${dataset}"
    fi
  done

  if [[ "${found_any}" == "0" ]]; then
    add_warning "No dataset roots found under ${ROOT}/datasets or ${ROOT}"
  fi
}

build_dataset_summary() {
  local out_file="$1"
  local sorted_file
  sorted_file="$(mktemp)"

  awk -F'\t' -v OFS=$'\t' '
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      next
    }
    {
      dataset = (("dataset" in idx) && idx["dataset"] > 0) ? $(idx["dataset"]) : ""
      job = (("job" in idx) && idx["job"] > 0) ? $(idx["job"]) : ""
      state = (("state" in idx) && idx["state"] > 0) ? tolower($(idx["state"])) : "pending"
      if (dataset == "" || job == "") next

      jobs[dataset]++
      if (state == "running") running[dataset]++
      else if (state == "pending") pending[dataset]++
      else if (state == "failed") failed[dataset]++
      else if (state == "done") done[dataset]++
      else if (state == "partial") partial[dataset]++
      else other[dataset]++
    }
    END {
      print "dataset\tjobs\trunning\tpending\tfailed\tdone\tpartial\tother\tprocessed\tprogress_pct"
      for (dataset in jobs) {
        processed = done[dataset] + partial[dataset] + failed[dataset]
        pct = (jobs[dataset] > 0) ? (processed * 100.0 / jobs[dataset]) : 0
        printf "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.1f\n",
          dataset,
          jobs[dataset] + 0,
          running[dataset] + 0,
          pending[dataset] + 0,
          failed[dataset] + 0,
          done[dataset] + 0,
          partial[dataset] + 0,
          other[dataset] + 0,
          processed,
          pct
      }
    }
  ' "${OUT_TSV}" > "${sorted_file}"

  {
    header="$(head -n 1 "${sorted_file}" 2>/dev/null || true)"
    if [[ -n "${header}" ]]; then
      printf "%s\n" "${header}"
    fi
    tail -n +2 "${sorted_file}" | sort -t $'\t' -k5,5nr -k3,3nr -k4,4nr -k1,1
  } > "${out_file}"

  rm -f "${sorted_file}"
}

build_state_counts() {
  local out_file="$1"
  awk -F'\t' -v OFS=$'\t' '
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      next
    }
    {
      state = (("state" in idx) && idx["state"] > 0) ? tolower($(idx["state"])) : "pending"
      if (state == "") state = "pending"
      c[state]++
    }
    END {
      print "state\tcount"
      order[1] = "running"
      order[2] = "pending"
      order[3] = "failed"
      order[4] = "done"
      order[5] = "partial"
      for (i = 1; i <= 5; i++) {
        s = order[i]
        print s, c[s] + 0
        seen[s] = 1
      }
      for (k in c) {
        if (!(k in seen)) print k, c[k]
      }
    }
  ' "${OUT_TSV}" > "${out_file}"
}

build_status_counts() {
  local out_file="$1"
  awk -F'\t' -v OFS=$'\t' '
    function state_from_status(status, lower) {
      lower = tolower(status)
      if (status == "" || status == "-" || lower == "<none>") return "pending"
      if (lower ~ /running|in_progress/) return "running"
      if (lower == "partial") return "partial"
      if (lower == "ok" || lower == "skipped_existing" || lower == "recovered_predict_ok") return "done"
      if (lower == "pending" || lower == "planned") return "pending"
      if (lower == "reference") return "reference"
      return "failed"
    }
    function rank(state, lower) {
      lower = tolower(state)
      if (lower == "running") return 1
      if (lower == "pending") return 2
      if (lower == "failed") return 3
      if (lower == "done") return 4
      if (lower == "partial") return 5
      if (lower == "reference") return 6
      return 99
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      next
    }
    {
      status = (("status" in idx) && idx["status"] > 0) ? $(idx["status"]) : ""
      if (status == "") status = "<none>"
      c[status]++
    }
    END {
      print "rank\tstatus\tcount"
      for (k in c) {
        st = state_from_status(k)
        print rank(st), k, c[k]
      }
    }
  ' "${OUT_TSV}" | {
    IFS= read -r header || true
    if [[ -n "${header}" ]]; then
      printf "status\tcount\n"
    fi
    sort -t $'\t' -k1,1n -k3,3nr -k2,2 | cut -f2-
  } > "${out_file}"
}

build_job_overview() {
  local out_file="$1"
  local sorted_file
  local root_abs=""
  sorted_file="$(mktemp)"
  if [[ -d "${ROOT}" ]]; then
    root_abs="$(cd "${ROOT}" && pwd)/"
  fi

  awk -F'\t' -v OFS=$'\t' -v root_prefix="${root_abs}" -v root_rel_prefix="${ROOT%/}/" '
    function col(name, fallback) {
      if ((name in idx) && idx[name] > 0) return $(idx[name])
      return fallback
    }
    function shorten_path(p) {
      if (p == "" || p == "-") return "-"
      if (root_prefix != "" && index(p, root_prefix) == 1) {
        return substr(p, length(root_prefix) + 1)
      }
      if (root_rel_prefix != "" && index(p, root_rel_prefix) == 1) {
        return substr(p, length(root_rel_prefix) + 1)
      }
      return p
    }
    function state_rank(state, lower) {
      lower = tolower(state)
      if (lower == "failed") return 1
      if (lower == "running") return 2
      if (lower == "partial") return 3
      if (lower == "pending") return 4
      if (lower == "done") return 5
      if (lower == "reference") return 6
      return 99
    }
    function fmt_minutes(v, lower, n) {
      lower = tolower(v)
      if (v == "" || v == "-" || lower == "nan" || lower == "none") return "0.00"
      n = (v + 0.0) / 60.0
      if (n < 0) n = 0
      return sprintf("%.2f", n)
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      print "rank\tdataset\tjob\tgroup\tgpu\tstate\tstatus\truns\telapsed_min\trerun\tanc_retry\toom_pred_fallback\trecovery\tseg\tanndata\txenium\tlog_file"
      next
    }
    {
      dataset = col("dataset", "-")
      job = col("job", "-")
      state = tolower(col("state", "pending"))
      status = col("status", "pending")
      if (dataset == "" || job == "") next

      print state_rank(state),
            dataset,
            job,
            col("group", "-"),
            col("gpu", "-"),
            state,
            status,
            col("run_count", "0"),
            fmt_minutes(col("elapsed_s", "0")),
            col("had_rerun", "0"),
            col("had_anc_retry", "0"),
            col("had_predict_fallback", "0"),
            col("had_recovery_pass", "0"),
            col("seg_exists", "0"),
            col("anndata_exists", "0"),
            col("xenium_exists", "0"),
            shorten_path(col("log_file", "-"))
    }
  ' "${OUT_TSV}" > "${sorted_file}"

  {
    header="$(head -n 1 "${sorted_file}" 2>/dev/null || true)"
    if [[ -n "${header}" ]]; then
      printf "dataset\tjob\tgroup\tgpu\tstate\tstatus\truns\telapsed_min\trerun\tanc_retry\toom_pred_fallback\trecovery\tseg\tanndata\txenium\tlog_file\n"
    fi
    tail -n +2 "${sorted_file}" | sort -t $'\t' -k1,1n -k2,2 -k3,3 | cut -f2-
  } > "${out_file}"

  rm -f "${sorted_file}"
}

build_recent_runs() {
  local out_file="$1"
  local raw_file
  local dataset_dir dataset out_log job log_name
  raw_file="$(mktemp)"

  printf "sort_key\tdataset\tjob\tlsf_job\texit\tsegger_rc\tanndata\tcompleted\tstarted_at\tterminated_at\tout_log\n" > "${raw_file}"

  for dataset_dir in "${DATASET_DIRS[@]}"; do
    [[ -d "${dataset_dir}" ]] || continue
    dataset="$(basename "${dataset_dir}")"
    for out_log in "${dataset_dir}"/logs/*.attempt*.out; do
      [[ -f "${out_log}" ]] || continue
      log_name="$(basename "${out_log}")"
      job="${log_name%%.attempt*}"

      awk -v dataset="${dataset}" -v job="${job}" -v out_log="${out_log}" '
        function month_num(mon) {
          if (mon == "Jan") return 1
          if (mon == "Feb") return 2
          if (mon == "Mar") return 3
          if (mon == "Apr") return 4
          if (mon == "May") return 5
          if (mon == "Jun") return 6
          if (mon == "Jul") return 7
          if (mon == "Aug") return 8
          if (mon == "Sep") return 9
          if (mon == "Oct") return 10
          if (mon == "Nov") return 11
          if (mon == "Dec") return 12
          return 0
        }
        function make_key(mon, day, time, year, mm) {
          mm = month_num(mon)
          if (mm == 0 || time == "" || year == "") return ""
          return sprintf("%04d-%02d-%02d %s", year + 0, mm, day + 0, time)
        }
        function flush(sort_key) {
          if (!in_block || job_id == "") return
          sort_key = start_key
          if (sort_key == "") sort_key = end_key
          if (sort_key == "") sort_key = "0000-00-00 00:00:00"
          printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n",
            sort_key,
            dataset,
            job,
            job_id,
            exit_code,
            segger_rc,
            wrote_anndata ? "1" : "0",
            completed,
            started_at,
            terminated_at,
            out_log
        }
        /^Subject: Job / {
          flush()
          in_block = 1
          job_id = $3
          sub(/:$/, "", job_id)
          exit_code = ""
          segger_rc = ""
          wrote_anndata = 0
          completed = 0
          started_at = ""
          terminated_at = ""
          start_key = ""
          end_key = ""
          next
        }
        in_block && /^Started at / {
          started_at = $0
          sub(/^Started at /, "", started_at)
          start_key = make_key($4, $5, $6, $7)
          next
        }
        in_block && /^Terminated at / {
          terminated_at = $0
          sub(/^Terminated at /, "", terminated_at)
          end_key = make_key($4, $5, $6, $7)
          next
        }
        in_block && /^Exited with exit code / {
          exit_code = $5
          sub(/\.$/, "", exit_code)
          next
        }
        in_block && /^\[JOB\] segger_rc=/ {
          segger_rc = $0
          sub(/^\[JOB\] segger_rc=/, "", segger_rc)
          next
        }
        in_block && /^\[JOB\] completed$/ {
          completed = 1
          next
        }
        in_block && /^Written AnnData output:/ {
          wrote_anndata = 1
          next
        }
        END {
          flush()
        }
      ' "${out_log}" >> "${raw_file}"
    done
  done

  {
    printf "dataset\tjob\tlsf_job\texit\tsegger_rc\tanndata\tcompleted\tstarted_at\tterminated_at\tout_log\n"
    if [[ "${RECENT_LIMIT}" == "0" ]]; then
      tail -n +2 "${raw_file}" | sort -t $'\t' -k1,1 -k2,2 -k3,3 | cut -f2-
    else
      tail -n +2 "${raw_file}" | sort -t $'\t' -k1,1 -k2,2 -k3,3 | tail -n "${RECENT_LIMIT}" | cut -f2-
    fi
  } > "${out_file}"

  rm -f "${raw_file}"
}

render_once() {
  local snapshot="${OUT_TSV}"
  local counts dataset_count total_jobs done_count running_count pending_count partial_count failed_count
  local processed
  local dataset_summary_file state_counts_file status_counts_file job_overview_file recent_runs_file

  build_aggregate_snapshot

  counts="$(awk -F'\t' '
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      next
    }
    {
      dataset = (("dataset" in idx) && idx["dataset"] > 0) ? $(idx["dataset"]) : ""
      job = (("job" in idx) && idx["job"] > 0) ? $(idx["job"]) : ""
      state = (("state" in idx) && idx["state"] > 0) ? tolower($(idx["state"])) : "pending"
      if (dataset == "" || job == "") next

      seen[dataset] = 1
      jobs++
      if (state == "done") done++
      else if (state == "running") running++
      else if (state == "pending") pending++
      else if (state == "partial") partial++
      else if (state == "failed") failed++
      else pending++
    }
    END {
      for (dataset in seen) datasets++
      printf "%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
        datasets + 0,
        jobs + 0,
        done + 0,
        running + 0,
        pending + 0,
        partial + 0,
        failed + 0
    }
  ' "${snapshot}")"

  IFS=$'\t' read -r dataset_count total_jobs done_count running_count pending_count partial_count failed_count <<< "${counts}"
  processed=$((done_count + partial_count + failed_count))

  dataset_summary_file="$(mktemp)"
  state_counts_file="$(mktemp)"
  status_counts_file="$(mktemp)"
  job_overview_file="$(mktemp)"
  recent_runs_file="$(mktemp)"

  build_dataset_summary "${dataset_summary_file}"
  build_state_counts "${state_counts_file}"
  build_status_counts "${status_counts_file}"
  build_job_overview "${job_overview_file}"
  build_recent_runs "${recent_runs_file}"

  printf "%b\n" "${C_CYAN}Aggregate Benchmark Dashboard${C_RESET} | $(timestamp)"
  printf "Root: %s\n" "${ROOT}"
  printf "Snapshot: %s\n" "${snapshot}"
  printf "Datasets: %d  Jobs: %d\n" "${dataset_count}" "${total_jobs}"
  printf "Progress: %s\n" "$(draw_progress_bar "${processed}" "${total_jobs}")"
  echo
  printf "%b\n" "${C_BLUE}running=${running_count}${C_RESET}  ${C_YELLOW}pending=${pending_count}${C_RESET}  ${C_RED}failed=${failed_count}${C_RESET}  ${C_GREEN}done=${done_count}${C_RESET}  ${C_CYAN}partial=${partial_count}${C_RESET}"

  if [[ "${#DASH_WARNINGS[@]}" -gt 0 ]]; then
    echo
    printf "%b\n" "${C_YELLOW}Warnings:${C_RESET}"
    for warning in "${DASH_WARNINGS[@]}"; do
      printf -- "- %s\n" "${warning}"
    done
  fi

  echo
  echo "State Counts:"
  colorize_rows_by_state_column 1 < "${state_counts_file}" | render_tsv_table

  echo
  echo "Status Counts:"
  colorize_rows_by_status_column 1 < "${status_counts_file}" | render_tsv_table

  echo
  echo "Dataset Summary:"
  render_tsv_table < "${dataset_summary_file}"

  echo
  echo "All Jobs:"
  colorize_rows_by_state_column 5 < "${job_overview_file}" | render_tsv_table

  if [[ "$(awk 'END { print NR-1 }' "${recent_runs_file}")" -gt 0 ]]; then
    echo
    if [[ "${RECENT_LIMIT}" == "0" ]]; then
      echo "All Parsed Runs:"
    else
      echo "Recent Runs (Last ${RECENT_LIMIT}):"
    fi
    render_tsv_table < "${recent_runs_file}"
  fi

  rm -f "${dataset_summary_file}" "${state_counts_file}" "${status_counts_file}" "${job_overview_file}" "${recent_runs_file}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      require_value "$1" "${2-}"
      ROOT="$2"
      shift 2
      ;;
    --out-tsv)
      require_value "$1" "${2-}"
      OUT_TSV="$2"
      shift 2
      ;;
    --no-refresh)
      REFRESH=0
      shift
      ;;
    --watch)
      if [[ $# -ge 2 ]] && [[ ! "${2-}" =~ ^- ]]; then
        WATCH_SEC="$2"
        shift 2
      else
        WATCH_SEC=20
        shift
      fi
      ;;
    --recent-limit)
      require_value "$1" "${2-}"
      RECENT_LIMIT="$2"
      shift 2
      ;;
    --no-color)
      NO_COLOR=1
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

normalize_root_layout

if [[ -z "${OUT_TSV}" ]]; then
  OUT_TSV="${ROOT}/summaries/aggregate_status_snapshot.tsv"
fi

if ! [[ "${WATCH_SEC}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --watch must be a non-negative integer." >&2
  exit 1
fi
if ! [[ "${RECENT_LIMIT}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --recent-limit must be a non-negative integer." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$(dirname "${OUT_TSV}")"

collect_dataset_dirs
configure_colors

if [[ "${WATCH_SEC}" -gt 0 ]]; then
  while true; do
    if [[ -t 1 ]]; then
      clear
    fi
    render_once
    sleep "${WATCH_SEC}"
  done
else
  render_once
fi
