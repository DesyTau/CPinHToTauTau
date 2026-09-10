#!/usr/bin/env bash
set -euo pipefail

usage() {
cat <<'EOF'
Usage:
  ./MSSM_create_datacards_multi_eras.sh "ERA1 ERA2 ..." [options for MSSM_create_datacards.sh]

Examples:
  MAX_PARALLEL_ERAS=2 ./MSSM_create_datacards_multi_eras.sh \
    "22_emu 22EE_emu" \
    --mass 60 --poll-interval 30m --workers 1 --local-scheduler

  MAX_PARALLEL_ERAS=4 ./MSSM_create_datacards_multi_eras.sh \
    "22_emu 22EE_emu 23_emu 23BPix_emu" \
    --masses "60 65 70" --poll-interval 1h --workers 1 --local-scheduler

Environment:
  MAX_PARALLEL_ERAS              Maximum number of eras submitted in parallel. Default: 2
  CF_JOB_BASE                    Base directory for LAW job metadata
  CF_HTCONDOR_USERLOG_DIR        Base directory for HTCondor user logs
  CF_HTCONDOR_CLEAN_SUCCESS_LOGS Default: 1
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

eras_raw="$1"
shift

# Allow both comma-separated and space-separated era lists.
eras_raw="${eras_raw//,/ }"
read -r -a eras <<< "$eras_raw"

if [[ ${#eras[@]} -eq 0 ]]; then
    echo "[error] No eras provided" >&2
    exit 1
fi

max_parallel="${MAX_PARALLEL_ERAS:-2}"

if ! [[ "$max_parallel" =~ ^[0-9]+$ ]] || [[ "$max_parallel" -lt 1 ]]; then
    echo "[error] Invalid MAX_PARALLEL_ERAS: $max_parallel" >&2
    exit 1
fi

# Keep original bases and derive one subdirectory per era/run.
base_cf_job_base="${CF_JOB_BASE:-}"
base_userlog_dir="${CF_HTCONDOR_USERLOG_DIR:-$HOME/htcondor_userlogs_columnflow}"

export CF_HTCONDOR_USERLOG_DIR="$base_userlog_dir"
export CF_HTCONDOR_CLEAN_SUCCESS_LOGS="${CF_HTCONDOR_CLEAN_SUCCESS_LOGS:-1}"

mkdir -p "$CF_HTCONDOR_USERLOG_DIR"
mkdir -p logs

common_args=("$@")

has_arg() {
    local needle="$1"
    local arg
    for arg in "${common_args[@]}"; do
        if [[ "$arg" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

# Strongly recommended when launching several LAW processes from the same shell.
if ! has_arg "--local-scheduler"; then
    common_args+=(--local-scheduler)
fi

# Keep each era process light. Override by passing --workers yourself.
if ! has_arg "--workers"; then
    common_args+=(--workers 1)
fi

echo "[info] Eras: ${eras[*]}"
echo "[info] MAX_PARALLEL_ERAS: $max_parallel"
echo "[info] CF_HTCONDOR_USERLOG_DIR: $CF_HTCONDOR_USERLOG_DIR"
if [[ -n "$base_cf_job_base" ]]; then
    echo "[info] Base CF_JOB_BASE: $base_cf_job_base"
else
    echo "[info] CF_JOB_BASE is not set in this shell"
fi
echo "[info] Common args: ${common_args[*]}"

pids=()

for era in "${eras[@]}"; do
    while [[ "$(jobs -rp | wc -l)" -ge "$max_parallel" ]]; do
        sleep 20
    done

    run_id="${era}_$(hostname -s)_$(date +%Y%m%d_%H%M%S)_$$"
    run_id="$(echo "$run_id" | sed 's/[^A-Za-z0-9_.-]/_/g')"

    log_file="logs/${run_id}.log"

    echo
    echo "[info] Starting era: $era"
    echo "[info] Run id: $run_id"
    echo "[info] Log: $log_file"

    (
        set -euo pipefail

        export RUN_ID="$run_id"
        export CF_HTCONDOR_USERLOG_RUN_ID="$run_id"

        # Isolate LAW/HTCondor job metadata when CF_JOB_BASE is configured.
        if [[ -n "$base_cf_job_base" ]]; then
            export CF_JOB_BASE="${base_cf_job_base%/}/runs/${run_id}"
            mkdir -p "$CF_JOB_BASE"
            echo "[child:$era] CF_JOB_BASE=$CF_JOB_BASE"
        fi

        mkdir -p "$CF_HTCONDOR_USERLOG_DIR"

        echo "[child:$era] started at $(date)"
        echo "[child:$era] host: $(hostname -f)"
        echo "[child:$era] RUN_ID=$RUN_ID"
        echo "[child:$era] CF_HTCONDOR_USERLOG_RUN_ID=$CF_HTCONDOR_USERLOG_RUN_ID"
        echo "[child:$era] CF_HTCONDOR_USERLOG_DIR=$CF_HTCONDOR_USERLOG_DIR"
        echo "[child:$era] command: ./MSSM_create_datacards.sh $era ${common_args[*]}"

        ./MSSM_create_datacards.sh "$era" "${common_args[@]}"

        echo "[child:$era] finished at $(date)"
    ) > "$log_file" 2>&1 &

    pids+=("$!")
done

echo
echo "[info] Submitted ${#pids[@]} era process(es). Waiting..."

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "[error] One era process failed: pid=$pid" >&2
        failed=1
    fi
done

if [[ "$failed" -ne 0 ]]; then
    echo "[error] At least one era failed. Check logs/." >&2
    exit 1
fi

echo "[info] All era processes finished successfully."