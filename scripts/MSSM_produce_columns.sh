#!/usr/bin/env bash
set -euo pipefail

source ./common_run3_MSSM.sh

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 CONFIG [--poll-interval T] [extra law options]"
    exit 1
fi

set_common_vars "$1"
shift

# ----------------------------------------------------------------------
# Poll interval
# ----------------------------------------------------------------------

# Default polling interval for HTCondor tasks.
# Can be overridden either with:
#
#   POLL_INTERVAL=45m ./script.sh 23_emu
#
# or:
#
#   ./script.sh 23_emu --poll-interval 1h
#
poll_interval="${POLL_INTERVAL:-5m}"

extra_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --poll-interval)
            if [[ $# -lt 2 ]]; then
                echo "[error] Missing argument after --poll-interval" >&2
                exit 1
            fi

            poll_interval="$2"
            shift 2
            ;;

        --)
            shift
            extra_args+=("$@")
            break
            ;;

        *)
            extra_args+=("$1")
            shift
            ;;
    esac
done

# ----------------------------------------------------------------------
# HTCondor history workaround
# ----------------------------------------------------------------------

condor_history_wrapper="${CONDOR_HISTORY_WRAPPER:-${PWD}/condor_history_userlog_wrapper.sh}"

if [[ ! -x "$condor_history_wrapper" ]]; then
    echo "[error] condor history wrapper not found or not executable:" >&2
    echo "        $condor_history_wrapper" >&2
    echo "Run:" >&2
    echo "        chmod +x $condor_history_wrapper" >&2
    exit 1
fi

export LAW__job__htcondor_cmd_history="$condor_history_wrapper"
export LAW__job__htcondor_chunk_size_query="${LAW_HTCONDOR_CHUNK_SIZE_QUERY:-1}"
export LAW__job__htcondor_job_query_timeout="${LAW_HTCONDOR_JOB_QUERY_TIMEOUT:-30s}"

# ----------------------------------------------------------------------
# LAW arguments
# ----------------------------------------------------------------------

args=(
    --configs "$config"
    --datasets "$datasets"

    --cf.CalibrateEvents-workflow "$workflow"
    --cf.CalibrateEvents-version "$version"
    --cf.CalibrateEvents-poll-interval "$poll_interval"

    --cf.SelectEvents-workflow "$workflow"
    --cf.SelectEvents-version "$version"
    --cf.SelectEvents-poll-interval "$poll_interval"

    --cf.MergeSelectionStats-workflow "$workflow"
    --cf.MergeSelectionStats-version "$version"
    --cf.MergeSelectionStats-poll-interval "$poll_interval"

    --cf.ReduceEvents-workflow "$workflow"
    --cf.ReduceEvents-version "$version"
    --cf.ReduceEvents-poll-interval "$poll_interval"

    --cf.MergeReducedEvents-workflow "$workflow"
    --cf.MergeReducedEvents-version "$version"
    --cf.MergeReducedEvents-poll-interval "$poll_interval"

    --cf.ProvideReducedEvents-workflow "$workflow"
    --cf.ProvideReducedEvents-version "$version"
    --cf.ProvideReducedEvents-poll-interval "$poll_interval"

    --cf.ProduceColumns-workflow "$workflow"
    --cf.ProduceColumns-version "$version"
    --cf.ProduceColumns-poll-interval "$poll_interval"

    --version "$version"

    "${extra_args[@]}"
)

echo "[info] Config: $config"
echo "[info] Workflow: $workflow"
echo "[info] Version: $version"
echo "[info] Poll interval: $poll_interval"

echo
echo law run cf.ProduceColumnsWrapper "${args[@]}"
law run cf.ProduceColumnsWrapper "${args[@]}"