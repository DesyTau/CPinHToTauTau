#!/usr/bin/env bash
set -euo pipefail

source ./common_run3_MSSM.sh

set_common_vars "$1"

# HTCondor history workaround
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

args=(
    --configs "$config"
    --datasets "$datasets"

    --cf.CalibrateEvents-workflow "$workflow"
    --cf.CalibrateEvents-version "$version"

    --cf.SelectEvents-workflow "$workflow"
    --cf.SelectEvents-version "$version"

    --cf.MergeSelectionStats-workflow "$workflow"
    --cf.MergeSelectionStats-version "$version"

    --cf.ReduceEvents-workflow "$workflow"
    --cf.ReduceEvents-version "$version"

    --cf.MergeReducedEvents-workflow "$workflow"
    --cf.MergeReducedEvents-version "$version"

    --cf.ProvideReducedEvents-workflow "$workflow"
    --cf.ProvideReducedEvents-version "$version"

    --cf.ProduceColumns-workflow "$workflow"
    --cf.ProduceColumns-version "$version"

    --version "$version"
    "${@:2}"
)

echo law run cf.ProduceColumnsWrapper "${args[@]}"
law run cf.ProduceColumnsWrapper "${args[@]}"