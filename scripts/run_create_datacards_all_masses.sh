#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run cf.CreateDatacards for all MSSM mass points for one era/config.
#
# Usage:
#   ./run_create_datacards_all_masses.sh ERA_OR_CONFIG [extra law args...]
#
# Example:
#   MAX_PARALLEL_MASSES=4 ./run_create_datacards_all_masses.sh run3_2022_preEE_emu --workers 8
#
# Useful environment variables:
#   MAX_PARALLEL_MASSES=4
#       Number of mass points submitted in parallel.
#
#   KEEP_FULL_LOGS_ON_SUCCESS=0
#       If 0, delete full verbose logs for successful masses.
#       If 1, gzip and keep full logs also for successful masses.
#
#   CF_CONFIG_OPTION=--config
#       Use --config for older Columnflow setups.
#       Use --configs for newer multi-config Columnflow setups if needed.
# ============================================================

if (( $# < 1 )); then
    echo "Usage: $0 ERA_OR_CONFIG [extra law args...]" >&2
    exit 1
fi

source ./common_run3_MSSM.sh
set_common_vars "$1"

: "${config:?ERROR: set_common_vars did not define config}"
: "${workflow:?ERROR: set_common_vars did not define workflow}"

era_arg="$1"
version="desy_dev"

# Prefer the resolved Columnflow config name if set_common_vars defines it.
era_label="${config:-$era_arg}"

# Filename-safe era/config label.
era_label_safe="$(echo "$era_label" | sed 's#[^A-Za-z0-9_.-]#_#g')"

cf_config_option="${CF_CONFIG_OPTION:---config}"
max_parallel="${MAX_PARALLEL_MASSES:-4}"
keep_full_logs_on_success="${KEEP_FULL_LOGS_ON_SUCCESS:-0}"

if ! [[ "$max_parallel" =~ ^[0-9]+$ ]] || (( max_parallel < 1 )); then
    echo "[error] MAX_PARALLEL_MASSES must be a positive integer, got: ${max_parallel}" >&2
    exit 1
fi

if [[ "$keep_full_logs_on_success" != "0" && "$keep_full_logs_on_success" != "1" ]]; then
    echo "[error] KEEP_FULL_LOGS_ON_SUCCESS must be 0 or 1, got: ${keep_full_logs_on_success}" >&2
    exit 1
fi

masses=(
    60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140
    160 180 200 250 300 350 400 450 500 600 700 800 900 1000 1100
    1200 1400 1600 1800 2000 2300 2600 2900 3200 3500
)

extra_args=("${@:2}")

log_base="logs_create_datacards_${version}"
log_dir="${log_base}/${era_label_safe}"
mkdir -p "$log_dir"

print_command() {
    printf "law run cf.CreateDatacards"
    printf " %q" "$@"
    printf "\n"
}

run_one_mass() {
    local m="$1"

    local args=(
        "$cf_config_option" "$config"

        --cf.CalibrateEvents-version "$version"
        --cf.CalibrateEvents-workflow "$workflow"

        --cf.SelectEvents-version "$version"
        --cf.SelectEvents-workflow "$workflow"

        --cf.ReduceEvents-version "$version"
        --cf.ReduceEvents-workflow "$workflow"

        --cf.MergeReducedEvents-version "$version"
        --cf.MergeReducedEvents-workflow "$workflow"

        --cf.MergeSelectionStats-version "$version"
        --cf.MergeSelectionStats-workflow "$workflow"

        --cf.ProvideReducedEvents-version "$version"
        --cf.ProvideReducedEvents-workflow "$workflow"

        --cf.ProduceColumns-version "$version"
        --cf.ProduceColumns-workflow "$workflow"

        --cf.CreateHistograms-version "$version"
        --cf.CreateHistograms-workflow "$workflow"

        --cf.MergeHistograms-version "$version"
        --cf.MergeHistograms-workflow "$workflow"

        --cf.MergeShiftedHistograms-version "$version"
        --cf.MergeShiftedHistograms-workflow "$workflow"

        --pilot True
        --version "$version"

        --inference-model "MSSM_model_M${m}"
        --hist-hooks qcd

        "${extra_args[@]}"
    )

    echo "[${era_label_safe}][M${m}] starting"
    print_command "${args[@]}"
    law run cf.CreateDatacards "${args[@]}"
    echo "[${era_label_safe}][M${m}] finished"
}

write_success_compact_log() {
    local m="$1"
    local full_log_file="$2"
    local compact_log_file="$3"

    {
        echo "[${era_label_safe}][M${m}] finished successfully"
        echo
        echo "Compact summary from full log:"
        grep -E "^\[${era_label_safe}\]\[M${m}\]|finished|Finished|complete|Complete|DONE|Done|success|Success" "$full_log_file" | tail -n 40 || true
        echo
        echo "Last 30 lines:"
        tail -n 30 "$full_log_file" || true
    } >"$compact_log_file"
}

write_failure_compact_log() {
    local m="$1"
    local status="$2"
    local full_log_file="$3"
    local compact_log_file="$4"

    {
        echo "[${era_label_safe}][M${m}] failed with exit code ${status}"
        echo
        echo "Relevant error/warning lines:"
        grep -E "ERROR|Error|error|FAILED|Failed|failed|Traceback|Exception|WARNING|Warning|warning|missing|Missing|No such file|not found|segmentation|Segmentation" "$full_log_file" | tail -n 160 || true
        echo
        echo "Last 160 lines of full log:"
        tail -n 160 "$full_log_file" || true
        echo
        echo "Full verbose log:"
        echo "${full_log_file}.gz"
    } >"$compact_log_file"
}

echo "Submitting cf.CreateDatacards for all mass points"
echo "Era/config:             ${era_label_safe}"
echo "Resolved config:        ${config}"
echo "Workflow:               ${workflow}"
echo "Version:                ${version}"
echo "Config option:          ${cf_config_option}"
echo "Mass points:            ${#masses[@]}"
echo "Max parallel masses:    ${max_parallel}"
echo "Compact logs:           ${log_dir}"
echo "Keep full logs success: ${keep_full_logs_on_success}"
echo

running=0
failed=0

for m in "${masses[@]}"; do
    compact_log_file="${log_dir}/create_datacards_${era_label_safe}_M${m}.log"
    full_log_file="${log_dir}/create_datacards_${era_label_safe}_M${m}.full.log"

    (
        set +e

        run_one_mass "$m" >"$full_log_file" 2>&1
        status=$?

        if (( status == 0 )); then
            write_success_compact_log "$m" "$full_log_file" "$compact_log_file"

            if (( keep_full_logs_on_success == 1 )); then
                gzip -f "$full_log_file"
            else
                rm -f "$full_log_file"
            fi
        else
            write_failure_compact_log "$m" "$status" "$full_log_file" "$compact_log_file"
            gzip -f "$full_log_file"
            exit "$status"
        fi
    ) &

    running=$((running + 1))
    echo "[${era_label_safe}][M${m}] submitted, log: ${compact_log_file}"

    if (( running >= max_parallel )); then
        if ! wait -n; then
            failed=1
            echo "[warn] at least one mass failed for ${era_label_safe}; check ${log_dir}" >&2
        fi
        running=$((running - 1))
    fi
done

while (( running > 0 )); do
    if ! wait -n; then
        failed=1
        echo "[warn] at least one mass failed for ${era_label_safe}; check ${log_dir}" >&2
    fi
    running=$((running - 1))
done

echo

if (( failed != 0 )); then
    echo "[error] Some mass points failed for ${era_label_safe}."
    echo "Inspect compact logs:"
    echo "  ${log_dir}/create_datacards_${era_label_safe}_M*.log"
    echo
    echo "Failed masses:"
    grep -H "failed with exit code" "${log_dir}"/create_datacards_"${era_label_safe}"_M*.log || true
    exit 1
fi

echo "[done] All mass points finished successfully for ${era_label_safe}."