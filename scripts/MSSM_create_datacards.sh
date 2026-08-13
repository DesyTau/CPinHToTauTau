#!/bin/bash
set -euo pipefail

usage() {
cat <<'EOF'
Usage:
./MSSM_create_datacards.sh CONFIG [options] [extra law options]

Options:
  --masses "M1 M2 M3"       Run only these masses
  --masses M1,M2,M3         Same, comma-separated
  --mass M                   Add one mass point; can be repeated
  --all-masses               Run the full default mass list
  --poll-interval T          Polling interval for HTCondor tasks, e.g. 30m, 1h.
                             Default: $POLL_INTERVAL if set, otherwise 5m.
  -h, --help                 Show this help

This script creates four datacards per mass point:

1. MSSM_model_D_sig_vs_Disc_ggphi_M{MASS}
   category: bdt_cat_ggphi_and_bbphi_M{MASS}
   variable: bdt_D_sig_vs_Disc_ggphi_M{MASS}
   signal:   ggphi

2. MSSM_model_D_sig_vs_Disc_bbphi_M{MASS}
   category: bdt_cat_ggphi_and_bbphi_M{MASS}
   variable: bdt_D_sig_vs_Disc_bbphi_M{MASS}
   signal:   bbphi

3. MSSM_model_D_DY_M{MASS}
   category: bdt_cat_dy_M{MASS}
   variable: bdt_D_DY_M{MASS}

4. MSSM_model_D_TT_M{MASS}
   category: bdt_cat_tt_M{MASS}
   variable: bdt_D_TT_M{MASS}

Examples:
./MSSM_create_datacards.sh 23_emu

./MSSM_create_datacards.sh 23_emu --masses "100 200 300"

./MSSM_create_datacards.sh 23_emu --masses 100,200,300

./MSSM_create_datacards.sh 23_emu --mass 100 --mass 200 --mass 300

./MSSM_create_datacards.sh 23_emu --masses "100 200" --poll-interval 1h --workers 1

POLL_INTERVAL=45m ./MSSM_create_datacards.sh 23_emu --mass 60 --workers 1
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

config_arg="$1"
shift

source ./common_run3_MSSM.sh
set_common_vars "$config_arg"

# ----------------------------------------------------------------------
# Isolate simultaneous submissions from different lxplus machines.
#
# This avoids sharing transient LAW job files and HTCondor user-log
# directories between independent script invocations.
# ----------------------------------------------------------------------

run_id="${RUN_ID:-${config_arg}_$(hostname -s)_$(date +%Y%m%d_%H%M%S)_$$}"
run_id="$(echo "$run_id" | sed 's/[^A-Za-z0-9_.-]/_/g')"

echo "[info] Run id: $run_id"

# Isolate LAW/HTCondor submission metadata if CF_JOB_BASE is used by law.cfg.
if [[ -n "${CF_JOB_BASE:-}" ]]; then
    export CF_JOB_BASE="${CF_JOB_BASE%/}/runs/${run_id}"
    mkdir -p "$CF_JOB_BASE"
    echo "[info] CF_JOB_BASE: $CF_JOB_BASE"
fi

# Isolate the HTCondor event-log layout used by the condor_history -userlog wrapper.
export CF_HTCONDOR_USERLOG_RUN_ID="$run_id"
export CF_HTCONDOR_CLEAN_SUCCESS_LOGS="${CF_HTCONDOR_CLEAN_SUCCESS_LOGS:-1}"

if [[ -n "${CF_HTCONDOR_USERLOG_DIR:-}" ]]; then
    mkdir -p "$CF_HTCONDOR_USERLOG_DIR"
    echo "[info] CF_HTCONDOR_USERLOG_DIR: $CF_HTCONDOR_USERLOG_DIR"
fi

# ----------------------------------------------------------------------
# Versions
#
# Reuse all existing outputs up to ProduceColumns.
# Recreate only the histogram layer and the final datacards.
# ----------------------------------------------------------------------

upstream_version=dust_dev
hist_version=dust_dev
# version="dust_dev"
# Sparse polling by default.
poll_interval="${POLL_INTERVAL:-5m}"

default_masses=(
    60 65 70 75 80 85 90 95
    100 105 110 115 120 125 130 135 140
    160 180 200 250 300 350
    400 450 500 600
    700 800 900 1000 1100
    1200 1400 1600 1800
    2000 2300
    2600 2900 3200 3500
)

# ----------------------------------------------------------------------
# Tasks that should keep using the existing desy_dev outputs.
# ----------------------------------------------------------------------

upstream_tasks=(
    cf.CalibrateEvents
    cf.SelectEvents
    cf.ReduceEvents
    cf.MergeReducedEvents
    cf.MergeSelectionStats
    cf.ProvideReducedEvents
    cf.ProduceColumns
)

# ----------------------------------------------------------------------
# Tasks affected by the mass-block histogram grouping.
# ----------------------------------------------------------------------

hist_tasks=(
    cf.CreateHistograms
    cf.MergeHistograms
    cf.MergeShiftedHistograms
)

masses=()
extra_args=()

add_masses_from_string() {
    local raw="$1"

    # Allow both comma-separated and space-separated input.
    raw="${raw//,/ }"

    local m
    for m in $raw; do
        if [[ ! "$m" =~ ^[0-9]+$ ]]; then
            echo "[error] Invalid mass value: $m" >&2
            exit 1
        fi

        masses+=("$m")
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --masses|-m)
            if [[ $# -lt 2 ]]; then
                echo "[error] Missing argument after $1" >&2
                exit 1
            fi

            add_masses_from_string "$2"
            shift 2
            ;;

        --mass)
            if [[ $# -lt 2 ]]; then
                echo "[error] Missing argument after $1" >&2
                exit 1
            fi

            add_masses_from_string "$2"
            shift 2
            ;;

        --all-masses)
            masses=("${default_masses[@]}")
            shift
            ;;

        --poll-interval)
            if [[ $# -lt 2 ]]; then
                echo "[error] Missing argument after $1" >&2
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

if [[ ${#masses[@]} -eq 0 ]]; then
    masses=("${default_masses[@]}")
fi

echo "[info] Config: $config"
echo "[info] Workflow: $workflow"
echo "[info] Upstream version: $upstream_version"
echo "[info] Histogram/datacard version: $hist_version"
echo "[info] Poll interval: $poll_interval"
echo "[info] Masses to run: ${masses[*]}"

for m in "${masses[@]}"; do
    inference_models=(
        "MSSM_model_D_sig_vs_Disc_ggphi_M${m}"
        "MSSM_model_D_sig_vs_Disc_bbphi_M${m}"
        "MSSM_model_D_DY_M${m}"
        "MSSM_model_D_TT_M${m}"
    )

    echo
    echo "[info] Running mass M${m}"
    echo "[info] Inference models:"
    printf '  - %s\n' "${inference_models[@]}"

    for inference_model in "${inference_models[@]}"; do
        args=(
            --config "$config"

            --pilot True

            # Version of cf.CreateDatacards itself.
            --version "$hist_version"

            --inference-model "$inference_model"
            --hist-hooks qcd
        )

        # --------------------------------------------------------------
        # Reuse existing upstream outputs from desy_dev.
        # --------------------------------------------------------------

        for task in "${upstream_tasks[@]}"; do
            args+=(
                "--${task}-version" "$upstream_version"
                "--${task}-workflow" "$workflow"
                "--${task}-poll-interval" "$poll_interval"
            )
        done

        # --------------------------------------------------------------
        # Recreate the histogram layer with the new mass-block version.
        # --------------------------------------------------------------

        for task in "${hist_tasks[@]}"; do
            args+=(
                "--${task}-version" "$hist_version"
                "--${task}-workflow" "$workflow"
                "--${task}-poll-interval" "$poll_interval"
            )
        done

        args+=("${extra_args[@]}")

        echo
        echo "[info] Running inference model: $inference_model"
        echo "[info] Upstream version: $upstream_version"
        echo "[info] Histogram/datacard version: $hist_version"
        echo "[info] Poll interval: $poll_interval"

        echo law run cf.CreateDatacards "${args[@]}"
        law run cf.CreateDatacards "${args[@]}"
    done
done
