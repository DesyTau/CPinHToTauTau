#!/bin/bash

set -euo pipefail

source ./common_run3_MSSM.sh

# The following function defines:
# config, processes, version, datasets, categories, variables
set_common_vars "$1"

all_shift_sources=(
    unclustered
    recoilresp
    recoilres
    CMS_PS_FSR
    CMS_PS_ISR
    CMS_Scale_muF
    CMS_Scale_muR
    btag_weight_hf
    btag_weight_lf
    btag_weight_hfstats1
    btag_weight_hfstats2
    btag_weight_lfstats1
    btag_weight_lfstats2
    btag_weight_cferr1
    btag_weight_cferr2
    jec_Total
    jer
    Trigger_SF_weight
    electron_weight
    muon_weight
    zpt_weight
    pu_weight
    top_pt_weight
)

# Convert comma-separated or space-separated process list into bash array
process_list=($(echo "$processes" | tr ',' ' '))

MAX_JOBS=6
running_jobs=0

run_one_process() {
    local proc="$1"
    shift

    # Skip data processes only
    if [[ "$proc" == *data* || "$proc" == *Data* || "$proc" == *DATA* ]]; then
        echo "Skipping data process: $proc"
        return 0
    fi

    local shift_sources=("${all_shift_sources[@]}")

    local shift_sources_csv
    shift_sources_csv=$(IFS=, ; echo "${shift_sources[*]}")

    args=(
        --config "$config"
        --processes "$proc"
        --datasets "$datasets"
        --version "$version"
        --categories "$categories"
        --variables "$variables"
        --shift-sources "$shift_sources_csv"
        --file-types png
        --general-settings "cms-label=pw,yscale=log"
        "${@:2}"
    )

    echo
    echo "Running process: $proc"
    echo "Shift sources: $shift_sources_csv"
    echo "law run cf.PlotShiftedVariablesPerShift1D ${args[*]}"

    law run cf.PlotShiftedVariablesPerShift1D "${args[@]}"
}

for proc in "${process_list[@]}"; do

    run_one_process "$proc" "${@:2}" &

    running_jobs=$((running_jobs + 1))

    if (( running_jobs >= MAX_JOBS )); then
        wait -n
        running_jobs=$((running_jobs - 1))
    fi

done

wait

echo
echo "All per-process shifted-variable plots finished."