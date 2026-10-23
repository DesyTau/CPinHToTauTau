#!/bin/bash

source ./common_run3_MSSM.sh

# -------------------------------------------------------------------------
# Common configuration
# -------------------------------------------------------------------------

if [[ -z "$1" ]]; then
    echo "ERROR: no configuration option provided"
    echo
    echo "Usage:"
    echo "  $0 <configuration> [extra law arguments]"
    echo
    echo "Available options:"
    echo "  22and23_emu"
    echo "  22_emu"
    echo "  22EE_emu"
    echo "  23_emu"
    echo "  23BPix_emu"
    exit 1
fi

if ! set_common_vars "$1"; then
    exit 1
fi

extra_args=("${@:2}")


# -------------------------------------------------------------------------
# Sanity checks
# -------------------------------------------------------------------------

if [[ -z "$config" ]]; then
    echo "ERROR: config is empty"
    exit 1
fi

if [[ -z "$datasets" ]]; then
    echo "ERROR: datasets is empty"
    exit 1
fi

if [[ -z "$version" ]]; then
    echo "ERROR: version is empty"
    exit 1
fi


# ============================================================================
# JEC sources
# ============================================================================

# Sources common to all Run 3 eras
jec_sources=(
    jec_Regrouped_Absolute
    jec_Regrouped_BBEC1
    jec_Regrouped_EC2
    jec_Regrouped_HF
    jec_Regrouped_RelativeBal
    jec_Regrouped_FlavorQCD
)


# -------------------------------------------------------------------------
# Add era-dependent JEC sources for all requested configurations
# -------------------------------------------------------------------------

IFS=',' read -ra config_list <<< "$config"

for cfg in "${config_list[@]}"; do

    case "$cfg" in
        *2022_preEE*)
            jec_era="2022"
            ;;
        *2022_postEE*)
            jec_era="2022EE"
            ;;
        *2023_preBPix*)
            jec_era="2023"
            ;;
        *2023_postBPix*)
            jec_era="2023BPix"
            ;;
        *)
            echo "ERROR: cannot determine JEC era from config: $cfg"
            exit 1
            ;;
    esac

    jec_sources+=(
        "jec_Regrouped_Absolute_${jec_era}"
        "jec_Regrouped_BBEC1_${jec_era}"
        "jec_Regrouped_EC2_${jec_era}"
        "jec_Regrouped_HF_${jec_era}"
        "jec_Regrouped_RelativeSample_${jec_era}"
    )

done


# ============================================================================
# Selection shifts
#
# Selection must be rerun only for:
#
#   - nominal
#   - all JEC variations
#   - JER up/down
#
# Weight systematics, unclustered MET and recoil systematics do not require
# a separate SelectEvents task.
# ============================================================================

selection_shifts=(
    nominal
    jer_up
    jer_down
)

for source in "${jec_sources[@]}"; do
    selection_shifts+=(
        "${source}_up"
        "${source}_down"
    )
done


# -------------------------------------------------------------------------
# Remove possible duplicates
#
# This is useful when multiple requested configs resolve to the same JEC era.
# -------------------------------------------------------------------------

declare -A seen_shifts
unique_selection_shifts=()

for shift in "${selection_shifts[@]}"; do
    if [[ -z "${seen_shifts[$shift]}" ]]; then
        unique_selection_shifts+=("$shift")
        seen_shifts[$shift]=1
    fi
done

selection_shifts_csv="$(
    IFS=,
    echo "${unique_selection_shifts[*]}"
)"


# -------------------------------------------------------------------------
# Final sanity check
# -------------------------------------------------------------------------

if [[ -z "$selection_shifts_csv" ]]; then
    echo "ERROR: selection_shifts_csv is empty"
    exit 1
fi


# ============================================================================
# Schedule SelectEvents
# ============================================================================

args=(
    --configs "$config"
    --datasets "$datasets"
    --version "$version"
    --calibrators "main"
    --selector "main"
    --shifts "$selection_shifts_csv"
    --poll-interval "5m"
    --pilot "True"
    --retries "10"

    "${extra_args[@]}"
)


# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------

echo
echo "======================================================================"
echo "SelectEvents production"
echo "======================================================================"
echo "Configuration option: $1"
echo "Configs:              $config"
echo "Datasets:             $datasets"
echo "Version:              $version"
echo "Calibrators:          main"
echo "Selector:             main"
echo "Workflow:             htcondor"
echo "Retries:              10"
echo
echo "Selection shifts:"
for shift in "${unique_selection_shifts[@]}"; do
    echo "  - $shift"
done
echo
echo "Number of selection shifts: ${#unique_selection_shifts[@]}"
echo "======================================================================"
echo

echo law run cf.SelectEventsWrapper "${args[@]}"
echo


# -------------------------------------------------------------------------
# Run
# -------------------------------------------------------------------------

law run cf.SelectEventsWrapper "${args[@]}"

status=$?

if [[ $status -ne 0 ]]; then
    echo
    echo "ERROR: SelectEvents production failed"
    echo
    exit $status
fi


echo
echo "======================================================================"
echo "SelectEvents production completed"
echo "======================================================================"