#!/bin/bash

source ./common_run3_MSSM.sh


# ============================================================================
# Common configuration
# ============================================================================

if [[ -z "$1" ]]; then
    echo "ERROR: no configuration option provided"
    echo
    echo "Usage:"
    echo "  $0 <configuration> [extra law arguments]"
    echo
    echo "Examples:"
    echo "  $0 22_emu_data"
    echo "  $0 22_emu_signal"
    echo "  $0 22_emu_bkg"
    echo "  $0 22_emu"
    echo "  $0 22and23_emu"
    exit 1
fi


if ! set_common_vars "$1"; then
    exit 1
fi


configuration_option="$1"

extra_args=("${@:2}")


# ============================================================================
# Sanity checks
# ============================================================================

if [[ -z "$config" ]]; then
    echo "ERROR: config is empty"
    exit 1
fi

if [[ -z "$version" ]]; then
    echo "ERROR: version is empty"
    exit 1
fi


# ============================================================================
# Determine requested production mode
#
# Examples:
#
#   22_emu_data     -> data only
#   22_emu_signal   -> signals only
#   22_emu_bkg      -> backgrounds only
#
#   22_emu          -> data + signals + backgrounds
#   22and23_emu     -> data + signals + backgrounds for all eras
# ============================================================================

case "$configuration_option" in

    *_data)
        production_mode="data"
        ;;

    *_signal)
        production_mode="signal"
        ;;

    *_bkg)
        production_mode="bkg"
        ;;

    22and23_emu|22_emu|22EE_emu|23_emu|23BPix_emu)
        production_mode="all"
        ;;

    *)
        echo "ERROR: unsupported ProduceColumns configuration:"
        echo "  $configuration_option"
        exit 1
        ;;

esac


# ============================================================================
# JEC sources
#
# These are required only for MC.
# ============================================================================

jec_sources=(
    jec_Regrouped_Absolute
    jec_Regrouped_BBEC1
    jec_Regrouped_EC2
    jec_Regrouped_HF
    jec_Regrouped_RelativeBal
    jec_Regrouped_FlavorQCD
)


# -------------------------------------------------------------------------
# Add era-dependent JEC sources
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
            echo "ERROR: cannot determine JEC era from config:"
            echo "  $cfg"
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


# -------------------------------------------------------------------------
# Remove duplicated JEC sources
# -------------------------------------------------------------------------

declare -A seen_jec
unique_jec_sources=()

for source in "${jec_sources[@]}"; do

    if [[ -z "${seen_jec[$source]}" ]]; then

        unique_jec_sources+=("$source")

        seen_jec[$source]=1

    fi

done


# ============================================================================
# MC ProduceColumns shifts
#
# Separate ProduceColumns outputs are needed for the kinematic shifts used
# later by MergeShiftedHistograms:
#
#   nominal
#
#   JEC
#   JER
#
#   unclustered MET
#   recoil response
#   recoil resolution
#
# Weight-only systematics are not included here.
# ============================================================================

mc_shifts=(
    nominal

    jer_up
    jer_down

    unclustered_up
    unclustered_down

    recoilresp_up
    recoilresp_down

    recoilres_up
    recoilres_down
)


for source in "${unique_jec_sources[@]}"; do

    mc_shifts+=(
        "${source}_up"
        "${source}_down"
    )

done


# -------------------------------------------------------------------------
# Remove duplicate MC shifts
# -------------------------------------------------------------------------

declare -A seen_shift
unique_mc_shifts=()

for shift in "${mc_shifts[@]}"; do

    if [[ -z "${seen_shift[$shift]}" ]]; then

        unique_mc_shifts+=("$shift")

        seen_shift[$shift]=1

    fi

done


mc_shifts_csv="$(
    IFS=,
    echo "${unique_mc_shifts[*]}"
)"


if [[ -z "$mc_shifts_csv" ]]; then
    echo "ERROR: MC shift list is empty"
    exit 1
fi


# ============================================================================
# Dataset lists for the combined modes
#
# When using one of:
#
#   22_emu
#   22EE_emu
#   23_emu
#   23BPix_emu
#   22and23_emu
#
# we split the production into data / signal / background ourselves.
#
# For *_data, *_signal and *_bkg, the dataset list defined by
# common_run3_MSSM.sh is used directly.
# ============================================================================

if [[ "$production_mode" == "all" ]]; then

    case "$configuration_option" in

        22_emu)

            data_datasets="${data_egamma_2022preEE}${data_mu_2022preEE}"

            signal_datasets="${signal_all}"

            background_datasets="${bkgs}"
            ;;


        22EE_emu)

            data_datasets="${data_egamma_2022postEE}${data_mu_2022postEE}"

            signal_datasets="${signal_all}"

            background_datasets="${bkgs}"
            ;;


        23_emu)

            data_datasets="${data_egamma_2023preBPix}${data_mu_2023preBPix}"

            signal_datasets="${signal_all}"

            background_datasets="${bkgs}"
            ;;


        23BPix_emu)

            data_datasets="${data_egamma_2023postBPix}${data_mu_2023postBPix}"

            signal_datasets="${signal_all}"

            background_datasets="${bkgs}"
            ;;


        22and23_emu)

            data_datasets="${data_egamma_2022preEE}${data_mu_2022preEE}:"
            data_datasets+="${data_egamma_2022postEE}${data_mu_2022postEE}:"
            data_datasets+="${data_egamma_2023preBPix}${data_mu_2023preBPix}:"
            data_datasets+="${data_egamma_2023postBPix}${data_mu_2023postBPix}"


            signal_datasets="${signal_all}:"
            signal_datasets+="${signal_all}:"
            signal_datasets+="${signal_all}:"
            signal_datasets+="${signal_all}"


            background_datasets="${bkgs}:"
            background_datasets+="${bkgs}:"
            background_datasets+="${bkgs}:"
            background_datasets+="${bkgs}"
            ;;

    esac

fi


# ============================================================================
# Function to run ProduceColumns
# ============================================================================

run_produce_columns() {

    sample_type="$1"
    selected_datasets="$2"
    selected_shifts="$3"


    if [[ -z "$selected_datasets" ]]; then
        echo "ERROR: dataset list for $sample_type is empty"
        exit 1
    fi


    # -------------------------------------------------------------------------
    # Separate LAW controller metadata
    # -------------------------------------------------------------------------

    export CF_JOB_BASE="${CF_DATA}/jobs/${configuration_option}/${version}/produce_columns_${sample_type}"

    mkdir -p "$CF_JOB_BASE"


    # -------------------------------------------------------------------------
    # Arguments
    # -------------------------------------------------------------------------

    args=(
        --configs "$config"

        --datasets "$selected_datasets"

        --shifts "$selected_shifts"

        --version "$version"

        --calibrators "main"
        --selector "main"
        --producers "main"

        --workflow "htcondor"

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
    echo "ProduceColumns: $sample_type"
    echo "======================================================================"
    echo "Configuration option: $configuration_option"
    echo "Configs:              $config"
    echo "Version:              $version"
    echo "Datasets:             $selected_datasets"
    echo "Shifts:               $selected_shifts"
    echo "Calibrators:          main"
    echo "Selector:             main"
    echo "Producer:             main"
    echo "Workflow:             htcondor"
    echo "Retries:              10"
    echo "CF_JOB_BASE:          $CF_JOB_BASE"
    echo "======================================================================"
    echo


    echo law run cf.ProduceColumnsWrapper "${args[@]}"
    echo


    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------

    law run cf.ProduceColumnsWrapper "${args[@]}"

    status=$?


    if [[ $status -ne 0 ]]; then

        echo
        echo "ERROR: ProduceColumns failed for $sample_type"
        echo

        exit $status

    fi


    echo
    echo "======================================================================"
    echo "ProduceColumns completed for $sample_type"
    echo "======================================================================"
    echo
}


# ============================================================================
# Run requested production
# ============================================================================

case "$production_mode" in


# -------------------------------------------------------------------------
# DATA
# -------------------------------------------------------------------------

data)

    run_produce_columns \
        "data" \
        "$datasets" \
        "nominal" \
        "5"

    ;;


# -------------------------------------------------------------------------
# SIGNAL
# -------------------------------------------------------------------------

signal)

    run_produce_columns \
        "signals" \
        "$datasets" \
        "$mc_shifts_csv" \
        "10"

    ;;


# -------------------------------------------------------------------------
# BACKGROUNDS
# -------------------------------------------------------------------------

bkg)

    run_produce_columns \
        "backgrounds" \
        "$datasets" \
        "$mc_shifts_csv" \
        "25"

    ;;


# -------------------------------------------------------------------------
# ALL
# -------------------------------------------------------------------------

all)

    run_produce_columns \
        "data" \
        "$data_datasets" \
        "nominal" \
        "5"


    run_produce_columns \
        "signals" \
        "$signal_datasets" \
        "$mc_shifts_csv" \
        "10"


    run_produce_columns \
        "backgrounds" \
        "$background_datasets" \
        "$mc_shifts_csv" \
        "25"

    ;;

esac


# ============================================================================
# Done
# ============================================================================

echo
echo "======================================================================"
echo "Requested ProduceColumns production completed successfully"
echo "======================================================================"
echo "Configuration option: $configuration_option"
echo "Production mode:      $production_mode"
echo "Version:              $version"
echo "======================================================================"
echo