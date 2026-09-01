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

configuration_option="$1"

extra_args=("${@:2}")


# -------------------------------------------------------------------------
# Analysis channel
# -------------------------------------------------------------------------

channel="emu"


# -------------------------------------------------------------------------
# Background datasets
#
# Start from all configured datasets and remove:
#
#   - data
#   - ggphi MSSM signals
#   - bbphi MSSM signals
#
# Everything else is treated as background.
# -------------------------------------------------------------------------

background_datasets="*"

background_skip_datasets="data_*,ggphi_phitt_*,bbphi_phitt_*"


# -------------------------------------------------------------------------
# Inclusive variables
# -------------------------------------------------------------------------

variables_emu_list=(
    "emu_mt_tot"
)

variables_emu="$(
    IFS=,
    echo "${variables_emu_list[*]}"
)"


# -------------------------------------------------------------------------
# BDT seed variable
#
# Background datasets evaluate all configured BDT masses.
#
# Therefore requesting one BDT datacard variable triggers the histogram
# expander and produces:
#
#   42 masses x 4 BDT datacard variables
# -------------------------------------------------------------------------

first_bdt_mass="$(
python - <<'PY'
from MSSM_H_tt.config.mass_points import read_bdt_masses
print(read_bdt_masses()[0])
PY
)"

if [[ -z "$first_bdt_mass" ]]; then
    echo "ERROR: could not determine the first configured BDT mass"
    exit 1
fi

bdt_variable="bdt_D_DY_M${first_bdt_mass}"


# ============================================================================
# JEC sources
# ============================================================================

jec_sources=(
    jec_Regrouped_Absolute
    jec_Regrouped_BBEC1
    jec_Regrouped_EC2
    jec_Regrouped_HF
    jec_Regrouped_RelativeBal
    jec_Regrouped_FlavorQCD
)

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
# All systematic sources
#
# Keep this identical to the final plotting script.
#
# Weight-only shifts are embedded into the nominal histogram.
# MergeShiftedHistograms only creates separate MergeHistograms requirements
# for the non-embedded, typically kinematic, shifts.
# ============================================================================

shift_sources=(
    muon_weight
    electron_weight
    Trigger_SF_weight
    pu_weight
    top_pt_weight
    zpt_weight

    unclustered
    recoilresp
    recoilres

    CMS_PS_ISR
    CMS_PS_FSR
    CMS_Scale_muR
    CMS_Scale_muF

    btag_weight_hf
    btag_weight_lf
    btag_weight_hfstats1
    btag_weight_hfstats2
    btag_weight_lfstats1
    btag_weight_lfstats2
    btag_weight_cferr1
    btag_weight_cferr2

    "${jec_sources[@]}"

    jer
)

shift_sources_csv="$(
    IFS=,
    echo "${shift_sources[*]}"
)"


# -------------------------------------------------------------------------
# Sanity checks
# -------------------------------------------------------------------------

if [[ -z "$config" ]]; then
    echo "ERROR: config is empty"
    exit 1
fi

if [[ -z "$version" ]]; then
    echo "ERROR: version is empty"
    exit 1
fi

if [[ -z "$background_datasets" ]]; then
    echo "ERROR: background dataset selection is empty"
    exit 1
fi

if [[ -z "$shift_sources_csv" ]]; then
    echo "ERROR: shift_sources_csv is empty"
    exit 1
fi


# -------------------------------------------------------------------------
# Separate LAW controller metadata from the signal/data submissions.
#
# Physics outputs still use the common configured output store.
# -------------------------------------------------------------------------

export CF_JOB_BASE="${CF_DATA}/jobs/${configuration_option}/${version}/backgrounds"

mkdir -p "$CF_JOB_BASE"


# -------------------------------------------------------------------------
# Common arguments
# -------------------------------------------------------------------------

common_args=(
    --configs "$config"

    --datasets "$background_datasets"
    --skip-datasets "$background_skip_datasets"

    --version "$version"

    --shift-sources "$shift_sources_csv"

    --workflow "htcondor"
    --workers "25"
    --poll-interval "5m"
    --pilot "True"

    "${extra_args[@]}"
)


# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------

echo
echo "======================================================================"
echo "MSSM background histogram production"
echo "======================================================================"
echo "Configuration option: $configuration_option"
echo "Configs:              $config"
echo "Channel:              $channel"
echo "Version:              $version"
echo "Datasets:             all non-data, non-MSSM-signal datasets"
echo "Skipped datasets:     $background_skip_datasets"
echo "BDT seed variable:    $bdt_variable"
echo "Inclusive variables:  $variables_emu"
echo "CF_JOB_BASE:          $CF_JOB_BASE"
echo
echo "The workflow will stop at MergeShiftedHistograms."
echo "No plots will be produced."
echo "======================================================================"
echo


# ============================================================================
# 1. Produce the complete BDT histogram block
# ============================================================================

echo
echo "======================================================================"
echo "Producing background BDT histograms up to MergeShiftedHistograms"
echo "======================================================================"
echo

echo \
    law run cf.MergeShiftedHistogramsWrapper \
    "${common_args[@]}" \
    --variables "$bdt_variable"

echo

law run cf.MergeShiftedHistogramsWrapper \
    "${common_args[@]}" \
    --variables "$bdt_variable"

status=$?

if [[ $status -ne 0 ]]; then
    echo
    echo "ERROR: background BDT histogram production failed"
    echo
    exit $status
fi


# ============================================================================
# 2. Produce inclusive SR histograms
# ============================================================================

echo
echo "======================================================================"
echo "Producing background inclusive histograms up to MergeShiftedHistograms"
echo "======================================================================"
echo

echo \
    law run cf.MergeShiftedHistogramsWrapper \
    "${common_args[@]}" \
    --variables "$variables_emu"

echo

law run cf.MergeShiftedHistogramsWrapper \
    "${common_args[@]}" \
    --variables "$variables_emu"

status=$?

if [[ $status -ne 0 ]]; then
    echo
    echo "ERROR: background inclusive histogram production failed"
    echo
    exit $status
fi


# -------------------------------------------------------------------------
# Done
# -------------------------------------------------------------------------

echo
echo "======================================================================"
echo "Background production completed successfully"
echo "======================================================================"
echo
echo "All requested background datasets have reached:"
echo
echo "  MergeShiftedHistograms"
echo
echo "for:"
echo
echo "  - the complete BDT datacard histogram block"
echo "  - the inclusive SR distribution(s)"
echo
echo "No plotting task was run."
echo "======================================================================"
echo