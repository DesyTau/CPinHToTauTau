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
# Signal datasets
#
# MergeShiftedHistogramsWrapper accepts dataset patterns, therefore both
# signal production mechanisms and all configured masses can be scheduled
# simultaneously.
# -------------------------------------------------------------------------

signal_datasets="ggphi_phitt_*,bbphi_phitt_*"


# -------------------------------------------------------------------------
# Inclusive variables
#
# Keep this as a separate MergeShiftedHistograms requirement so that the
# output task is exactly the same one that will later be requested by
# PlotDatacardDistributions.
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
# Requesting one BDT datacard variable triggers the histogram-variable
# expander.
#
# With the dataset-aware expander:
#
#   ggphi_phitt_500
#       -> all four BDT card variables at M500
#
#   bbphi_phitt_1200
#       -> all four BDT card variables at M1200
#
# The actual first configured mass is read from bdt_masses.yaml so that
# it is not duplicated in this script.
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

# Common across eras
jec_sources=(
    jec_Regrouped_Absolute
    jec_Regrouped_BBEC1
    jec_Regrouped_EC2
    jec_Regrouped_HF
    jec_Regrouped_RelativeBal
    jec_Regrouped_FlavorQCD
)

# Add the era-dependent sources needed by all requested configs.
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
# Weight-only shifts are embedded into the nominal histogram by your custom
# histogram producer. MergeShiftedHistograms knows which sources are embedded
# and does not request independent MergeHistograms tasks for them.
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

if [[ -z "$signal_datasets" ]]; then
    echo "ERROR: signal dataset selection is empty"
    exit 1
fi

if [[ -z "$shift_sources_csv" ]]; then
    echo "ERROR: shift_sources_csv is empty"
    exit 1
fi


# -------------------------------------------------------------------------
# Separate LAW controller files from other simultaneous submissions.
#
# This changes only LAW/HTCondor bookkeeping. The actual physics outputs
# still use the normal common output store and version.
# -------------------------------------------------------------------------

export CF_JOB_BASE="${CF_DATA}/jobs/${configuration_option}/${version}/signals"

mkdir -p "$CF_JOB_BASE"


# -------------------------------------------------------------------------
# Common arguments for MergeShiftedHistogramsWrapper
#
# The wrapper creates one independent MergeShiftedHistograms workflow for
# each selected signal dataset.
#
# For example:
#
#   MergeShiftedHistograms(ggphi_phitt_60)
#   MergeShiftedHistograms(bbphi_phitt_60)
#   MergeShiftedHistograms(ggphi_phitt_65)
#   MergeShiftedHistograms(bbphi_phitt_65)
#   ...
#
# Luigi resolves all dependencies independently, so one signal can already
# be running ProduceColumns while another is still in ReduceEvents, etc.
# -------------------------------------------------------------------------

common_args=(
    --configs "$config"

    --datasets "$signal_datasets"

    --version "$version"

    --shift-sources "$shift_sources_csv"

    --workflow "htcondor"
    --workers "10"

    --poll-interval "5m"
    --pilot "True"

    "${extra_args[@]}"
)


# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------

echo
echo "======================================================================"
echo "MSSM signal histogram production"
echo "======================================================================"
echo "Configuration option: $configuration_option"
echo "Configs:              $config"
echo "Channel:              $channel"
echo "Version:              $version"
echo "Signal datasets:      $signal_datasets"
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
echo "Producing signal BDT histograms up to MergeShiftedHistograms"
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
    echo "ERROR: signal BDT histogram production failed"
    echo
    exit $status
fi


# ============================================================================
# 2. Produce the inclusive SR histogram
#
# This is deliberately a second task rather than adding emu_mt_tot to the BDT
# variable list. This makes the resulting task identical to the inclusive
# requirement of the final PlotDatacardDistributions workflow.
# ============================================================================

echo
echo "======================================================================"
echo "Producing signal inclusive histograms up to MergeShiftedHistograms"
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
    echo "ERROR: signal inclusive histogram production failed"
    echo
    exit $status
fi


# -------------------------------------------------------------------------
# Done
# -------------------------------------------------------------------------

echo
echo "======================================================================"
echo "Signal production completed successfully"
echo "======================================================================"
echo
echo "All requested signal datasets have reached:"
echo
echo "  MergeShiftedHistograms"
echo
echo "for:"
echo
echo "  - the BDT datacard distributions"
echo "  - the inclusive SR distribution(s)"
echo
echo "No plotting task was run."
echo "======================================================================"
echo