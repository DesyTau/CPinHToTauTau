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
# Data datasets
# -------------------------------------------------------------------------

data_datasets="data_*"
data_skip_datasets="data_tau_*"

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
# Data are not mass-specific signal samples, therefore the histogram
# variable expander expands this request to the complete BDT mass block.
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

if [[ -z "$data_datasets" ]]; then
    echo "ERROR: data dataset selection is empty"
    exit 1
fi


# -------------------------------------------------------------------------
# Separate LAW controller metadata
# -------------------------------------------------------------------------

export CF_JOB_BASE="${CF_DATA}/jobs/${configuration_option}/${version}/data"

mkdir -p "$CF_JOB_BASE"


# -------------------------------------------------------------------------
# Common arguments
#
# IMPORTANT:
#
# Data use MergeHistograms with the nominal shift only.
#
# This is also exactly what PlotShiftedVariables1D requests later for data.
# -------------------------------------------------------------------------

common_args=(
    --configs "$config"

    --datasets "$data_datasets"
    --skip-datasets "$data_skip_datasets"

    --shifts "nominal"

    --version "$version"

    --workflow "htcondor"
    --workers "2"
    --poll-interval "5m"
    --pilot "True"

    "${extra_args[@]}"
)

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------

echo
echo "======================================================================"
echo "MSSM data histogram production"
echo "======================================================================"
echo "Configuration option: $configuration_option"
echo "Configs:              $config"
echo "Channel:              $channel"
echo "Version:              $version"
echo "Data datasets:        $data_datasets"
echo "Skipped datasets:     $data_skip_datasets"
echo "BDT seed variable:    $bdt_variable"
echo "Inclusive variables:  $variables_emu"
echo "Shift:                nominal"
echo "CF_JOB_BASE:          $CF_JOB_BASE"
echo
echo "Data will stop at MergeHistograms."
echo "No MergeShiftedHistograms task is needed for data."
echo "No plots will be produced."
echo "======================================================================"
echo


# ============================================================================
# 1. Produce BDT histograms
# ============================================================================

echo
echo "======================================================================"
echo "Producing data BDT histograms up to MergeHistograms"
echo "======================================================================"
echo

echo \
    law run cf.MergeHistogramsWrapper \
    "${common_args[@]}" \
    --variables "$bdt_variable"

echo


law run cf.MergeHistogramsWrapper \
    "${common_args[@]}" \
    --variables "$bdt_variable"

status=$?

if [[ $status -ne 0 ]]; then
    echo
    echo "ERROR: data BDT histogram production failed"
    echo
    exit $status
fi


# ============================================================================
# 2. Produce inclusive SR histogram
# ============================================================================

echo
echo "======================================================================"
echo "Producing data inclusive histograms up to MergeHistograms"
echo "======================================================================"
echo

echo \
    law run cf.MergeHistogramsWrapper \
    "${common_args[@]}" \
    --variables "$variables_emu"

echo


law run cf.MergeHistogramsWrapper \
    "${common_args[@]}" \
    --variables "$variables_emu"

status=$?

if [[ $status -ne 0 ]]; then
    echo
    echo "ERROR: data inclusive histogram production failed"
    echo
    exit $status
fi


# -------------------------------------------------------------------------
# Done
# -------------------------------------------------------------------------

echo
echo "======================================================================"
echo "Data production completed successfully"
echo "======================================================================"
echo
echo "All requested data datasets have reached:"
echo
echo "  MergeHistograms(nominal)"
echo
echo "for:"
echo
echo "  - the complete BDT histogram block"
echo "  - the inclusive SR distribution(s)"
echo
echo
echo "This is the final histogram stage required for data."
echo "No MergeShiftedHistograms task is required."
echo "No plotting task was run."
echo "======================================================================"
echo