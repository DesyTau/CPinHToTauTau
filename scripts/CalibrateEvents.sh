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


# -------------------------------------------------------------------------
# Schedule calibration for all configs and datasets
#
# One CalibrateEvents workflow is created for every config/dataset pair.
# The individual file branches are submitted to HTCondor.
# -------------------------------------------------------------------------

args=(
    --configs "$config"
    --datasets "$datasets"
    --version "$version"

    --shifts "nominal"
    --calibrator "main"

    --workflow "htcondor"
    --htcondor-memory "18GB"
    --workers "8"
    --poll-interval "5m"
    --pilot "True"
    --retries "10"

    "${extra_args[@]}"
)


echo
echo "======================================================================"
echo "CalibrateEvents production"
echo "======================================================================"
echo "Configuration option: $1"
echo "Configs:              $config"
echo "Datasets:             $datasets"
echo "Version:              $version"
echo "Calibrator:           main"
echo "Shift:                nominal"
echo "Workflow:             htcondor"
echo "Luigi workers:        8"
echo "Retries:              10"
echo "======================================================================"
echo

echo law run cf.CalibrateEventsWrapper "${args[@]}"
echo

law run cf.CalibrateEventsWrapper "${args[@]}"

status=$?

if [[ $status -ne 0 ]]; then
    echo
    echo "ERROR: CalibrateEvents production failed"
    echo
    exit $status
fi

echo
echo "======================================================================"
echo "CalibrateEvents production completed"
echo "======================================================================"