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


# -------------------------------------------------------------------------
# Get all dataset LFNs in parallel
#
# GetDatasetLFNsWrapper creates one cf.GetDatasetLFNs task for each
# requested config/dataset combination.
#
# Luigi workers execute independent GetDatasetLFNs tasks concurrently.
# -------------------------------------------------------------------------

args=(
    --configs "$config"
    --datasets "$datasets"
    --shifts "nominal"

    --workers "8"
    --local-scheduler "True"

    "${extra_args[@]}"
)


echo
echo "======================================================================"
echo "GetDatasetLFNs production"
echo "======================================================================"
echo "Configuration option: $1"
echo "Configs:              $config"
echo "Datasets:             $datasets"
echo "Parallel workers:     8"
echo "======================================================================"
echo

echo law run cf.GetDatasetLFNsWrapper "${args[@]}"
echo

law run cf.GetDatasetLFNsWrapper "${args[@]}"

status=$?

if [[ $status -ne 0 ]]; then
    echo
    echo "ERROR: GetDatasetLFNs production failed"
    echo
    exit $status
fi

echo
echo "======================================================================"
echo "GetDatasetLFNs production completed"
echo "======================================================================"