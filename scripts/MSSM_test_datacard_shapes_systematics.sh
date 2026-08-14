#!/bin/bash
set -euo pipefail


# ============================================================================
# Limited systematic test of the four MSSM datacard distributions.
#
# Uses one file per dataset through the "_limited" config.
# ============================================================================

config="run3_2022_preEE_emu_limited"
mass=100

upstream_version="${UPSTREAM_VERSION:-dust_dev}"
test_version="${TEST_VERSION:-datacard_shapes_test}"

workflow="${WORKFLOW:-htcondor}"
poll_interval="${POLL_INTERVAL:-5m}"
workers="${WORKERS:-15}"
pilot="${PILOT:-True}"

# M100 is in this block.
producers="main_common,bdt_card_M90_95_100_105_110_115"


# ============================================================================
# LIMITED SAMPLE
#
# IMPORTANT:
# use actual config process names, not process-group names such as "wj".
# ============================================================================

datasets=(
    "TTto2L2Nu"
    "DYto2Tau_MLL_50_0J_amcatnloFXFX"
    "DYto2L_M_50_0J_amcatnloFXFX"
    "ggphi_phitt_100"
    "bbphi_phitt_100"
)

processes=(
    "tt_dl"
    "dy_tt_m50_0j"
    "dy_ll_m50_0j"
    "ggphi_phitt_100"
    "bbphi_phitt_100"
)

datasets_csv=$(IFS=,; echo "${datasets[*]}")
processes_csv=$(IFS=,; echo "${processes[*]}")


# ============================================================================
# SYSTEMATICS
#
# Start with a representative set.
#
# This tests:
#   - event-weight propagation
#   - lepton SFs
#   - b tagging
#   - MET-dependent BDT response
#   - jet-dependent BDT response
#   - recoil-dependent BDT response
# ============================================================================

shift_sources="${SHIFT_SOURCES:-\
jec_TimePtEta_up,\
jec_TimePtEta_down,\
jer,\
}"


# ============================================================================
# Versions
# ============================================================================

upstream_tasks=(
    cf.CalibrateEvents
    cf.SelectEvents
    cf.ReduceEvents
    cf.MergeReducedEvents
    cf.MergeSelectionStats
    cf.ProvideReducedEvents
)

test_tasks=(
    cf.ProduceColumns
    cf.CreateHistograms
    cf.MergeHistograms
    cf.MergeShiftedHistograms
)


run_plot() {

    local category="$1"
    local variables="$2"

    echo
    echo "======================================================================"
    echo "[test] Category : $category"
    echo "[test] Variables: $variables"
    echo "[test] Datasets : $datasets_csv"
    echo "[test] Processes: $processes_csv"
    echo "[test] Shifts   : $shift_sources"
    echo "======================================================================"
    echo

    args=(
        --configs "$config"

        --version "$test_version"

        --datasets "$datasets_csv"
        --processes "$processes_csv"

        --categories "$category"
        --variables "$variables"

        --shift-sources "$shift_sources"

        --producers "$producers"

        --workflow "$workflow"
        --poll-interval "$poll_interval"
        --pilot "$pilot"
    )


    # Reuse reduced events.
    for task in "${upstream_tasks[@]}"; do
        args+=(
            "--${task}-version" "$upstream_version"
            "--${task}-workflow" "$workflow"
            "--${task}-poll-interval" "$poll_interval"
            "--${task}-pilot" "$pilot"
        )
    done


    # Recreate BDT columns + histograms.
    for task in "${test_tasks[@]}"; do
        args+=(
            "--${task}-version" "$test_version"
            "--${task}-workflow" "$workflow"
            "--${task}-poll-interval" "$poll_interval"
            "--${task}-pilot" "$pilot"
        )
    done


    echo "law run cf.PlotShiftedVariablesPerShift1D \\"
    printf '  %q ' "${args[@]}"
    echo "--workers $workers"
    echo

    law run cf.PlotShiftedVariablesPerShift1D \
        "${args[@]}" \
        --workers "$workers"
}


# ============================================================================
# 1. ggphi signal datacard variable
# ============================================================================

run_plot \
    "cat_emu_sr__bdt_ggphi_and_bbphi_M${mass}" \
    "bdt_D_sig_vs_Disc_ggphi_M${mass}"


# ============================================================================
# 2. bbphi signal datacard variable
# ============================================================================

run_plot \
    "cat_emu_sr__bdt_ggphi_and_bbphi_M${mass}" \
    "bdt_D_sig_vs_Disc_bbphi_M${mass}"


# ============================================================================
# 3. DY-region datacard variable
# ============================================================================

run_plot \
    "cat_emu_sr__bdt_dy_M${mass}" \
    "bdt_D_DY_M${mass}"


# ============================================================================
# 4. tt-region datacard variable
# ============================================================================

run_plot \
    "cat_emu_sr__bdt_tt_M${mass}" \
    "bdt_D_TT_M${mass}"


echo
echo "======================================================================"
echo "[done] Limited systematic datacard-distribution test completed."
echo "======================================================================"