#!/bin/bash
set -euo pipefail

usage() {
cat <<'EOF'
Usage:
  ./MSSM_create_datacards.sh CONFIG [options] [extra law options]

Options:
  --masses "M1 M2 M3"       Run only these masses
  --masses M1,M2,M3         Same, comma-separated
  --mass M                  Add one mass point; can be repeated
  --all-masses              Run the full default mass list
  -h, --help                Show this help

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

  ./MSSM_create_datacards.sh 23_emu --masses "100 200" --workers 10
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

version=desy_dev

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
echo "[info] Version: $version"
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

            --cf.CalibrateEvents-version "$version"
            --cf.CalibrateEvents-workflow "$workflow"

            --cf.SelectEvents-version "$version"
            --cf.SelectEvents-workflow "$workflow"

            --cf.ReduceEvents-version "$version"
            --cf.ReduceEvents-workflow "$workflow"

            --cf.MergeReducedEvents-version "$version"
            --cf.MergeReducedEvents-workflow "$workflow"

            --cf.MergeSelectionStats-version "$version"
            --cf.MergeSelectionStats-workflow "$workflow"

            --cf.ProvideReducedEvents-version "$version"
            --cf.ProvideReducedEvents-workflow "$workflow"

            --cf.ProduceColumns-version "$version"
            --cf.ProduceColumns-workflow "$workflow"

            --cf.MergeHistograms-version "$version"
            --cf.MergeHistograms-workflow "$workflow"

            --cf.CreateHistograms-version "$version"
            --cf.CreateHistograms-workflow "$workflow"

            --cf.MergeShiftedHistograms-version "$version"
            --cf.MergeShiftedHistograms-workflow "$workflow"

            --pilot True
            --version "$version"

            --inference-model "$inference_model"
            --hist-hooks qcd

            "${extra_args[@]}"
        )

        echo
        echo "[info] Running inference model: $inference_model"
        echo law run cf.CreateDatacards "${args[@]}"
        law run cf.CreateDatacards "${args[@]}"
    done
done