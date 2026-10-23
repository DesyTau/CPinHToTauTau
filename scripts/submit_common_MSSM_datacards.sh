#!/bin/bash
set -euo pipefail

usage_common() {
cat <<'EOF'
Usage:
  SCRIPT CONFIG [options] [extra law options]

Options:
  --masses "M1 M2 M3"       Run only these masses
  --masses M1,M2,M3         Same, comma-separated
  --mass M                  Add one mass point; can be repeated
  --all-masses              Run the full default mass list
  --workers N               Forwarded to law. Default: 1
  -h, --help                Show this help

Environment:
  VERSION                   Default: desy_dev
  SHIFT_SOURCES             Override comma-separated shift sources
  HIST_PRODUCER             Default: httcp_hist_producer
EOF
}

parse_common_args() {
    if [[ $# -lt 1 ]]; then
        usage_common
        exit 1
    fi

    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage_common
        exit 0
    fi

    config_arg="$1"
    shift

    source ./common_run3_MSSM.sh
    set_common_vars "$config_arg"

    version="${VERSION:-desy_dev}"
    workers=1

    hist_producer="${HIST_PRODUCER:-httcp_hist_producer}"

    shift_sources="${SHIFT_SOURCES:-Trigger_SF_weight,btag_weight_cferr1,btag_weight_cferr2,btag_weight_hf,btag_weight_hfstats1,btag_weight_hfstats2,btag_weight_lf,btag_weight_lfstats1,btag_weight_lfstats2,electron_weight,jec_Total,jer,muon_weight,nominal,pu_weight,top_pt_weight,unclustered}"

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

            --workers)
                if [[ $# -lt 2 ]]; then
                    echo "[error] Missing argument after $1" >&2
                    exit 1
                fi
                workers="$2"
                extra_args+=("--workers" "$2")
                shift 2
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
    echo "[info] Workers: $workers"
    echo "[info] Masses: ${masses[*]}"
}

model_names_for_mass() {
    local m="$1"

    printf '%s\n' \
        "MSSM_model_D_sig_vs_Disc_ggphi_M${m}" \
        "MSSM_model_D_sig_vs_Disc_bbphi_M${m}" \
        "MSSM_model_D_DY_M${m}" \
        "MSSM_model_D_TT_M${m}"
}

variable_for_model() {
    local model="$1"

    if [[ "$model" =~ MSSM_model_D_sig_vs_Disc_ggphi_M([0-9]+)$ ]]; then
        echo "bdt_D_sig_vs_Disc_ggphi_M${BASH_REMATCH[1]}"
    elif [[ "$model" =~ MSSM_model_D_sig_vs_Disc_bbphi_M([0-9]+)$ ]]; then
        echo "bdt_D_sig_vs_Disc_bbphi_M${BASH_REMATCH[1]}"
    elif [[ "$model" =~ MSSM_model_D_DY_M([0-9]+)$ ]]; then
        echo "bdt_D_DY_M${BASH_REMATCH[1]}"
    elif [[ "$model" =~ MSSM_model_D_TT_M([0-9]+)$ ]]; then
        echo "bdt_D_TT_M${BASH_REMATCH[1]}"
    else
        echo "[error] Cannot infer variable for model: $model" >&2
        exit 1
    fi
}

category_for_model() {
    local model="$1"

    if [[ "$model" =~ MSSM_model_D_sig_vs_Disc_ggphi_M([0-9]+)$ ]]; then
        echo "bdt_cat_ggphi_and_bbphi_M${BASH_REMATCH[1]}"
    elif [[ "$model" =~ MSSM_model_D_sig_vs_Disc_bbphi_M([0-9]+)$ ]]; then
        echo "bdt_cat_ggphi_and_bbphi_M${BASH_REMATCH[1]}"
    elif [[ "$model" =~ MSSM_model_D_DY_M([0-9]+)$ ]]; then
        echo "bdt_cat_dy_M${BASH_REMATCH[1]}"
    elif [[ "$model" =~ MSSM_model_D_TT_M([0-9]+)$ ]]; then
        echo "bdt_cat_tt_M${BASH_REMATCH[1]}"
    else
        echo "[error] Cannot infer category for model: $model" >&2
        exit 1
    fi
}

run_remote_task_no_poll() {
    local task="$1"
    shift

    echo
    echo "[submit] law run $task ..."
    law run "$task" \
        --config "$config" \
        --version "$version" \
        --workflow "$workflow" \
        --no-poll \
        "$@" \
        "${extra_args[@]}"
}

run_local_task() {
    local task="$1"
    shift

    echo
    echo "[run] law run $task ..."
    law run "$task" \
        --config "$config" \
        --version "$version" \
        "$@" \
        "${extra_args[@]}"
}