#!/bin/bash
source ./common_run3_MSSM.sh
set_common_vars "$1"

pt_bins=${pt_bins:-"20,30,50,70,100,140,200,300,600,1000"}
eta_bins=${eta_bins:-"0.0,1.5,2.5"}
jet_pt_min=${jet_pt_min:-20.0}
jet_abseta_max=${jet_abseta_max:-2.5}
event_weight_column=${event_weight_column:-"normalization_weight"}
producers=${producers:-"main"}

# split configs (comma-separated)
IFS=',' read -r -a cfgs <<< "$config"

# datasets can be single list or ':'-separated per config
IFS=':' read -r -a dsets_per_cfg <<< "$datasets"

clean_csv() {
  local s="$1"
  s="${s#,}"
  s="${s%,}"
  echo "$s"
}

processes_clean="$(clean_csv "${processes:-}")"

for i in "${!cfgs[@]}"; do
  cfg="${cfgs[$i]}"

  if [ "${#dsets_per_cfg[@]}" -eq 1 ]; then
    ds="${dsets_per_cfg[0]}"
  else
    ds="${dsets_per_cfg[$i]}"
  fi
  ds="$(clean_csv "$ds")"

  args=(
    --config "$cfg"
    --datasets "$ds"
    --version "$version"
    --producers "$producers"

    --cf.BundleBashSandbox-workflow local
    --cf.BundleRepo-workflow local
    --cf.BundleSoftware-workflow local 
    --cf.BundleExternalFiles-workflow local 
    --cf.BundleCMSSWSandbox-workflow local
    --cf.CalibrateEvents-workflow $workflow
    --cf.SelectEvents-workflow $workflow
    --cf.ReduceEvents-workflow $workflow
    --cf.ProduceColumns-workflow $workflow
    --cf.UniteColumns-workflow $workflow
    --cf.CreateBTagEfficiencyMaps-workflow local

    --pt-bins "$pt_bins"
    --eta-bins "$eta_bins"
    --jet-pt-min "$jet_pt_min"
    --jet-abseta-max "$jet_abseta_max"

    "${@:2}"
  )

  if [ -n "$processes_clean" ]; then
    args+=( --processes "$processes_clean" )
  fi

  if [ -n "$event_weight_column" ]; then
    args+=( --event-weight-column "$event_weight_column" )
  fi

  echo law run cf.CreateBTagEfficiencyMaps "${args[@]}"
  law run cf.CreateBTagEfficiencyMaps "${args[@]}" || exit $?
done