# coding: utf-8

"""
Configuration of the MSSM analysis using nano v15
"""
import law
import order as od

# ------------------------ #
# The main analysis object #
# ------------------------ #
analysis_MSSM_H_tt_2024_nano_v15 = ana = od.Analysis(
    name="analysis_MSSM_H_tt_2024_nano_v15",
    id=1,
)

# analysis-global versions
# (see cfg.x.versions below for more info)
ana.x.versions = {}

ana.x.bash_sandboxes = [
    "$CF_BASE/sandboxes/cf.sh",
    "$CF_BASE/sandboxes/venv_columnar.sh",
    "$HTTCP_BASE/sandboxes/venv_columnar_xgb.sh",
]

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
ana.x.config_groups = {}

# ------------- #
# setup configs #
# ------------- #

from MSSM_H_tt.config.config_run3_2024_nanov15_v1 import add_run3

# ------------------------------------------------------------- #

# channels = ['etau','mutau','emu','tautau']

channels = ['emu']

#------------------------ Run3 2024 samples ----------------------- #

from cmsdb.campaigns.run3_2024_nano_v15 import campaign_run3_2024_nano_v15
for counter, value in enumerate(channels):
    add_run3(
        analysis_MSSM_H_tt_2024_nano_v15,
        campaign_run3_2024_nano_v15.copy(),
        channel=value,
        config_name=f"run3_2024_{value}",
        config_id=40+counter)
for counter, value in enumerate(channels):
    add_run3(
        analysis_MSSM_H_tt_2024_nano_v15,
        campaign_run3_2024_nano_v15.copy(),
        channel=value,
        config_name=f"run3_2024_{value}_limited",
        config_id=44+counter,
        limit_dataset_files=1)