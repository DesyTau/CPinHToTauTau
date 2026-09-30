# coding: utf-8

"""
Inference model for the MSSM analysis.
"""

import law

from columnflow.inference import inference_model, ParameterType
from columnflow.config_util import get_datasets_from_process
from MSSM_H_tt.inference.base import HCPModelBase
from MSSM_H_tt.config.mass_points import read_bdt_masses


class MSSM_model(HCPModelBase):
    """
    Default statistical model for MSSM analysis.
    """

    name = "MSSM_model"
    add_qcd = True

    # Keep qcd in the datacard, but do not attach shape nuisances to it.
    # Set to True only if you really want qcd to receive shape systematics.
    use_qcd_shape_uncertainties = False

    # Keep the combine/datacard process name explicit and consistent.
    qcd_combine_name = "qcd"

    # Specialization knobs for derived inference models
    signal_mass = None      # e.g. 105
    signal_kind = None      # None, "ggphi", or "bbphi"

    processes: list = []
    config_categories: list = []
    systematics: list = []

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------

    def get_mass_points(self):
        masses = list(read_bdt_masses())

        if self.signal_mass is None:
            return masses

        ref = masses[0]
        target = str(self.signal_mass) if isinstance(ref, str) else int(self.signal_mass)

        if target not in masses:
            raise ValueError(
                f"Requested signal mass {target} not found in available mass points: {masses}"
            )

        return [target]

    def _get_config_insts(self):
        config_insts = getattr(self, "config_insts", None)
        if config_insts:
            return list(config_insts)

        config_insts = []
        for cfg in getattr(self, "config", []):
            if isinstance(cfg, (list, tuple, set)):
                config_insts.extend(cfg)
            else:
                config_insts.append(cfg)
        return config_insts

    @staticmethod
    def _dedup_keep_order(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _resolve_representative_process(self, config_inst, preferred_process, dataset_processes):
        """
        Return a valid config-process name to be used in process_config_spec(process=...).

        Priority:
          1. preferred representative process, if it exists;
          2. if there is exactly one dataset process, use that if it exists;
          3. otherwise return None.
        """
        if preferred_process is not None:
            try:
                config_inst.get_process(preferred_process)
                return preferred_process
            except Exception:
                pass

        if len(dataset_processes) == 1:
            only_proc = dataset_processes[0]
            try:
                config_inst.get_process(only_proc)
                return only_proc
            except Exception:
                pass

        return None

    def _get_data_prefixes(self, ch):
        data_prefixes = {
            "etau": ["data_egamma_", "data_e_"],
            "mutau": ["data_mu_", "data_singlemu_"],
            "emu": ["data_egamma_", "data_mu_"],  # add "data_muoneg_" if needed
            "tautau": ["data_tau_"],
        }
        return data_prefixes.get(ch, [f"data_{ch}_"])

    def _get_data_datasets(self, config_inst, ch):
        prefixes = self._get_data_prefixes(ch)

        data_datasets = [
            ds_name
            for ds_name in config_inst.datasets.names()
            if any(ds_name.startswith(prefix) for prefix in prefixes)
        ]

        if not data_datasets:
            raise ValueError(
                f"No data datasets found for channel '{ch}' in config '{config_inst.name}'. "
                f"Available datasets: {list(config_inst.datasets.names())}"
            )

        return data_datasets

    # -------------------------------------------------------------------------
    # process map
    # -------------------------------------------------------------------------

    def init_proc_map(self) -> None:
        """
        Mapping between combine process names and:
          - one representative config process name (`process`)
          - the config processes used to collect datasets (`dataset_processes`)
        """

        self.proc_map = {
            "vv": {
                "process": "vv",
                "dataset_processes": ["ww", "wz", "zz"],
                "is_signal": False,
                "is_data_driven": False,
            },
            "vvv": {
                "process": "vvv",
                "dataset_processes": ["www", "wwz", "zzz"],
                "is_signal": False,
                "is_data_driven": False,
            },
            "tt": {
                "process": "tt",
                "dataset_processes": ["tt_dl", "tt_fh", "tt_sl"],
                "is_signal": False,
                "is_data_driven": False,
            },
            "st": {
                "process": "st",
                "dataset_processes": [
                    "st_tchannel_tbar",
                    "st_tchannel_t",
                    "st_schannel_t_lep",
                    "st_schannel_tbar_lep",
                    "st_twchannel_tbar_fh",
                    "st_twchannel_t_fh",
                    "st_twchannel_tbar_dl",
                    "st_twchannel_tbar_sl",
                    "st_twchannel_t_dl",
                    "st_twchannel_t_sl",
                ],
                "is_signal": False,
                "is_data_driven": False,
            },
            "h_ggf_htt_sm_prod_sm": {
                "process": "h_ggf_htt_sm_prod_sm",
                "dataset_processes": ["h_ggf_htt_sm_prod_sm"],
                "is_signal": False,
                "is_data_driven": False,
            },
            "h_vbf_htt_sm": {
                "process": "h_vbf_htt_sm",
                "dataset_processes": ["h_vbf_htt_sm"],
                "is_signal": False,
                "is_data_driven": False,
            },
            "vh_htt": {
                "process": "vh_htt",
                "dataset_processes": [
                    "zh_htt_flat",
                    "wph_htt_flat",
                    "wmh_htt_flat",
                ],
                "is_signal": False,
                "is_data_driven": False,
            },
            "wj": {
                "process": "w",
                "dataset_processes": [
                    "wj",
                    "wj_1j",
                    "wj_2j",
                    "wj_3j",
                    "wj_4j",
                ],
                "is_signal": False,
                "is_data_driven": False,
            },
            "dy_tt_m50": {
                "process": "dy_tt_m50",
                "dataset_processes": [
                    "dy_tt_m50_0j",
                    "dy_tt_m50_1j",
                    "dy_tt_m50_2j",
                ],
                "is_signal": False,
                "is_data_driven": False,
            },
            "dy_lep": {
                "process": "dy_lep",
                "dataset_processes": [
                    "dy_lep_m10to50",
                    "dy_ll_m50_0j",
                    "dy_ll_m50_1j",
                    "dy_ll_m50_2j",
                    "dy_ll_m50",
                ],
                "is_signal": False,
                "is_data_driven": False,
            },
        }

        if self.add_qcd:
            self.proc_map[self.qcd_combine_name] = {
                "process": "qcd",
                "dataset_processes": [],
                "is_signal": False,
                "is_data_driven": True,
            }

        for m in self.get_mass_points():
            if self.signal_kind in (None, "ggphi"):
                g = f"ggphi_phitt_{m}"
                self.proc_map[g] = {
                    "process": g,
                    "dataset_processes": [g],
                    "is_signal": True,
                    "is_data_driven": False,
                }

            if self.signal_kind in (None, "bbphi"):
                b = f"bbphi_phitt_{m}"
                self.proc_map[b] = {
                    "process": b,
                    "dataset_processes": [b],
                    "is_signal": True,
                    "is_data_driven": False,
                }

    # -------------------------------------------------------------------------
    # categories
    # -------------------------------------------------------------------------

    def init_categories(self) -> None:
        config_insts = self._get_config_insts()

        cfg0 = config_insts[0]
        ch = cfg0.channels.names()[0]

        category_kinds = []

        # keep the existing signal_kind-dependent behavior for signal-optimized regions
        if self.signal_kind in (None, "ggphi"):
            category_kinds.append("ggphi")
        if self.signal_kind in (None, "bbphi"):
            category_kinds.append("bbphi")

        # always add DY and TT categories
        category_kinds.extend(["dy", "tt"])
        category_kinds = self._dedup_keep_order(category_kinds)

        for mass in self.get_mass_points():
            for kind in category_kinds:
                config_data = {}

                for config_inst in config_insts:
                    data_datasets = self._get_data_datasets(config_inst, ch)

                    config_data[config_inst.name] = self.category_config_spec(
                        category=f"cat_{ch}_sr__bdt_{kind}_M{mass}",
                        variable=f"bdt_raw_score_{kind}_M{mass}",
                        data_datasets=data_datasets,
                    )

                self.add_category(
                    name=f"cat_{ch}_sr__bdt_{kind}_M{mass}",
                    config_data=config_data,
                    mc_stats=True,
                    empty_bin_value=0.0,
                )

    # -------------------------------------------------------------------------
    # processes
    # -------------------------------------------------------------------------

    def init_processes(self) -> None:
        """
        Build processes using the new `config_data` + `process_config_spec` API.

        Important:
        - `process` must be a single valid config-process name
        - `mc_datasets` may be the union of many contributing datasets
        - data-driven processes such as qcd get config_data without mc_datasets
        """

        config_insts = self._get_config_insts()

        for combine_name, entry in self.proc_map.items():
            preferred_process = entry["process"]
            dataset_processes = entry["dataset_processes"]
            is_signal = entry.get("is_signal", False)
            is_data_driven = entry.get("is_data_driven", False)

            config_data = {}

            for config_inst in config_insts:
                rep_process = self._resolve_representative_process(
                    config_inst=config_inst,
                    preferred_process=preferred_process,
                    dataset_processes=dataset_processes,
                )

                if rep_process is None:
                    raise ValueError(
                        f"Representative process '{preferred_process}' for combine process "
                        f"'{combine_name}' does not exist in config '{config_inst.name}'."
                    )

                if is_data_driven:
                    config_data[config_inst.name] = self.process_config_spec(
                        process=rep_process,
                    )
                    continue

                dataset_names = []

                for p in dataset_processes:
                    try:
                        config_inst.get_process(p)
                    except Exception:
                        print(
                            f"skipping dataset process {p} in inference model {self.cls_name}, "
                            f"not found in config {config_inst.name}"
                        )
                        continue

                    dsets = [
                        d.name
                        for d in get_datasets_from_process(
                            config_inst,
                            p,
                            strategy="all",
                        )
                    ]
                    dataset_names.extend(dsets)

                dataset_names = self._dedup_keep_order(dataset_names)

                if not dataset_names:
                    continue

                config_data[config_inst.name] = self.process_config_spec(
                    process=rep_process,
                    mc_datasets=dataset_names,
                )

            if not config_data:
                print(
                    f"skipping combine process {combine_name} in inference model {self.cls_name}, "
                    f"no matching datasets or config_data in any config"
                )
                continue

            self.add_process(
                name=combine_name,
                is_signal=is_signal,
                config_data=config_data,
            )

    # -------------------------------------------------------------------------
    # parameters
    # -------------------------------------------------------------------------

    def init_parameters(self) -> None:
        if hasattr(self, "add_parameter_group"):
            for group_name in [
                "experiment",
                "theory",
                "rate_nuisances",
                "shape_nuisances",
                "signal_norm_xs",
                "signal_norm_xsbr",
            ]:
                if not self.has_parameter_group(group_name):
                    self.add_parameter_group(group_name)

        config_insts = self._get_config_insts()

        cfg0 = config_insts[0]
        ch_name = cfg0.channels.names()[0] if getattr(cfg0, "channels", None) else ""
        has_tau = "tau" in ch_name and ch_name != "emu"
        has_mu = "mu" in ch_name or ch_name in ("emu", "mutau")
        has_e = "e" in ch_name or ch_name in ("emu", "etau")

        all_processes = [
            proc_name
            for proc_name in self.proc_map.keys()
            if self.has_process(proc_name)
        ]

        non_qcd_processes = [
            proc_name
            for proc_name in all_processes
            if proc_name != self.qcd_combine_name
        ]

        # ---------------------------------------------------------------------
        # lumi uncertainties
        # ---------------------------------------------------------------------

        lumi_uncs = []
        seen_uncs = set()
        for cfg in config_insts:
            for unc_name in cfg.x.luminosity.uncertainties:
                if unc_name not in seen_uncs:
                    seen_uncs.add(unc_name)
                    lumi_uncs.append(unc_name)

        rate_group = (
            ["experiment", "rate_nuisances"]
            if hasattr(self, "add_parameter_group")
            else "experiment"
        )

        for unc_name in lumi_uncs:
            ref_eff = None
            ref_cfg = None
            for cfg in config_insts:
                lumi = cfg.x.luminosity
                if unc_name not in lumi.uncertainties:
                    continue
                eff = lumi.get(names=unc_name, direction=("down", "up"), factor=True)
                if ref_eff is None:
                    ref_eff, ref_cfg = eff, cfg.name
                else:
                    if eff != ref_eff:
                        raise ValueError(
                            f"lumi nuisance '{unc_name}' has different effects across configs "
                            f"(e.g. {ref_cfg}: {ref_eff}, {cfg.name}: {eff}). "
                            "Either harmonize the lumi config or use per-config parameter names."
                        )

            self.add_parameter(
                unc_name,
                type=ParameterType.rate_gauss,
                effect=ref_eff,
                process=non_qcd_processes,
                group=rate_group,
            )

        # ---------------------------------------------------------------------
        # shape systematics
        # ---------------------------------------------------------------------

        theory_shape_sources = [
            "CMS_PS_ISR",
            "CMS_PS_FSR",
            "CMS_Scale_muR",
            "CMS_Scale_muF",
        ]

        def _has_shift_source(cfg, src: str) -> bool:
            try:
                cfg.get_shift(f"{src}_up")
                cfg.get_shift(f"{src}_down")
                return True
            except Exception:
                return False

        def _nuis_name(src: str) -> str:
            if src == "tau_weight":
                return "CMS_eff_t_SF"
            if src == "muon_weight":
                return "CMS_eff_mu_SF"
            if src == "electron_weight":
                return "CMS_eff_e_SF"
            if src == "Trigger_SF_weight":
                return "CMS_bbtt_eff_trig_SF"
            if src == "top_pt_weight":
                return "CMS_top_pT_reweighting"
            if src == "pu_weight":
                return "CMS_pu_SF"
            if src == "zpt_weight":
                return "CMS_zpt_reweighting"
            if src == "jer":
                return "CMS_res_j"
            if src == "unclustered":
                return "CMS_scale_met_unclustered"
            if src == "recoilresp":
                return "CMS_met_recoil_response"
            if src == "recoilres":
                return "CMS_met_recoil_resolution"
            if src.startswith("jec_"):
                return f"CMS_scale_j_{src[4:]}"
            if src.startswith("btag_weight_"):
                return f"CMS_btag_{src[len('btag_weight_'):]}"
            return src

        def _default_shape_scope() -> list[str]:
            return list(all_processes) if self.use_qcd_shape_uncertainties else list(non_qcd_processes)

        def _recoil_shape_scope() -> list[str]:
            """
            Recoil corrections are configured for DY, W+jets, SM Higgs, VH,
            and the MSSM signal samples in cfg.x.met_recoil["datasets"].
            Do not attach them to tt, single-top, VV, VVV, or QCD.
            """
            default = _default_shape_scope()

            recoil_processes = {
                "dy_tt_m50",
                "dy_lep",
                "wj",
                "h_ggf_htt_sm_prod_sm",
                "h_vbf_htt_sm",
                "vh_htt",
            }

            return [
                p for p in default
                if (
                    p in recoil_processes
                    or p.startswith("ggphi_phitt_")
                    or p.startswith("bbphi_phitt_")
                )
            ]

        def _process_scope(src: str) -> list[str]:
            default = _default_shape_scope()

            if src in theory_shape_sources:
                excluded_theory_processes = {
                    "tt",
                    "dy_tt_m50",
                    "dy_lep",
                }
                return [
                    p for p in default
                    if p not in excluded_theory_processes
                ]

            if src == "top_pt_weight":
                return ["tt"]

            if src == "zpt_weight":
                return ["dy_tt_m50", "dy_lep"]

            if src == "unclustered":
                return default

            if src in ("recoilresp", "recoilres"):
                return _recoil_shape_scope()

            if src.startswith("btag_weight_"):
                return default

            if src.startswith("jec_") or src == "jer":
                return default

            if src == "pu_weight":
                return default

            if src == "tau_weight":
                return default if has_tau else []

            if src == "muon_weight":
                return default if has_mu else []

            if src == "electron_weight":
                return default if has_e else []

            if src == "Trigger_SF_weight":
                return default if (has_mu or has_e or has_tau) else []

            return default

        expected_sources = [
            "tau_weight",
            "muon_weight",
            "electron_weight",
            "Trigger_SF_weight",
            "pu_weight",
            "top_pt_weight",
            "zpt_weight",
            "jer",
            "unclustered",
            "recoilresp",
            "recoilres",
        ]

        expected_sources.extend(theory_shape_sources)

        try:
            expected_sources.extend(
                [f"jec_{src}" for src in cfg0.x.jec.Jet.uncertainty_sources]
            )
        except Exception:
            pass

        try:
            expected_sources.extend(
                [f"btag_weight_{unc}" for unc in cfg0.x.btag_unc_names]
            )
        except Exception:
            pass

        shape_sources = []
        for src in expected_sources:
            if src in lumi_uncs or src == "nominal":
                continue
            if src not in shape_sources and any(_has_shift_source(cfg, src) for cfg in config_insts):
                shape_sources.append(src)

        exp_group = (
            ["experiment", "shape_nuisances"]
            if hasattr(self, "add_parameter_group")
            else "experiment"
        )
        th_group = (
            ["theory", "shape_nuisances"]
            if hasattr(self, "add_parameter_group")
            else "theory"
        )

        def _is_theory_like(src: str) -> bool:
            return src in theory_shape_sources

        added = {}
        for src in shape_sources:
            proc_scope = _process_scope(src)
            if not proc_scope:
                continue

            nuis = _nuis_name(src)
            if nuis in added and added[nuis] != src:
                raise ValueError(
                    f"nuisance name collision: '{nuis}' would be used for both "
                    f"'{added[nuis]}' and '{src}'. Adjust _nuis_name mapping."
                )
            added[nuis] = src

            config_data = {
                cfg.name: self.parameter_config_spec(shift_source=src)
                for cfg in config_insts
                if _has_shift_source(cfg, src)
            }
            if not config_data:
                continue

            self.add_parameter(
                nuis,
                type=ParameterType.shape,
                config_data=config_data,
                process=proc_scope,
                group=(th_group if _is_theory_like(src) else exp_group),
            )

        # ---------------------------------------------------------------------
        # explicit safety: remove shape nuisances from qcd only
        # ---------------------------------------------------------------------

        if self.add_qcd and not self.use_qcd_shape_uncertainties:
            for category_name, process_name, parameter in list(self.iter_parameters()):
                if process_name not in ("qcd", "QCD"):
                    continue

                remove = (
                    parameter.type.is_shape
                    or parameter.transformations.any_from_shape
                )

                if remove:
                    self.remove_parameter(
                        parameter.name,
                        process=process_name,
                        category=category_name,
                    )


# -----------------------------------------------------------------------------
# inference-model variants
# -----------------------------------------------------------------------------


@MSSM_model.inference_model
def MSSM_model_no_shifts(self):
    print("Producing inference models without shape-based shifts")

    super(MSSM_model_no_shifts, self).init_func()

    for category_name, process_name, parameter in self.iter_parameters():
        remove = (
            (parameter.type.is_shape and not parameter.transformations.any_from_rate)
            or (parameter.type.is_rate and parameter.transformations.any_from_shape)
        )
        if remove:
            self.remove_parameter(
                parameter.name,
                process=process_name,
                category=category_name,
            )

    self.init_cleanup()


@MSSM_model.inference_model(empty_bin_value=0)
def MSSM_model_bin_opt(self):
    super(MSSM_model_bin_opt, self).init_func()

    keep_parameters = {
        "BR_*",
        "QCDscale_*",
        "bbphi_norm_*",
        "lumi_*",
        "CMS_bbtt_eff_trig_*",
        "CMS_btag_*",
        "CMS_eff_e_*",
        "CMS_eff_mu_*",
        "CMS_eff_t_*",
        "CMS_pu_*",
        "CMS_res_j",
        "CMS_scale_j_*",
        "CMS_scale_met_unclustered",
        "CMS_met_recoil_*",
        "CMS_top_pT_reweighting",
        "CMS_zpt_reweighting",
        "CMS_PS_*",
        "CMS_Scale_*",
        "pdf_*",
        "ps_*",
        "scale_*",
    }

    for category_name, process_name, parameter in self.iter_parameters():
        if not law.util.multi_match(parameter.name, keep_parameters):
            self.remove_parameter(
                parameter.name,
                process=process_name,
                category=category_name,
            )

    self.init_cleanup()


# -----------------------------------------------------------------------------
# mass- and production-specific derived models
# -----------------------------------------------------------------------------

for _m in read_bdt_masses():
    globals()[f"MSSM_model_M{_m}"] = MSSM_model.derive(
        f"MSSM_model_M{_m}",
        cls_dict={
            "signal_mass": _m,
            "signal_kind": None,
        },
    )

    globals()[f"MSSM_model_ggphi_M{_m}"] = MSSM_model.derive(
        f"MSSM_model_ggphi_M{_m}",
        cls_dict={
            "signal_mass": _m,
            "signal_kind": "ggphi",
        },
    )

    globals()[f"MSSM_model_bbphi_M{_m}"] = MSSM_model.derive(
        f"MSSM_model_bbphi_M{_m}",
        cls_dict={
            "signal_mass": _m,
            "signal_kind": "bbphi",
        },
    )