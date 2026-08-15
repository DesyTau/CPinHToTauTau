# coding: utf-8

"""
Inference model for the MSSM analysis.
"""

import law
import re

from columnflow.inference import inference_model, ParameterType
from columnflow.config_util import get_datasets_from_process
from MSSM_H_tt.inference.base import HCPModelBase
from MSSM_H_tt.config.mass_points import (
    read_bdt_masses,
    get_bdt_mass_block,
    get_bdt_card_producer_name,
)


# -----------------------------------------------------------------------------
# BDT histogram grouping
# -----------------------------------------------------------------------------

# Final variables needed by the four datacards at each mass point.
BDT_CARD_VARIABLES = (
    "D_sig_vs_Disc_ggphi",
    "D_sig_vs_Disc_bbphi",
    "D_DY",
    "D_TT",
)

# Number of neighbouring mass points whose histograms are produced together.
#
# With the current mass list and block size 6:
#
#   block 0: 60, 65, 70, 75, 80, 85
#   block 1: 90, 95, 100, 105, 110, 115
#   block 2: 120, 125, 130, 135, 140, 160
#   ...
#
    
class MSSM_model(HCPModelBase):
    """
    Default statistical model for MSSM analysis.
    """

    @staticmethod
    def _bdt_hist_group_for_variable(variable: str) -> set[str]:
        """
        For any final MSSM BDT datacard variable, return the complete set
        of datacard variables belonging to the same mass block.

        Example for block size 6:

            bdt_D_DY_M100

        expands to the four datacard variables for

            M90, M95, M100, M105, M110, M115.

        This affects only histogram requirements. It does not change the
        datacard category, signal mass, BDT response, or physics definition.
        """
        match = re.match(
            r"^bdt_(D_sig_vs_Disc_ggphi|D_sig_vs_Disc_bbphi|D_DY|D_TT)_M([0-9]+)$",
            variable,
        )

        # Non-BDT variables are left untouched.
        if not match:
            return {variable}

        mass = int(match.group(2))
        masses = list(read_bdt_masses())

        if mass not in masses:
            raise ValueError(
                f"BDT mass {mass} is not present in the configured mass points: "
                f"{masses}"
            )

        # Find the block containing this mass.
        block_masses = get_bdt_mass_block(
            mass
        )

        # Crucially, every mass belonging to the same block returns exactly
        # the same set of histogram variables.
        return {
            f"bdt_{discriminant}_M{block_mass}"
            for block_mass in block_masses
            for discriminant in BDT_CARD_VARIABLES
        }

    def get_hist_requirement_variables(self, variables: set[str]) -> set[str]:
        """
        Expand the variables requested by the inference model into complete
        BDT mass blocks.

        This allows several mass hypotheses to share the same upstream
        CreateHistograms / MergeHistograms tasks.
        """
        out = set()

        for variable in variables:
            out |= self._bdt_hist_group_for_variable(variable)

        return out
    def get_hist_requirement_producers(
        self,
        variables: set[str],
        default_producers: tuple[str, ...],
        ) -> tuple[str, ...]:
        """
        For a one-mass MSSM datacard, use the common
        producer plus the BDT producer corresponding to
        that mass block.
        """

        masses = set()

        for variable in variables:
            match = re.match(
                r"^bdt_"
                r"(D_sig_vs_Disc_ggphi|"
                r"D_sig_vs_Disc_bbphi|"
                r"D_DY|D_TT)"
                r"_M([0-9]+)$",
                variable,
            )

            if match:
                masses.add(
                    int(match.group(2))
                )

        # Nothing BDT-specific.
        if not masses:
            return tuple(default_producers)

        blocks = {
            tuple(get_bdt_mass_block(mass))
            for mass in masses
        }

        # The unspecialized model can contain all masses.
        # Keep the normal producer behavior in that case.
        if len(blocks) != 1:
            return tuple(default_producers)

        mass = next(iter(masses))

        bdt_producer = (
            get_bdt_card_producer_name(mass)
        )

        # Remove the old full producer if present.
        other_producers = [
            producer
            for producer in default_producers
            if (
                producer != "main"
                and not producer.startswith(
                    "bdt_card_"
                )
                and producer != "main_common"
            )
        ]

        return tuple(
            [
                *other_producers,
                "main_common",
                bdt_producer,
            ]
        )

    name = "MSSM_model"
    add_qcd = True

    # Keep qcd in the datacard, but do not attach shape nuisances to it.
    # Set to True only if you really want qcd to receive shape systematics.
    use_qcd_shape_uncertainties = False

    # Keep the combine/datacard process name explicit and consistent.
    qcd_combine_name = "qcd"

    # Specialization knobs for derived inference models.
    signal_mass = None          # e.g. 100
    signal_kind = None          # None, "ggphi", or "bbphi"

    # Supported canonical values:
    #
    #   None
    #   "D_sig_vs_Disc_ggphi"
    #   "D_sig_vs_Disc_bbphi"
    #   "D_DY"
    #   "D_TT"
    #
    # Backward-compatible aliases such as "sig_vs_disc_ggphi",
    # "sig_vs_disc_bbphi", "dy", and "tt" are normalized below.
    bdt_discriminant = None

    # Canonical names used for BDT datacard categories and variables.
    #
    # Region names should match the category-config names:
    #
    #   cat_{ch}_sr__bdt_signal_M{mass}
    #   cat_{ch}_sr__bdt_dy_M{mass}
    #   cat_{ch}_sr__bdt_tt_M{mass}
    #
    # The previous signal-like region name,
    #
    #   cat_{ch}_sr__bdt_ggphi_and_bbphi_M{mass}
    #
    # is kept only as a fallback alias.
    bdt_discriminant_aliases = {
        "sig_vs_disc_ggphi": "D_sig_vs_Disc_ggphi",
        "sig_vs_disc_bbphi": "D_sig_vs_Disc_bbphi",
        "dy": "D_DY",
        "tt": "D_TT",
    }

    bdt_card_specs = {
        "D_sig_vs_Disc_ggphi": {
            "region": "signal",
            "variable": "D_sig_vs_Disc_ggphi",
        },
        "D_sig_vs_Disc_bbphi": {
            "region": "signal",
            "variable": "D_sig_vs_Disc_bbphi",
        },
        "D_DY": {
            "region": "dy",
            "variable": "D_DY",
        },
        "D_TT": {
            "region": "tt",
            "variable": "D_TT",
        },
    }

    bdt_region_aliases = {
        "signal": ("signal", "ggphi_and_bbphi"),
        "dy": ("dy",),
        "tt": ("tt",),
    }

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

    def _resolve_representative_process(
        self,
        config_inst,
        preferred_process,
        dataset_processes,
    ):
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
            "emu": ["data_egamma_", "data_mu_"],
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

    def _normalize_bdt_discriminant(self, discriminant):
        if discriminant is None:
            return None

        return self.bdt_discriminant_aliases.get(discriminant, discriminant)

    def _bdt_region_category_candidates(self, ch: str, region: str, mass) -> list[str]:
        region_aliases = self.bdt_region_aliases.get(region, (region,))

        return [
            f"cat_{ch}_sr__bdt_{region_name}_M{mass}"
            for region_name in region_aliases
        ]

    def _resolve_bdt_region_category(self, config_inst, ch: str, region: str, mass) -> str:
        candidates = self._bdt_region_category_candidates(ch, region, mass)

        for category_name in candidates:
            try:
                config_inst.get_category(category_name)
                return category_name
            except Exception:
                pass

        raise ValueError(
            f"Could not find any BDT category for region '{region}' and mass {mass} "
            f"in config '{config_inst.name}'. Tried: {candidates}"
        )

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

        # New merged-region BDT datacard mode.
        #
        # Canonical region names:
        #
        #   signal : P_ggphi + P_bbphi is maximal against P_DY and P_TT
        #   dy     : P_DY is maximal against P_ggphi + P_bbphi and P_TT
        #   tt     : P_TT is maximal against P_ggphi + P_bbphi and P_DY
        #
        # Canonical category names are:
        #
        #   cat_{ch}_sr__bdt_signal_M{mass}
        #   cat_{ch}_sr__bdt_dy_M{mass}
        #   cat_{ch}_sr__bdt_tt_M{mass}
        #
        # The old signal-like name
        #
        #   cat_{ch}_sr__bdt_ggphi_and_bbphi_M{mass}
        #
        # is accepted as a fallback while transitioning configs.
        #
        # Produced variable names are kept unchanged because they match the
        # BDT-score and BDT-2D producers:
        #
        #   bdt_D_sig_vs_Disc_ggphi_M{mass}
        #   bdt_D_sig_vs_Disc_bbphi_M{mass}
        #   bdt_D_DY_M{mass}
        #   bdt_D_TT_M{mass}
        if self.bdt_discriminant is not None:
            discriminant = self._normalize_bdt_discriminant(self.bdt_discriminant)

            if discriminant not in self.bdt_card_specs:
                valid = sorted(
                    set(self.bdt_card_specs.keys())
                    | set(self.bdt_discriminant_aliases.keys())
                )

                raise ValueError(
                    f"Invalid bdt_discriminant '{self.bdt_discriminant}'. "
                    f"Valid values are: {valid}"
                )

            masses = self.get_mass_points()

            if len(masses) != 1:
                raise ValueError(
                    "The one-category datacard setup requires exactly one mass point. "
                    "Use a derived model with signal_mass set, e.g. "
                    "MSSM_model_D_sig_vs_Disc_ggphi_M100."
                )

            mass = masses[0]
            card_spec = self.bdt_card_specs[discriminant]

            region = card_spec["region"]
            variable_name = f"bdt_{card_spec['variable']}_M{mass}"

            # Use the first config to define the inference-category name.  The
            # per-config category names below can still resolve independently,
            # which makes mixed old/new configs possible during transition.
            category_name = self._resolve_bdt_region_category(
                config_insts[0],
                ch,
                region,
                mass,
            )

            config_data = {}

            for config_inst in config_insts:
                data_datasets = self._get_data_datasets(config_inst, ch)
                cfg_category_name = self._resolve_bdt_region_category(
                    config_inst,
                    ch,
                    region,
                    mass,
                )

                config_data[config_inst.name] = self.category_config_spec(
                    category=cfg_category_name,
                    variable=variable_name,
                    data_datasets=data_datasets,
                )

            self.add_category(
                name=category_name,
                config_data=config_data,
                mc_stats=True,
                empty_bin_value=0.0,
            )

            return

        # Unspecialized base model: use the same merged three-region naming.
        # This mode is mostly for checks; production datacards should normally
        # use one of the mass/discriminant-specific derived models below.
        base_category_specs = [
            ("signal", "D_sig"),
            ("dy", "D_DY"),
            ("tt", "D_TT"),
        ]

        for mass in self.get_mass_points():
            for region, variable in base_category_specs:
                config_data = {}

                category_name = self._resolve_bdt_region_category(
                    config_insts[0],
                    ch,
                    region,
                    mass,
                )

                for config_inst in config_insts:
                    data_datasets = self._get_data_datasets(config_inst, ch)
                    cfg_category_name = self._resolve_bdt_region_category(
                        config_inst,
                        ch,
                        region,
                        mass,
                    )

                    config_data[config_inst.name] = self.category_config_spec(
                        category=cfg_category_name,
                        variable=f"bdt_{variable}_M{mass}",
                        data_datasets=data_datasets,
                    )

                self.add_category(
                    name=category_name,
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

                eff = lumi.get(
                    names=unc_name,
                    direction=("down", "up"),
                    factor=True,
                )

                if ref_eff is None:
                    ref_eff = eff
                    ref_cfg = cfg.name
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
            if self.use_qcd_shape_uncertainties:
                return list(all_processes)

            return list(non_qcd_processes)

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
                theory_process_patterns = (
                    cfg0.x.theory_uncertainty_processes.get(src, ())
                )

                return [
                    p for p in default
                    if law.util.multi_match(
                        p,
                        theory_process_patterns,
                    )
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

            if src not in shape_sources and any(
                _has_shift_source(cfg, src) for cfg in config_insts
            ):
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
    # -------------------------------------------------------------------------
    # 1. ggphi extraction
    #
    # category:
    #   cat_{ch}_sr__bdt_signal_M{mass}
    #
    # variable:
    #   bdt_D_sig_vs_Disc_ggphi_M{mass}
    #
    # signal:
    #   ggphi_phitt_{mass}
    # -------------------------------------------------------------------------

    globals()[f"MSSM_model_D_sig_vs_Disc_ggphi_M{_m}"] = MSSM_model.derive(
        f"MSSM_model_D_sig_vs_Disc_ggphi_M{_m}",
        cls_dict={
            "signal_mass": _m,
            "signal_kind": "ggphi",
            "bdt_discriminant": "D_sig_vs_Disc_ggphi",
        },
    )

    # -------------------------------------------------------------------------
    # 2. bbphi extraction
    #
    # category:
    #   cat_{ch}_sr__bdt_signal_M{mass}
    #
    # variable:
    #   bdt_D_sig_vs_Disc_bbphi_M{mass}
    #
    # signal:
    #   bbphi_phitt_{mass}
    # -------------------------------------------------------------------------

    globals()[f"MSSM_model_D_sig_vs_Disc_bbphi_M{_m}"] = MSSM_model.derive(
        f"MSSM_model_D_sig_vs_Disc_bbphi_M{_m}",
        cls_dict={
            "signal_mass": _m,
            "signal_kind": "bbphi",
            "bdt_discriminant": "D_sig_vs_Disc_bbphi",
        },
    )

    # -------------------------------------------------------------------------
    # 3. DY-region datacard
    #
    # category:
    #   cat_{ch}_sr__bdt_dy_M{mass}
    #
    # variable:
    #   bdt_D_DY_M{mass}
    #
    # signal:
    #   both ggphi_phitt_{mass} and bbphi_phitt_{mass}
    # -------------------------------------------------------------------------

    globals()[f"MSSM_model_D_DY_M{_m}"] = MSSM_model.derive(
        f"MSSM_model_D_DY_M{_m}",
        cls_dict={
            "signal_mass": _m,
            "signal_kind": None,
            "bdt_discriminant": "D_DY",
        },
    )

    # -------------------------------------------------------------------------
    # 4. TT-region datacard
    #
    # category:
    #   cat_{ch}_sr__bdt_tt_M{mass}
    #
    # variable:
    #   bdt_D_TT_M{mass}
    #
    # signal:
    #   both ggphi_phitt_{mass} and bbphi_phitt_{mass}
    # -------------------------------------------------------------------------

    globals()[f"MSSM_model_D_TT_M{_m}"] = MSSM_model.derive(
        f"MSSM_model_D_TT_M{_m}",
        cls_dict={
            "signal_mass": _m,
            "signal_kind": None,
            "bdt_discriminant": "D_TT",
        },
    )