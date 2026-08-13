# coding: utf-8

"""
Collection of patches of underlying columnflow tasks.
"""

import os

import law
from columnflow.util import memoize


logger = law.logger.get_logger(__name__)


@memoize
def patch_bundle_repo_exclude_files():
    from columnflow.tasks.framework.remote import BundleRepo

    # get the relative path to CF_BASE
    cf_rel = os.path.relpath(os.environ["CF_BASE"], os.environ["HTTCP_BASE"])

    # amend exclude files to start with the relative path to CF_BASE
    exclude_files = [os.path.join(cf_rel, path) for path in BundleRepo.exclude_files]

    # add additional files
    exclude_files.extend([
        "docs", "tests", "data", "assets", ".law", ".setups", ".data", ".github",
    ])

    # overwrite them
    BundleRepo.exclude_files[:] = exclude_files

    logger.debug("patched exclude_files of cf.BundleRepo")

@memoize
def patch_inference_hist_requirements():
    """
    Patch ColumnFlow inference histogram requirements to:

      1. expand one requested BDT datacard variable into the full variable group
         defined by the inference model;

      2. request only the unique shift sources actually needed by each
         process/config in the selected inference model.

    This allows different BDT datacards of the same mass to share the same
    upstream histogram production.
    """
    from columnflow.tasks.framework.inference import SerializeInferenceModelBase

    def _get_requirement_variables(self, config_data):
        variables = {config_data.variable}

        expand_variables = getattr(
            self.inference_model_inst,
            "get_hist_requirement_variables",
            None,
        )

        if callable(expand_variables):
            variables = set(expand_variables(variables))

        # Sorting is important: all four datacard models must generate exactly
        # the same task parameter representation.
        return tuple(sorted(variables))

    def _get_requirement_shift_sources(self, config_inst, proc_obj):
        # Only shape sources that are actually required by parameters attached
        # to this process in this config.
        shift_sources = {
            param_obj.config_data[config_inst.name].shift_source
            for param_obj in proc_obj.parameters
            if (
                config_inst.name in param_obj.config_data
                and self.inference_model_inst.require_shapes_for_parameter(param_obj)
            )
        }

        # Remove duplicates and enforce deterministic ordering so otherwise
        # identical histogram tasks are shared between inference models.
        return tuple(sorted(shift_sources))

    def _requires_cat_obj(self, cat_obj, **req_kwargs):
        reqs = {}

        for config_inst in self.config_insts:
            config_data = cat_obj.config_data.get(config_inst.name)

            if not config_data:
                continue

            variables = _get_requirement_variables(self, config_data)

            reqs[config_inst.name] = {}

            # -----------------------------------------------------------------
            # MC
            # -----------------------------------------------------------------

            for proc_obj in cat_obj.processes:
                if config_inst.name not in proc_obj.config_data:
                    continue

                # Dynamic processes such as QCD do not require their own
                # histogram production.
                if proc_obj.is_dynamic:
                    continue

                datasets = self.get_mc_datasets(config_inst, proc_obj)

                if not datasets:
                    continue

                shift_sources = _get_requirement_shift_sources(
                    self,
                    config_inst,
                    proc_obj,
                )

                logger.debug(
                    f"inference histogram requirements for "
                    f"{config_inst.name}/{proc_obj.name}: "
                    f"variables={variables}, "
                    f"shift_sources={shift_sources}"
                )

                reqs[config_inst.name][proc_obj.name] = {
                    dataset: self.reqs.MergeShiftedHistograms.req_different_branching(
                        self,
                        config=config_inst.name,
                        dataset=dataset,
                        shift_sources=shift_sources,
                        variables=variables,
                        **req_kwargs,
                    )
                    for dataset in datasets
                }

            # -----------------------------------------------------------------
            # data
            # -----------------------------------------------------------------

            # Data is needed when:
            #   - the category uses real data, or
            #   - at least one process is dynamic, e.g. QCD estimation.
            if (
                (
                    not cat_obj.data_from_processes
                    or any(proc_obj.is_dynamic for proc_obj in cat_obj.processes)
                )
                and (
                    data_datasets := self.get_data_datasets(
                        config_inst,
                        cat_obj,
                    )
                )
            ):
                reqs[config_inst.name]["data"] = {
                    dataset: self.reqs.MergeHistograms.req_different_branching(
                        self,
                        config=config_inst.name,
                        dataset=dataset,
                        variables=variables,
                        **req_kwargs,
                    )
                    for dataset in data_datasets
                }

        return reqs

    SerializeInferenceModelBase._requires_cat_obj = _requires_cat_obj

    logger.debug(
        "patched inference histogram requirements for grouped variables "
        "and pruned shift sources"
    )
@memoize
def patch_all():
    patch_bundle_repo_exclude_files()
    patch_inference_hist_requirements()
