# coding: utf-8

"""
Wrapper task to schedule all MSSM datacard distribution plots
within a single LAW / Luigi dependency graph.
"""

from __future__ import annotations

import law
import luigi

from luigi.util import inherits

from columnflow.tasks.plotting import PlotShiftedVariables1D

from MSSM_H_tt.config.mass_points import (
    read_bdt_masses,
    get_mssm_signal_mass,
)
from MSSM_H_tt.tasks.base import HTTCPTask


def _filter_names_for_mass(
    names,
    mass: int,
) -> tuple[str, ...]:
    """
    Keep all backgrounds and data, but only MSSM signals
    corresponding to the requested mass.
    """

    mass = int(
        mass
    )

    filtered = []

    for name in names:

        signal_mass = get_mssm_signal_mass(
            name
        )

        # Background or data.
        if signal_mass is None:
            filtered.append(
                name
            )
            continue

        # MSSM signal of the requested mass.
        if signal_mass == mass:
            filtered.append(
                name
            )

    return tuple(
        filtered
    )


@inherits(PlotShiftedVariables1D)
class PlotDatacardDistributions(
    HTTCPTask,
    law.WrapperTask,
):

    channel = luigi.ChoiceParameter(
        default="emu",
        choices=(
            "emu",
            "mutau",
            "etau",
            "tautau",
        ),
        description="analysis channel",
    )

    bdt_masses = law.CSVParameter(
        default=(),
        parse_empty=True,
        description=(
            "optional comma-separated subset of BDT masses; "
            "when empty, use all configured masses"
        ),
    )

    include_inclusive = luigi.BoolParameter(
        default=True,
        description=(
            "include the inclusive SR distribution; "
            "default: True"
        ),
    )

    inclusive_variables = law.CSVParameter(
        default=(),
        parse_empty=True,
        description=(
            "inclusive variables to plot; when empty, "
            "use <channel>_mt_tot"
        ),
    )

    def get_bdt_masses(
        self,
    ) -> tuple[int, ...]:
        """
        Return the requested BDT masses while checking that
        they are present in the central mass configuration.
        """

        configured_masses = tuple(
            int(mass)
            for mass
            in read_bdt_masses()
        )

        if not self.bdt_masses:
            return configured_masses

        requested_masses = tuple(
            int(mass)
            for mass
            in self.bdt_masses
        )

        unknown_masses = (
            set(requested_masses)
            -
            set(configured_masses)
        )

        if unknown_masses:
            raise ValueError(
                "requested BDT masses "
                "are not configured: "
                f"{sorted(unknown_masses)}; "
                "available masses: "
                f"{configured_masses}"
            )

        return requested_masses

    def get_mass_inputs(
        self,
        mass: int,
    ):
        """
        Return the dataset and process selections for one
        mass-dependent BDT plot.

        All backgrounds and data are retained.

        MSSM signals are retained only when their mass matches
        the requested BDT mass.
        """

        datasets = tuple(
            _filter_names_for_mass(
                config_datasets,
                mass,
            )
            for config_datasets
            in self.datasets
        )

        processes = tuple(
            _filter_names_for_mass(
                config_processes,
                mass,
            )
            for config_processes
            in self.processes
        )

        return (
            datasets,
            processes,
        )

    def get_plot_task(
        self,
        category: str,
        variables: tuple[str, ...],
        mass: int | None = None,
    ):
        """
        Create one PlotShiftedVariables1D workflow.

        For BDT plots, all backgrounds and data are retained
        while MSSM signals are filtered to the requested mass.

        For inclusive plots, mass is None and the complete
        original process/dataset selection is preserved.
        """

        kwargs = {
            "categories": (
                category,
            ),
            "variables":
                variables,
            "branch":
                -1,
        }

        if mass is not None:

            (
                datasets,
                processes,
            ) = self.get_mass_inputs(
                mass
            )

            kwargs.update({
                "datasets":
                    datasets,
                "processes":
                    processes,
            })

        return self.clone(
            PlotShiftedVariables1D,
            **kwargs,
        )

    def requires(
        self,
    ):

        if not self.datasets:
            raise ValueError(
                "PlotDatacardDistributions "
                "received no datasets. "
                "Check the --datasets argument."
            )

        if not self.processes:
            raise ValueError(
                "PlotDatacardDistributions "
                "received no processes. "
                "Check the --processes argument."
            )

        if not self.shift_sources:
            raise ValueError(
                "PlotDatacardDistributions "
                "received no shift sources. "
                "Check the --shift-sources argument."
            )

        reqs = {}

        channel = self.channel

        #
        # Inclusive SR
        #

        if self.include_inclusive:

            inclusive_variables = (
                tuple(
                    self.inclusive_variables
                )
                if self.inclusive_variables
                else (
                    f"{channel}_mt_tot",
                )
            )

            reqs[
                "inclusive_sr"
            ] = self.get_plot_task(
                category=(
                    f"cat_{channel}_sr"
                ),
                variables=(
                    inclusive_variables
                ),
            )

        #
        # BDT categories
        #

        for mass in (
            self.get_bdt_masses()
        ):

            #
            # merged ggphi + bbphi signal region
            #

            reqs[
                f"M{mass}_ggphi_and_bbphi"
            ] = self.get_plot_task(
                category=(
                    f"cat_{channel}_sr"
                    f"__bdt_ggphi_and_bbphi_M{mass}"
                ),
                variables=(
                    f"bdt_D_sig_vs_Disc_ggphi_M{mass}",
                    f"bdt_D_sig_vs_Disc_bbphi_M{mass}",
                ),
                mass=mass,
            )

            #
            # DY region
            #

            reqs[
                f"M{mass}_dy"
            ] = self.get_plot_task(
                category=(
                    f"cat_{channel}_sr"
                    f"__bdt_dy_M{mass}"
                ),
                variables=(
                    f"bdt_D_DY_M{mass}",
                ),
                mass=mass,
            )

            #
            # tt region
            #

            reqs[
                f"M{mass}_tt"
            ] = self.get_plot_task(
                category=(
                    f"cat_{channel}_sr"
                    f"__bdt_tt_M{mass}"
                ),
                variables=(
                    f"bdt_D_TT_M{mass}",
                ),
                mass=mass,
            )

        return reqs