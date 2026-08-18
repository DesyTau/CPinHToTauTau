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

from MSSM_H_tt.config.mass_points import read_bdt_masses
from MSSM_H_tt.tasks.base import HTTCPTask


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

    def get_bdt_masses(self) -> tuple[int, ...]:
        """
        Return the requested BDT masses while checking that
        they are present in the central mass configuration.
        """

        configured_masses = tuple(
            int(mass)
            for mass in read_bdt_masses()
        )

        if not self.bdt_masses:
            return configured_masses

        requested_masses = tuple(
            int(mass)
            for mass in self.bdt_masses
        )

        unknown_masses = (
            set(requested_masses)
            - set(configured_masses)
        )

        if unknown_masses:
            raise ValueError(
                "requested BDT masses are not configured: "
                f"{sorted(unknown_masses)}; "
                f"available masses: {configured_masses}"
            )

        return requested_masses

    def get_plot_task(
        self,
        category: str,
        variables: tuple[str, ...],
    ):
        """
        Create one PlotShiftedVariables1D workflow while
        forwarding all common parameters from this wrapper.

        In particular, this propagates configs, processes,
        datasets, shift sources, version, workflow settings,
        plot settings, pilot mode, etc.
        """

        return self.clone(
            PlotShiftedVariables1D,
            categories=(category,),
            variables=variables,
            branch=-1,
        )

    def requires(self):

        if not self.datasets:
            raise ValueError(
                "PlotDatacardDistributions received no datasets. "
                "Check the --datasets argument."
            )

        if not self.processes:
            raise ValueError(
                "PlotDatacardDistributions received no processes. "
                "Check the --processes argument."
            )

        if not self.shift_sources:
            raise ValueError(
                "PlotDatacardDistributions received no shift sources. "
                "Check the --shift-sources argument."
            )

        reqs = {}


        channel = self.channel

        #
        # Inclusive SR
        #

        if self.include_inclusive:

            inclusive_variables = (
                tuple(self.inclusive_variables)
                if self.inclusive_variables
                else (
                    f"{channel}_mt_tot",
                )
            )

            reqs["inclusive_sr"] = (
                self.get_plot_task(
                    category=f"cat_{channel}_sr",
                    variables=inclusive_variables,
                )
            )

        #
        # BDT categories
        #

        for mass in self.get_bdt_masses():

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
            )

        return reqs