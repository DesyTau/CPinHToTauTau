# coding: utf-8

"""
Definition of categories.
"""

import order as od

from columnflow.config_util import add_category


def add_categories(config: od.Config) -> None:
    """
    Adds all categories to a *config*.
    """
    add_category(
        config,
        name="incl",
        id=1,
        selection="cat_incl",
        label="inclusive",
    )
    add_category(
        config,
        name="2j",
        id=100,
        selection="cat_2j",
        label="2 jets",
    )
    add_category(
        config,
        name="etau",
        id=101,
        selection="sel_etau",
        label="etau_channel",
    )
    add_category(
        config,
        name="mutau",
        id=102,
        selection="sel_mutau",
        label="mutau_channel",
    )
    add_category(
        config,
        name="tautau",
        id=103,
        selection="sel_tautau",
        label="tautau_channel",
    )
