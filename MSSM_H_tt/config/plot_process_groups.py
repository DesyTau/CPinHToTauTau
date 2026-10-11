# coding: utf-8
"""Build grouped plotting processes from the configuration's component lists.

Call add_grouped_plot_processes(cfg) after registering processes, datasets and
cfg.x.process_groups. Plot with --processes plot_backgrounds_grouped.

Each plot_<group> process contains the configured members and their descendants.
The original physics process IDs and dataset assignments are unchanged.
"""

import warnings

GROUPED_PLOT_GROUPS = (
    "dy_lep",
    "dy_tt_m50",
    "tt",
    "st",
    "wjets",
    "vv",
    "vvv",
    "SM_higgs",
    "vh_htt",
)

PLOT_LABELS = {
    "dy_lep": r"DY $\rightarrow\ell\ell$",
    "dy_tt_m50": r"$Z\rightarrow\tau\tau$",
    "tt": r"$t\bar{t}$",
    "st": "Single top",
    "wjets": "W + jets",
    "vv": "VV",
    "vvv": "VVV",
    "SM_higgs": r"$H_{\mathrm{SM}}\rightarrow\tau\tau$",
    "vh_htt": r"$VH\rightarrow\tau\tau$",
}

# Exact hex codes from the supplied 10-colour palette, shared across all eras.
# QCD has its own reserved colour when a hook adds a process named "qcd".
PLOT_COLORS = {
    "dy_lep": "#3f90da",       # kP10Blue
    "dy_tt_m50": "#e76300",    # kP10Orange
    "tt": "#bd1f01",           # kP10Red
    "st": "#b9ac70",           # kP10Green
    "wjets": "#832db6",        # kP10Violet
    "vv": "#92dadd",           # kP10Cyan
    "vvv": "#94a4a2",          # kP10Gray
    "SM_higgs": "#ffa90e",     # kP10Yellow
    "vh_htt": "#a96b59",       # kP10Brown
    "qcd": "#717581",          # kP10Ash
}


def _color_rgb(hex_color):
    """Convert a six-digit hex colour to an RGB tuple accepted by order."""
    if len(hex_color) != 7 or not hex_color.startswith("#"):
        raise ValueError(f"invalid plotting colour: {hex_color!r}")
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def _resolve_plot_members(cfg, group_name, component_names):
    """Resolve process entries and dataset entries to registered processes."""
    members_by_id = {}
    unresolved = []
    for name in dict.fromkeys(component_names):
        if cfg.has_process(name, deep=True):
            processes = [cfg.get_process(name)]
        elif cfg.has_dataset(name):
            # For example, TbarWplusto4Q is a dataset; its process has a
            # different name, st_twchannel_tbar_fh, in this configuration.
            processes = list(cfg.get_dataset(name).processes)
            if not processes:
                unresolved.append(f"{name} (dataset has no processes)")
        else:
            unresolved.append(name)
            continue

        for process in processes:
            if not cfg.has_process(process.name, deep=True):
                raise ValueError(
                    f"process '{process.name}' from component '{name}' of "
                    f"plot group '{group_name}' is not registered in the config"
                )
            registered_process = cfg.get_process(process.name)
            if registered_process.id != process.id:
                raise ValueError(
                    f"inconsistent process IDs for '{process.name}' in "
                    f"plot group '{group_name}'"
                )
            members_by_id.setdefault(process.id, registered_process)

    if unresolved:
        message = (
            f"plot group '{group_name}': components not resolved as processes "
            f"or datasets: {', '.join(unresolved)}"
        )
        if not members_by_id:
            raise ValueError(message)
        warnings.warn(message + "; these components are skipped", RuntimeWarning, stacklevel=2)

    return list(members_by_id.values())


# Reserve this range for plotting parents. IDs must be unique within the config.
PLOT_PROCESS_ID_BASE = 950_000_000


def add_grouped_plot_processes(cfg):
    """Create one plotting parent per configured background group.

    Existing component aliases are kept. The *_components fallback accepts
    configurations in which the previous helper already renamed these aliases.
    Component entries can be process names or registered dataset names. Missing
    entries emit a warning; a nonempty group that cannot be resolved raises an
    error rather than silently disappearing from the plot.
    """
    groups = cfg.x.process_groups

    # Fail immediately if a future palette edit gives two backgrounds the same
    # RGB colour. Checking RGB values also catches different hex letter cases.
    background_colors = [
        _color_rgb(PLOT_COLORS[name])
        for name in (*GROUPED_PLOT_GROUPS, "qcd")
    ]
    if len(set(background_colors)) != len(background_colors):
        raise ValueError("each plotted background must have a different colour")
    process_colors = {"data": "#000000", "qcd": PLOT_COLORS["qcd"]}

    # Resolve and validate all members before adding plotting parents.
    resolved = {}
    process_owners = {}
    for group_name in GROUPED_PLOT_GROUPS:
        component_names = groups.get(group_name)
        if component_names is None:
            component_names = groups.get(f"{group_name}_components")
        if component_names is None:
            raise KeyError(
                f"missing process group '{group_name}' in cfg.x.process_groups"
            )
        if not isinstance(component_names, (list, tuple)):
            raise TypeError(f"process group '{group_name}' must be a list or tuple")

        members = _resolve_plot_members(cfg, group_name, component_names)
        if any(proc.is_data for proc in members):
            raise ValueError(f"background group '{group_name}' contains data")

        # Shared descendants within one group are allowed; across groups they
        # would duplicate event contributions and must be rejected.
        for member in members:
            for proc, _, _ in member.walk_processes(include_self=True):
                previous_group = process_owners.get(proc.id)
                if previous_group is not None and previous_group != group_name:
                    raise ValueError(
                        f"process '{proc.name}' belongs to both "
                        f"'{previous_group}' and '{group_name}'"
                    )
                process_owners[proc.id] = group_name
        resolved[group_name] = members

    selected = []
    if cfg.has_process("data", deep=True):
        # Keep the physical data process as the histogram key for data hooks.
        # A same-named alias would expand it into separate data components.
        if "data" in groups:
            groups["data_components"] = groups.pop("data")
        cfg.get_process("data").label = "Data"
        selected.append("data")

    for index, group_name in enumerate(GROUPED_PLOT_GROUPS):
        members = resolved[group_name]
        if not members:
            continue

        plot_name = f"plot_{group_name}"
        plot_id = PLOT_PROCESS_ID_BASE + index

        if plot_name in groups:
            raise ValueError(f"process alias '{plot_name}' shadows a plotting parent")

        if cfg.has_process(plot_name, deep=True):
            parent = cfg.get_process(plot_name)
            if parent.x("grouped_plot_group", None) != group_name:
                raise ValueError(f"process name '{plot_name}' is already in use")
            parent.clear_processes()
        else:
            if cfg.has_process(plot_id, deep=True):
                raise ValueError(
                    f"process ID {plot_id} is already in use; "
                    "change PLOT_PROCESS_ID_BASE"
                )
            parent = cfg.add_process(
                name=plot_name,
                id=plot_id,
                is_data=False,
                label=PLOT_LABELS[group_name],
                color=_color_rgb(PLOT_COLORS[group_name]),
                aux={"grouped_plot_group": group_name},
            )

        parent.label = PLOT_LABELS[group_name]
        for member in members:
            parent.add_process(member)

        selected.append(plot_name)
        process_colors[plot_name] = PLOT_COLORS[group_name]

    groups["plot_backgrounds_grouped"] = selected

    # Set colours on both newly created and reused parents. Default plotting
    # settings also cover QCD created later by a histogram hook. Preserve other
    # settings such as scaling and stacking. Explicit --process-settings can
    # still override these defaults when deliberately requested.
    settings = dict(cfg.x("default_process_settings", {}))
    for process_name, hex_color in process_colors.items():
        rgb = _color_rgb(hex_color)
        if cfg.has_process(process_name, deep=True):
            process = cfg.get_process(process_name)
            process.color1 = rgb
            process.color2 = rgb
            process.color3 = rgb

        process_settings = dict(settings.get(process_name, {}))
        for attribute in ("color", "color1", "color2", "color3"):
            process_settings.pop(attribute, None)
        process_settings.update(color1=rgb, color2=rgb, color3=rgb)
        settings[process_name] = process_settings
    cfg.x.default_process_settings = settings

