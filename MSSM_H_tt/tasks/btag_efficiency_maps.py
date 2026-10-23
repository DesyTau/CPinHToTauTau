# coding: utf-8
from __future__ import annotations

import json
import law
import luigi

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    CalibratorClassesMixin,
    SelectorClassMixin,
    ReducerClassMixin,
    ProducersMixin,
    DatasetsProcessesMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.union import UniteColumns
from columnflow.util import dev_sandbox, DotDict, maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")


def _pretty_json_pydantic(model) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(exclude_unset=True, indent=2) + "\n"
    try:
        return model.json(exclude_unset=True, indent=2) + "\n"
    except TypeError:
        payload = model.dict(exclude_unset=True)
        return json.dumps(payload, indent=2) + "\n"


def _iter_event_targets(unite_out):
    coll = unite_out.get("collection", unite_out)
    if hasattr(coll, "targets"):
        for v in coll.targets.values():
            yield v.get("events", v) if isinstance(v, dict) else v
    elif isinstance(coll, (list, tuple)):
        for v in coll:
            yield v.get("events", v) if isinstance(v, dict) else v
    else:
        yield coll.get("events", coll) if isinstance(coll, dict) else coll


def _iter_file_targets(obj):
    """
    Yield law file targets inside arbitrary law/columnflow containers (DotDict, TargetCollection, etc.).
    """
    if obj is None:
        return

    # direct file target
    if hasattr(obj, "localize") and hasattr(obj, "path"):
        yield obj
        return

    # law TargetCollection
    if hasattr(obj, "targets"):
        for v in obj.targets.values():
            yield from _iter_file_targets(v)
        return

    # dict / DotDict
    if isinstance(obj, dict):
        if "collection" in obj:
            yield from _iter_file_targets(obj["collection"])
        for v in obj.values():
            yield from _iter_file_targets(v)
        return

    # list/tuple
    if isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_file_targets(v)
        return


class _BTagEffCfgMixin:
    """
    Shared helpers to resolve tagger/wp/discriminator/wp_threshold from cfg.x and optional CLI overrides.
    Implements modify_param_values so resolved discriminator/wp_threshold appear in Luigi logs.
    """

    discriminator = luigi.Parameter(default="")
    wp_threshold = luigi.FloatParameter(default=-1.0)
    event_weight_column = luigi.Parameter(default="")

    @staticmethod
    def _norm_tagger(tagger: str) -> str:
        t = (tagger or "").strip()
        tl = t.lower().replace(" ", "").replace("-", "").replace("_", "")
        if tl in ("deepjet", "dj"):
            return "deepjet"
        if tl in ("particlenet", "pnet"):
            return "particleNet"
        if tl in ("robustparticletransformer", "rpt", "robustpt"):
            return "robustParticleTransformer"
        return t

    @classmethod
    def modify_param_values(cls, params: dict) -> dict:
        params = super().modify_param_values(params)

        cfg = params.get("config_inst", None)
        if cfg is None:
            analysis_inst = params.get("analysis_inst", None)
            if analysis_inst is None:
                analysis = params.get("analysis", None)
                if isinstance(analysis, str) and analysis:
                    analysis_inst = cls.get_analysis_inst(analysis)
            config_name = params.get("config", None)
            if analysis_inst is not None and config_name:
                try:
                    cfg = analysis_inst.get_config(config_name)
                except Exception:
                    cfg = None

        if cfg is None:
            return params

        sel = {}
        if cfg.has_aux("btag_eff_maps"):
            try:
                sel = dict(cfg.x.btag_eff_maps)
            except Exception:
                sel = cfg.x.btag_eff_maps

        tagger = cls._norm_tagger(str(sel.get("tagger", "deepjet")))
        wp_name = str(sel.get("wp", "medium")).strip()

        # discriminator
        if not str(params.get("discriminator", "")).strip():
            if "discriminator" in sel and str(sel["discriminator"]).strip():
                params["discriminator"] = str(sel["discriminator"]).strip()
            else:
                if tagger == "deepjet" and cfg.has_aux("btag_sf_deepjet"):
                    params["discriminator"] = str(cfg.x.btag_sf_deepjet.discriminator)
                elif tagger == "particleNet" and cfg.has_aux("btag_sf_pnet"):
                    params["discriminator"] = str(cfg.x.btag_sf_pnet.discriminator)
                elif tagger == "robustParticleTransformer" and cfg.has_aux("btag_sf_rpt"):
                    params["discriminator"] = str(cfg.x.btag_sf_rpt.discriminator)

        # wp_threshold
        try:
            wp_cli = float(params.get("wp_threshold", -1.0))
        except Exception:
            wp_cli = -1.0

        if wp_cli < 0:
            if "wp_threshold" in sel:
                params["wp_threshold"] = float(sel["wp_threshold"])
            else:
                try:
                    year = int(cfg.campaign.x.year)
                    tag = str(cfg.campaign.x.tag)
                    params["wp_threshold"] = float(cfg.x.btag_working_points[year][tag][tagger][wp_name])
                except Exception:
                    params["wp_threshold"] = -1.0

        # event weight column
        if not str(params.get("event_weight_column", "")).strip():
            if "event_weight_column" in sel and str(sel["event_weight_column"]).strip():
                params["event_weight_column"] = str(sel["event_weight_column"]).strip()

        return params

    def _resolve_effmap_cfg(self, cfg) -> dict:
        sel = {}
        if cfg.has_aux("btag_eff_maps"):
            try:
                sel = dict(cfg.x.btag_eff_maps)
            except Exception:
                sel = cfg.x.btag_eff_maps

        tagger = self._norm_tagger(str(sel.get("tagger", "deepjet")))
        wp_name = str(sel.get("wp", "medium")).strip()

        disc_cli = str(self.discriminator).strip()
        if disc_cli:
            disc = disc_cli
        elif "discriminator" in sel and str(sel["discriminator"]).strip():
            disc = str(sel["discriminator"]).strip()
        else:
            if tagger == "deepjet":
                disc = str(cfg.x.btag_sf_deepjet.discriminator)
            elif tagger == "particleNet":
                disc = str(cfg.x.btag_sf_pnet.discriminator)
            elif tagger == "robustParticleTransformer":
                if cfg.has_aux("btag_sf_rpt"):
                    disc = str(cfg.x.btag_sf_rpt.discriminator)
                else:
                    raise ValueError(
                        f"{cfg.name}: robustParticleTransformer selected but discriminator cannot be inferred. "
                        "Set cfg.x.btag_eff_maps['discriminator'] explicitly (or define cfg.x.btag_sf_rpt)."
                    )
            else:
                raise ValueError(
                    f"{cfg.name}: cannot infer discriminator for tagger='{tagger}'. "
                    "Set cfg.x.btag_eff_maps['discriminator'] explicitly."
                )

        wp_cli = float(self.wp_threshold)
        if wp_cli >= 0:
            wp_thr = wp_cli
        elif "wp_threshold" in sel:
            wp_thr = float(sel["wp_threshold"])
        else:
            year = int(cfg.campaign.x.year)
            tag = str(cfg.campaign.x.tag)
            wp_thr = float(cfg.x.btag_working_points[year][tag][tagger][wp_name])

        wcol_cli = str(self.event_weight_column).strip()
        if wcol_cli:
            wcol = wcol_cli
        else:
            wcol = str(sel.get("event_weight_column", "")).strip()

        return {"tagger": tagger, "wp": wp_name, "discriminator": disc, "wp_threshold": wp_thr, "wcol": wcol}


class BTagEfficiencyCounts(
    _BTagEffCfgMixin,
    CalibratorClassesMixin,
    SelectorClassMixin,
    ReducerClassMixin,
    ProducersMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    task_namespace = "cf"
    single_config = True
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    dataset = luigi.Parameter(description="Dataset name")

    pt_bins = law.CSVParameter(default=(20, 30, 50, 70, 100, 140, 200, 300, 600, 1000))
    eta_bins = law.CSVParameter(default=(0.0, 1.5, 2.5))
    jet_pt_min = luigi.FloatParameter(default=20.0)
    jet_abseta_max = luigi.FloatParameter(default=2.5)

    resolution_task_cls = UniteColumns
    reqs = Requirements(RemoteWorkflow.reqs, UniteColumns=UniteColumns)

    @classmethod
    def req_params(cls, inst, **kwargs) -> dict:
        _prefer_cli = law.util.make_set(kwargs.get("_prefer_cli", [])) | {"producers"}
        kwargs["_prefer_cli"] = _prefer_cli
        return super().req_params(inst, **kwargs)

    def create_branch_map(self):
        return [DotDict()]

    def requires(self):
        return self.reqs.UniteColumns.req(
            self,
            dataset=self.dataset,
            shift="nominal",
            branch=-1,
            _exclude={"branches"},
            _prefer_cli={"producers"},
        )

    def output(self):
        return self.target("btag_eff_counts.npz")

    @law.decorator.log
    def run(self):
        cfg = self.config_inst
        sel = self._resolve_effmap_cfg(cfg)

        disc = sel["discriminator"]
        wp = float(sel["wp_threshold"])
        wcol = sel["wcol"]

        pt_edges = [float(x) for x in self.pt_bins]
        eta_edges = [float(x) for x in self.eta_bins]
        shape = (len(pt_edges) - 1, len(eta_edges) - 1)

        den_b = np.zeros(shape, dtype=np.float64)
        num_b = np.zeros(shape, dtype=np.float64)
        den_c = np.zeros(shape, dtype=np.float64)
        num_c = np.zeros(shape, dtype=np.float64)
        den_l = np.zeros(shape, dtype=np.float64)
        num_l = np.zeros(shape, dtype=np.float64)

        unite_out = self.input()

        for t in _iter_event_targets(unite_out):
            with t.localize("r") as lt:
                cols = ["Jet.pt", "Jet.eta", "Jet.hadronFlavour", f"Jet.{disc}"]
                if wcol:
                    cols.append(wcol)
                events = ak.from_parquet(lt.path, columns=cols)

            if len(events) == 0:
                continue

            jets = events.Jet
            if disc not in jets.fields:
                raise RuntimeError(
                    f"Jet.{disc} not found in input parquet for dataset {self.dataset}. "
                    f"Available Jet fields: {sorted(jets.fields)}."
                )

            pt = jets.pt
            abseta = abs(jets.eta)
            flav = jets.hadronFlavour
            score = getattr(jets, disc)

            if wcol and hasattr(events, wcol):
                evt_w = getattr(events, wcol)
            else:
                evt_w = np.ones(len(events), dtype=np.float64)
            jet_w = ak.broadcast_arrays(evt_w, pt)[0]

            base = (pt >= self.jet_pt_min) & (abseta <= self.jet_abseta_max)
            tagged = base & (score > wp)

            is_b = base & (flav == 5)
            is_c = base & (flav == 4)
            is_l = base & (flav != 5) & (flav != 4)

            def fill(den, num, sel_mask, sel_tag_mask):
                pt1 = ak.to_numpy(ak.flatten(pt[sel_mask], axis=None))
                e1 = ak.to_numpy(ak.flatten(abseta[sel_mask], axis=None))
                w1 = ak.to_numpy(ak.flatten(jet_w[sel_mask], axis=None))
                if pt1.size:
                    h, _, _ = np.histogram2d(pt1, e1, bins=[pt_edges, eta_edges], weights=w1)
                    den += h

                pt2 = ak.to_numpy(ak.flatten(pt[sel_tag_mask], axis=None))
                e2 = ak.to_numpy(ak.flatten(abseta[sel_tag_mask], axis=None))
                w2 = ak.to_numpy(ak.flatten(jet_w[sel_tag_mask], axis=None))
                if pt2.size:
                    h, _, _ = np.histogram2d(pt2, e2, bins=[pt_edges, eta_edges], weights=w2)
                    num += h

            fill(den_b, num_b, is_b, tagged & (flav == 5))
            fill(den_c, num_c, is_c, tagged & (flav == 4))
            fill(den_l, num_l, is_l, tagged & is_l)

        out = self.output()
        with out.localize("w") as lt:
            np.savez_compressed(
                lt.path,
                pt_edges=np.array(pt_edges, dtype=np.float64),
                eta_edges=np.array(eta_edges, dtype=np.float64),
                den_b=den_b, num_b=num_b,
                den_c=den_c, num_c=num_c,
                den_l=den_l, num_l=num_l,
            )


class CreateBTagEfficiencyMaps(
    _BTagEffCfgMixin,
    CalibratorClassesMixin,
    SelectorClassMixin,
    ReducerClassMixin,
    ProducersMixin,
    DatasetsProcessesMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    task_namespace = "cf"
    single_config = True
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    resolution_task_cls = UniteColumns

    pt_bins = law.CSVParameter(default=(20, 30, 50, 70, 100, 140, 200, 300, 600, 1000))
    eta_bins = law.CSVParameter(default=(0.0, 1.5, 2.5))
    jet_pt_min = luigi.FloatParameter(default=20.0)
    jet_abseta_max = luigi.FloatParameter(default=2.5)

    reqs = Requirements(RemoteWorkflow.reqs, BTagEfficiencyCounts=BTagEfficiencyCounts)

    exclude_index = False

    def create_branch_map(self):
        return [DotDict()]

    @classmethod
    def req_params(cls, inst, **kwargs) -> dict:
        _prefer_cli = law.util.make_set(kwargs.get("_prefer_cli", [])) | {"producers"}
        kwargs["_prefer_cli"] = _prefer_cli
        return super().req_params(inst, **kwargs)

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        if "dataset" in parts:
            parts.pop("dataset")
        parts.insert_before("version", "datasets", f"datasets_{self.datasets_repr}")
        return parts

    def requires(self):
        req = {}
        for d in self.datasets:
            if d not in self.config_inst.datasets.names():
                continue
            req[d] = self.reqs.BTagEfficiencyCounts.req(
                self,
                dataset=d,
                branch=-1,
                _exclude={"branches"},
                _prefer_cli={"producers"},
            )
        return req

    def output(self):
        cfg = self.config_inst
        year = getattr(cfg.x, "year", cfg.campaign.aux.get("year", "NA"))
        tag = getattr(cfg.x, "tag", cfg.campaign.aux.get("tag", "NA"))
        channel = cfg.channels.get_first().name if len(cfg.channels) else "NA"

        disc = str(self.discriminator).strip() or "UNKNOWN_DISC"
        wp = float(self.wp_threshold)

        safe_disc = disc.replace("/", "_").replace(".", "_")
        wp_str = f"{wp:g}".replace(".", "p")  # 0.245 -> 0p245

        base = f"btag_eff_maps_{channel}_{year}_{tag}"
        suffix = f"{safe_disc}_wp{wp_str}"

        return {
            "json": self.target(f"{base}.json"),
            "plots": {
                "eff_b": self.target(f"{base}_eff_b_{suffix}.png"),
                "eff_c": self.target(f"{base}_eff_c_{suffix}.png"),
                "eff_light": self.target(f"{base}_eff_light_{suffix}.png"),
            },
        }

    @law.decorator.log
    def run(self):
        import correctionlib.schemav2 as cs
        import matplotlib.pyplot as plt

        inputs = self.input()

        first_obj = next(iter(inputs.values()))
        first_tgt = next(_iter_file_targets(first_obj))
        with first_tgt.localize("r") as lt:
            z0 = np.load(lt.path)
            pt_edges = z0["pt_edges"].tolist()
            eta_edges = z0["eta_edges"].tolist()
            shape = (len(pt_edges) - 1, len(eta_edges) - 1)

        den_b = np.zeros(shape, dtype=np.float64)
        num_b = np.zeros(shape, dtype=np.float64)
        den_c = np.zeros(shape, dtype=np.float64)
        num_c = np.zeros(shape, dtype=np.float64)
        den_l = np.zeros(shape, dtype=np.float64)
        num_l = np.zeros(shape, dtype=np.float64)

        for _, obj in inputs.items():
            for tgt in _iter_file_targets(obj):
                with tgt.localize("r") as lt:
                    z = np.load(lt.path)
                    den_b += z["den_b"]; num_b += z["num_b"]
                    den_c += z["den_c"]; num_c += z["num_c"]
                    den_l += z["den_l"]; num_l += z["num_l"]

        def eff(num, den):
            out = np.zeros_like(den, dtype=np.float64)
            m = den > 0
            out[m] = num[m] / den[m]
            return np.clip(out, 0.0, 1.0)

        eff_b = eff(num_b, den_b)
        eff_c = eff(num_c, den_c)
        eff_l = eff(num_l, den_l)

        def make_corr(name, arr):
            content = arr.reshape(-1).astype(float).tolist()
            return cs.Correction(
                name=name,
                description="btag efficiency in simulation (WP-based)",
                version=1,
                inputs=[
                    cs.Variable(name="pt", type="real", description="jet pt"),
                    cs.Variable(name="abseta", type="real", description="jet |eta|"),
                ],
                output=cs.Variable(name="eff", type="real", description="efficiency"),
                data=cs.MultiBinning(
                    nodetype="multibinning",
                    inputs=["pt", "abseta"],
                    edges=[pt_edges, eta_edges],
                    content=content,
                    flow="clamp",
                ),
            )

        cset = cs.CorrectionSet(
            schema_version=2,
            description="WP btag efficiencies (summed over datasets)",
            corrections=[
                make_corr("eff_b", eff_b),
                make_corr("eff_c", eff_c),
                make_corr("eff_light", eff_l),
            ],
        )

        self.output()["json"].dump(_pretty_json_pydantic(cset), formatter="text")

        def plot_eff(arr, title, out_target):
            fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
            pcm = ax.pcolormesh(eta_edges, pt_edges, arr, shading="auto")
            ax.set_xlabel(r"$|\eta|$")
            ax.set_ylabel(r"$p_T$ [GeV]")
            ax.set_title(title)
            fig.colorbar(pcm, ax=ax)
            out_target.dump(fig, formatter="mpl")
            plt.close(fig)

        cfg = self.config_inst
        disc = str(self.discriminator).strip()
        wp = float(self.wp_threshold)
        hdr = f"{cfg.name}: {disc} > {wp:g}"

        plot_eff(eff_b, f"{hdr} | eff b", self.output()["plots"]["eff_b"])
        plot_eff(eff_c, f"{hdr} | eff c", self.output()["plots"]["eff_c"])
        plot_eff(eff_l, f"{hdr} | eff light", self.output()["plots"]["eff_light"])

# -----------------------------------------------------------------------------
# Make --cf.Bundle*-workflow / --cf.Bundle*-effective-workflow CLI args recognized
# without relying on specific import locations of Bundle* tasks.
# -----------------------------------------------------------------------------
def _patch_bundle_workflow_params() -> None:

    from luigi.task_register import Register

    # task families we want to accept workflow args for
    families = (
        "cf.BundleRepo",
        "cf.BundleSoftware",
        "cf.BundleExternalFiles",
        "cf.BundleBashSandbox",
        "cf.BundleCMSSWSandbox",
    )

    for fam in families:
        try:
            task_cls = Register.get_task_cls(fam)
        except Exception:
            # task family not registered in this environment/version
            continue

        # Add parameters only if missing. Mark non-significant so they do not alter output paths/hashes.
        if not hasattr(task_cls, "workflow"):
            task_cls.workflow = luigi.Parameter(default="local", significant=False)
        if not hasattr(task_cls, "effective_workflow"):
            task_cls.effective_workflow = luigi.Parameter(default="local", significant=False)

_patch_bundle_workflow_params()