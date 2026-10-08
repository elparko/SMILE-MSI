"""The analysis setup — the method decisions fixed *before* an analysis starts.

A reviewer's first questions are about choices made before any result was seen: which mass
tolerance, which normalization, what counts as a replicate, what significance threshold. The
chat asks for them up front (a setup card before the first message, then a sidebar), logs
them as the conversation's opening record, and **enforces** them:

* slide-level settings (extraction tolerance, normalization, window reducer, TIC cap) are
  applied to the open slide, however it was opened;
* tool parameters the setup covers are filled in when the model leaves them out
  (:func:`fill_defaults`), and a value the model sets *differently* is logged as a
  deviation and flagged in the chat — allowed, never silent.

The method knobs come from :mod:`smile_msi.profiles` — the same named, versioned Analysis
Profiles the desktop app uses — so a lab's profile drives the chat too, and "Save as
profile" here writes a profile the app can load. Three study-level fields are chat-only:
the question being asked, the unit of replication, and the false-discovery threshold.
"""
from __future__ import annotations

import copy

from .. import profiles

#: Profile keys the chat enforces, in display order. Everything else in a profile still
#: travels with it (and into the log) but the chat doesn't act on it.
ENFORCED = ("mode", "ppm", "id_ppm", "norm", "reduce", "tic_max_amp", "snr", "prominence",
            "max_peaks", "segment_default_count", "spatial_min_morans",
            "spatial_max_candidates")
#: Shown open on the setup card; the rest sit under "More settings".
ESSENTIAL = ("mode", "ppm", "id_ppm", "norm", "snr")

#: Study-level decisions the chat records (not part of an Analysis Profile).
STUDY_FIELDS = (
    {"key": "question", "label": "What question are you asking?", "group": "Study",
     "kind": "text", "default": "",
     "help": "One sentence. Written into the log, so the analysis can be read against it."},
    {"key": "replicate", "label": "Unit of replication", "group": "Study", "kind": "choice",
     "default": "pixels (exploratory)",
     "choices": ["pixels (exploratory)", "regions / sections", "animals / patients"],
     "help": "What one independent observation is. With pixels, p-values describe pixels, "
             "not biology — results are exploratory."},
    {"key": "max_q", "label": "False-discovery threshold (q)", "group": "Study",
     "kind": "float", "default": 0.05, "lo": 0.001, "hi": 0.25,
     "help": "Benjamini–Hochberg q a feature must pass to be called significant."},
    {"key": "min_auc", "label": "Minimum effect (AUC)", "group": "Study", "kind": "float",
     "default": 0.7, "lo": 0.5, "hi": 1.0,
     "help": "Smallest separation worth reporting (0.5 = none, 1 = perfect)."},
)
_STUDY = {f["key"]: f for f in STUDY_FIELDS}

#: tool → {argument: setup key} filled in when the model leaves the argument out.
TOOL_ARGS = {
    "find_peaks": {"snr": "snr", "max_peaks": "max_peaks",
                   "min_morans": "spatial_min_morans",
                   "max_candidates": "spatial_max_candidates"},
    "annotate": {"mode": "mode", "match_ppm": "id_ppm"},
}
#: registry step params (inside run_analysis ``params``) the setup covers.
STEP_PARAMS = {"tol_ppm": "ppm", "norm": "norm", "max_q": "max_q", "min_auc": "min_auc",
               "n_clusters": None}          # None: covered elsewhere, never forced


def form() -> dict:
    """Everything the setup card needs: fields (profile schema + study fields), the saved
    profiles to start from, and which fields are essential."""
    fields = []
    for p in profiles.SCHEMA:
        if p.key not in ENFORCED:
            continue
        fields.append({"key": p.key, "label": p.label, "group": p.group, "kind": p.kind,
                       "default": p.default, "choices": list(p.choices), "help": p.help,
                       "lo": p.lo, "hi": p.hi, "essential": p.key in ESSENTIAL})
    fields.sort(key=lambda f: ENFORCED.index(f["key"]))
    study = [dict(f, essential=True) for f in STUDY_FIELDS]
    return {"fields": study + fields,
            "profiles": [{"name": pr["name"], "version": pr.get("version", 1),
                          "label": profiles.label(pr),
                          "params": {k: pr["params"].get(k) for k in ENFORCED}}
                         for pr in profiles.list_profiles()]}


def _coerce_study(key, value):
    f = _STUDY[key]
    if f["kind"] == "float":
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{f['label']} must be a number, got {value!r}") from None
        if not (f["lo"] <= v <= f["hi"]):
            raise ValueError(f"{f['label']} must be between {f['lo']} and {f['hi']}")
        return v
    if f["kind"] == "choice":
        if value not in f["choices"]:
            raise ValueError(f"{f['label']} must be one of {f['choices']}")
        return value
    return str(value).strip()[:500]


def resolve(values: dict | None = None, profile: str = "") -> dict:
    """A complete, validated setup: the named profile's parameters (the built-in default
    when ``profile`` is empty), overridden by ``values``, plus the study fields.
    Raises ValueError on an out-of-range or unknown value."""
    values = dict(values or {})
    base = profiles.load(profile) if profile else profiles.builtin()
    params = dict(base["params"])
    by_key = {p.key: p for p in profiles.SCHEMA}
    for k, v in values.items():
        if k in by_key and k in ENFORCED:
            p = by_key[k]
            if p.kind in ("float", "int"):          # validate first: _coerce would clamp
                try:
                    num = float(v)
                except (TypeError, ValueError):
                    raise ValueError(f"{p.label} must be a number, got {v!r}") from None
                if (p.lo is not None and num < p.lo) or (p.hi is not None and num > p.hi):
                    raise ValueError(f"{p.label} must be between {p.lo} and {p.hi}")
            elif p.kind == "choice" and p.choices and str(v) not in p.choices:
                raise ValueError(f"{p.label} must be one of {list(p.choices)}")
            params[k] = profiles._coerce(p, v)
        elif k not in _STUDY and k not in ("profile",):
            raise ValueError(f"unknown setting {k!r}")
    study = {f["key"]: _coerce_study(f["key"], values.get(f["key"], f["default"]))
             for f in STUDY_FIELDS}
    changed = sorted(k for k in ENFORCED if params.get(k) != base["params"].get(k))
    return {"profile": {"name": base["name"], "version": base.get("version", 1),
                        "hash": profiles.content_hash(base)},
            "params": {k: params[k] for k in ENFORCED}, "all_params": params,
            "study": study, "changed_from_profile": changed}


def save_as_profile(setup: dict, name: str) -> dict:
    """Write the setup's method parameters as a new Analysis Profile (the desktop app can
    load it too). An existing name gets a new version."""
    existing = profiles.versions_of(name)
    pr = profiles.make_profile(name, setup["all_params"],
                               version=(max(existing) + 1 if existing else 1))
    profiles.save(pr)
    return {"name": pr["name"], "version": pr["version"]}


# --------------------------------------------------------------------------- #
# enforcement
# --------------------------------------------------------------------------- #
def apply_to_slide(setup: dict, slide) -> dict:
    """Set the slide's extraction settings to the setup's; returns ``{key: (was, now)}`` for
    everything that changed (logged, so a reopened session's settings can't silently win)."""
    p = setup["params"]
    api, ds = slide.api, slide.ds
    changes = {}
    for attr, key, cast in (("ppm", "ppm", float), ("norm", "norm", str),
                            ("reduce", "reduce", str)):
        was = getattr(api, attr)
        if was != cast(p[key]):
            setattr(api, attr, cast(p[key]))
            changes[attr] = (was, cast(p[key]))
    was = float(getattr(ds, "tic_max_amp", 3.0))       # the engine's default cap is 3×
    if was != float(p["tic_max_amp"]):
        ds.tic_max_amp = float(p["tic_max_amp"])
        if hasattr(ds, "_norm_cache"):
            ds._norm_cache.clear()                  # factors depend on the cap
        changes["tic_max_amp"] = (was, float(p["tic_max_amp"]))
    return changes


def _value(setup, key):
    return setup["study"][key] if key in setup["study"] else setup["params"][key]


def fill_defaults(setup: dict, tool: str, args: dict) -> tuple[dict, list]:
    """``(effective args, deviations)`` for one tool call: arguments the setup covers are
    filled in when missing; ones the model set to something else are kept and reported as
    ``[{"arg", "setup", "used"}]``."""
    args = copy.deepcopy(args)
    deviations = []

    def check(container, arg, key, label):
        want = _value(setup, key)
        if arg not in container or container[arg] in (None, ""):
            container[arg] = want
        elif str(container[arg]) != str(want):
            try:
                same = float(container[arg]) == float(want)
            except (TypeError, ValueError):
                same = False
            if not same:
                deviations.append({"arg": label, "setup": want, "used": container[arg]})

    for arg, key in TOOL_ARGS.get(tool, {}).items():
        if tool == "find_peaks" and arg in ("min_morans", "max_candidates") \
                and not args.get("spatial"):
            continue                              # only the spatial finder uses these
        check(args, arg, key, arg)
    if tool == "run_analysis":
        from .. import registry

        step = str(args.get("step_id", ""))
        defaults = registry.default_params(step) if step in registry.REGISTRY else {}
        params = args.get("params") if isinstance(args.get("params"), dict) else {}
        for arg, key in STEP_PARAMS.items():
            if key and arg in defaults:
                check(params, arg, key, f"params.{arg}")
        if params:
            args["params"] = params
    return args, deviations


def brief(setup: dict) -> str:
    """The setup as the model reads it at the start of a conversation."""
    p, s = setup["params"], setup["study"]
    lines = [f"[Analysis setup locked by the scientist — profile "
             f"\"{setup['profile']['name']}\" v{setup['profile']['version']}"
             + (f", changed: {', '.join(setup['changed_from_profile'])}"
                if setup["changed_from_profile"] else "") + "]"]
    if s["question"]:
        lines.append(f"Question: {s['question']}")
    lines.append(f"Replication unit: {s['replicate']}. Significance: q ≤ {s['max_q']}, "
                 f"report effects with AUC ≥ {s['min_auc']}.")
    lines.append("Method: " + "; ".join(f"{k}={p[k]}" for k in ENFORCED))
    lines.append("These are applied to the slide and filled into tool calls automatically. "
                 "If you need a different value, say why — it is logged as a deviation.")
    return "\n".join(lines)


def diff_brief(old: dict, new: dict) -> str:
    """What changed when the scientist edits the setup mid-conversation."""
    ch = []
    for k in ENFORCED:
        if old["params"][k] != new["params"][k]:
            ch.append(f"{k}: {old['params'][k]} → {new['params'][k]}")
    for k in _STUDY:
        if old["study"][k] != new["study"][k]:
            ch.append(f"{k}: {old['study'][k]!r} → {new['study'][k]!r}")
    return ("[The scientist changed the analysis setup: " + "; ".join(ch) +
            ". Results computed before this used the old values.]") if ch else ""
