_UNLEARNERS = {}


def register_unlearner(name, kind, hparams=(), summary="", reference="", cls=None, modality="vision"):
    """Register an algorithm. Use as decorator OR call directly:
        @register_unlearner("Foo", kind="trainer", hparams=[...])
        class Foo: ...
    OR:
        register_unlearner("Foo", kind="trainer", hparams=[...], cls=Foo)
    """
    assert kind in ("trainer", "nontrainer")
    assert modality in ("vision", "seq", "clip"), modality  # seq = text LLM and image+text LVLM; clip = dual encoder

    def _register(c):
        required = {"trainer": ("setup", "fit"), "nontrainer": ("fit",)}[kind]
        missing = [m for m in required if not callable(getattr(c, m, None))]
        if missing:
            raise TypeError(
                f"{c.__name__} registered as {kind} but missing methods: {missing}"
            )
        _UNLEARNERS[name.lower()] = {
            "name": name, "cls": c, "kind": kind,
            "hparams": list(hparams),
            "modality": modality,
            "summary": summary or (c.__doc__ or "").strip().split("\n")[0],
            "reference": reference,
        }
        return c

    if cls is not None:
        return _register(cls)
    return _register     # acts as a decorator


def _get(name):
    key = name.lower()
    if key not in _UNLEARNERS:
        raise KeyError(
            f"Unknown algorithm '{name}'. "
            f"Available: {list_algorithms()}"
        )
    return _UNLEARNERS[key]


def list_algorithms(kind=None, modality=None):
    return [m["name"] for m in _UNLEARNERS.values()
            if (kind is None or m["kind"] == kind) and (modality is None or m["modality"] == modality)]


def get_modality(name):
    return _get(name)["modality"]


def get_kind(name):
    return _get(name)["kind"]


def short_spec(name) -> str:
    """Short auto-print used by the pipeline on first construction."""
    m = _get(name)
    setup_hp = [h for h in m["hparams"] if h.category == "setup"]
    algo_hp  = [h for h in m["hparams"] if h.category == "hparam"]

    lines = [f"[torchunlearn] {m['name']}  [{m['kind']}, {m['modality']}]",
             f"  {m['summary']}"]
    if setup_hp:
        lines.append("  setup:")
        for h in setup_hp:
            tag = "required" if h.required else f"default={h.default!r}"
            lines.append(f"    - {h.name} ({h.type.__name__}, {tag})")
    if algo_hp:
        lines.append("  hparams:")
        for h in algo_hp:
            tag = "required" if h.required else f"default={h.default!r}"
            lines.append(f"    - {h.name} ({h.type.__name__}, {tag})")
    if not m["hparams"]:
        lines.append("  (no hyperparameters)")
    lines.append(f"  Run describe('{m['name']}') for full details.")
    return "\n".join(lines)


def describe(name) -> None:
    """Full human-readable spec printout."""
    m = _get(name)
    print(f"{m['name']}  [{m['kind']}, {m['modality']}]")
    print(f"  {m['summary']}")
    if m["reference"]:
        print(f"  reference: {m['reference']}")

    for category, label in [("setup", "setup kwargs"),
                            ("hparam", "hyperparameters")]:
        group = [h for h in m["hparams"] if h.category == category]
        if not group:
            continue
        print(f"  {label}:")
        for h in group:
            tag = "required" if h.required else f"default={h.default!r}"
            line = f"    - {h.name} ({h.type.__name__}, {tag})"
            if h.choices is not None:
                line += f"  choices={list(h.choices)}"
            if h.range is not None:
                line += f"  range={h.range}"
            print(line)
            if h.description:
                print(f"        {h.description}")
            if h.example is not None:
                print(f"        example: {h.example!r}")
    if not m["hparams"]:
        print("  (no hyperparameters)")


def build_unlearner(name, rmodel, hparams=None, **init_kwargs):
    """Instantiate a registered algorithm with validated hparams."""
    m = _get(name)
    hparams = dict(hparams or {})
    declared = {h.name: h for h in m["hparams"]}

    unknown = set(hparams) - set(declared)
    if unknown:
        raise ValueError(
            f"Unknown hparams for {m['name']}: {sorted(unknown)}. "
            f"Valid keys: {list(declared)}"
        )

    resolved = {h.name: h.validate(hparams.get(h.name))
                for h in m["hparams"]}

    instance = m["cls"](rmodel, **init_kwargs)
    instance._spec = m
    instance._resolved_hparams = resolved
    return instance

def build_unlearner_split(name, rmodel, hparams=None, **init_kwargs):
    """Like :func:`build_unlearner` but *constructs* the algorithm with its "hparam"-category values (they are
    ``__init__`` kwargs) and returns ``(instance, setup_kwargs)`` with the "setup"-category values.  None-valued
    hparams are dropped so the class defaults apply (e.g. RMU's trainable regex is derived from layer_id).
    Used by the sequence/multimodal pipeline and CLI."""
    m = _get(name)
    hparams = dict(hparams or {})
    declared = {h.name: h for h in m["hparams"]}
    unknown = set(hparams) - set(declared)
    if unknown:
        raise ValueError(f"Unknown hparams for {m['name']}: {sorted(unknown)}. Valid keys: {list(declared)}")
    resolved = {h.name: h.validate(hparams.get(h.name)) for h in m["hparams"]}
    algo_kw = {h.name: resolved[h.name] for h in m["hparams"] if h.category == "hparam" and resolved[h.name] is not None}
    setup_kw = {h.name: resolved[h.name] for h in m["hparams"] if h.category == "setup"}
    instance = m["cls"](rmodel, **algo_kw, **init_kwargs)
    instance._spec = m
    instance._resolved_hparams = resolved
    return instance, setup_kw
