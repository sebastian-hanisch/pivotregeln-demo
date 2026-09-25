"""Auswertung: Regelvergleich, Stillstand, Zyklen, Kennzahlen über feste Instanzen, Sweeps und Kurven über die Größe."""

import math
import random
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import piv_algorithm as A
import piv_constants as C
import piv_scenario as S

INF = math.inf


@dataclass(frozen=True)
class Settings:
    kind: str = "textbook"
    m: int = C.DEFAULT_M
    n: int = C.DEFAULT_N
    density: float = C.DEFAULT_DENSITY
    seed: int = C.DEFAULT_SEED
    rule: str = "dantzig"
    ratio: str = "index"


@lru_cache(maxsize=256)
def instance_of(settings):
    return S.generate(settings.kind, settings.m, settings.n, settings.density, settings.seed)


@dataclass
class Analysis:
    settings: Settings
    inst: object
    res: object                        # Lauf mit der gewählten Regel (mit Tableau-Schnappschüssen)
    vertices: list                     # zulässige Ecken (nur bei zwei Diensten)

    @property
    def status(self):
        return self.res.status

    @property
    def certificate(self):
        """Selbstprüfung ohne fremden Löser; nur bei Optimum."""
        if self.res.status != "optimal":
            return None
        b = np.array(self.inst.b)
        return {"primal": A.primal_violation(self.inst, self.res.x), "dual": A.dual_violation(self.inst, self.res.duals), "gap": abs(self.res.obj - float(np.dot(self.res.duals, b)))}


@lru_cache(maxsize=64)
def analyse(settings):
    inst = instance_of(settings)
    res = A.tableau_simplex(inst, rule=settings.rule, ratio=settings.ratio, keep=True)
    return Analysis(settings, inst, res, A.vertices_2d(inst) if inst.n == 2 else [])


@lru_cache(maxsize=64)
def compare_rules(settings):
    """Alle fünf Regeln auf derselben Instanz (mit dem gewählten Quotiententest): {Regel: Lauf mit Schnappschüssen}."""
    inst = instance_of(settings)
    return {rule: A.tableau_simplex(inst, rule=rule, ratio=settings.ratio, keep=True) for rule in A.RULES}


def _stats(values):
    values = [v for v in values if v is not None and v == v and v != INF]
    if not values:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(values)), float(np.percentile(values, 10)), float(np.percentile(values, 90))


def metrics(inst, rule="dantzig", ratio="index"):
    r = A.tableau_simplex(inst, rule=rule, ratio=ratio)
    p = r.total_pivots
    return {"status": r.status, "pivots": p, "flops": r.total_flops, "pivot_flops": r.flops, "price_flops": r.price_flops, "price_share": (r.price_flops / r.total_flops) if r.total_flops else 0.0,
            "degenerate_share": (r.degenerate_pivots / p) if p else 0.0, "tie_share": (sum(t.ties > 1 for t in r.pivots) / p) if p else 0.0, "longest_stall": max(r.stall_runs) if r.stall_runs else 0,
            "cycled": r.cycled, "obj": r.obj}


KEYS = ("pivots", "flops", "price_share", "degenerate_share", "tie_share", "longest_stall")


def run_config(base, seeds=C.SWEEP_SEEDS, **changes):
    """Kennzahlen der gewählten Regel über feste Instanzen (Median, 10./90. Perzentil); Läufe, die kreisen oder die Grenze erreichen, werden getrennt gezählt."""
    s0 = replace(base, **changes)
    rows = [metrics(instance_of(replace(s0, seed=seed)), s0.rule, s0.ratio) for seed in seeds]
    ok = [r for r in rows if r["status"] == "optimal"]
    out = {"n_runs": len(rows), "solved": len(ok), "cycled": sum(r["cycled"] for r in rows)}
    for key in KEYS:
        out[key], out[f"{key}_lo"], out[f"{key}_hi"] = _stats([r[key] for r in ok])
    return out


def rule_table(base, seeds=C.SWEEP_SEEDS):
    """Je Regel die Kennzahlen über die festen Instanzen und das Verhältnis zu Dantzig (Pivots, Operationen)."""
    out = {rule: run_config(base, seeds, rule=rule) for rule in A.RULES}
    for rule in A.RULES:
        out[rule]["pivots_vs_dantzig"] = out[rule]["pivots"] / out["dantzig"]["pivots"] if out["dantzig"]["pivots"] else float("nan")
        out[rule]["flops_vs_dantzig"] = out[rule]["flops"] / out["dantzig"]["flops"] if out["dantzig"]["flops"] else float("nan")
    return out


SWEEP_VALUES = {"size": C.SIZE_SWEEP, "density": C.DENSITY_OPTIONS, "kind": ("random", "mixed", "degenerate", "transport")}
SWEEP_LABELS = {"size": "Größe (m = n)", "density": "Dichte der Matrix", "kind": "Instanztyp"}


def _sized(base, value):
    if base.kind == "transport":
        k, l = C.TRANSPORT_SIZES[value]
        return replace(base, m=k, n=l)
    return replace(base, m=value, n=value)


def rule_curve(base, sizes=C.SIZE_SWEEP, seeds=C.FEAS_SEEDS, rules=A.RULES):
    """Pivots und Gesamtoperationen je Regel über die Größe (Median über `seeds` Instanzen, gleiche Instanzen für alle Regeln)."""
    out = {rule: [] for rule in rules}
    for k in sizes:
        s0 = _sized(base, k)
        insts = [instance_of(replace(s0, seed=s)) for s in seeds]
        for rule in rules:
            ms = [metrics(i, rule, base.ratio) for i in insts]
            ok = [x for x in ms if x["status"] == "optimal"]
            out[rule].append({"size": k, "m": s0.m, "n": s0.n, "pivots": _stats([x["pivots"] for x in ok])[0], "flops": _stats([x["flops"] for x in ok])[0],
                              "degenerate_share": _stats([x["degenerate_share"] for x in ok])[0], "solved": len(ok)})
    return out


def stall_table(seeds=C.FEAS_SEEDS, ratio="index"):
    """Anteil der Nullschritte an allen Pivots je Instanztyp und Regel (Summe über `seeds` Instanzen)."""
    configs = (("random", 20, 20), ("mixed", 20, 20), ("degenerate", 20, 20), ("transport", 4, 8), ("transport", 8, 12))
    rows = []
    for kind, m, n in configs:
        row = {"kind": kind, "m": m, "n": n}
        for rule in A.RULES:
            zero = pivots = 0
            for s in seeds:
                r = A.tableau_simplex(S.generate(kind, m, n, C.DEFAULT_DENSITY, s), rule=rule, ratio=ratio)
                zero += r.degenerate_pivots
                pivots += r.total_pivots
            row[rule] = zero / pivots if pivots else 0.0
        rows.append(row)
    return rows


def cycle_search(count=400, kinds=(("random", 10, 8), ("mixed", 10, 8), ("degenerate", 12, 9), ("transport", 3, 5), ("transport", 5, 8))):
    """Wie oft kreist Dantzig mit dem Indexrest im Quotiententest auf erzeugten Instanzen? Rückgabe {Typ: (kreisend, geprüft)}."""
    out = {}
    for kind, m, n in kinds:
        cycled = sum(A.tableau_simplex(S.generate(kind, m, n, C.DEFAULT_DENSITY, s)).cycled for s in range(count))
        out[(kind, m, n)] = (cycled, count)
    return out


def beale_like_instance(seed):
    """Kleines LP im Stil von Beale (3 Ressourcen, 4 Dienste, Koeffizienten aus Vierteln, Bestand meist 0), zufällig gezogen: ob es je kreist, ist offen."""
    rng = random.Random(f"beale-like-{seed}")
    m, n = 3, 4
    rows = [[rng.choice([-12, -8, -1, -0.5, 0, 0.25, 0.5, 1, 3, 9]) for _ in range(n)] for _ in range(m)]
    b = [rng.choice([0, 0, 0, 1]) for _ in range(m)]
    c = [rng.choice([-20, -6, 0.5, 0.75, 1, 3]) for _ in range(n)]
    return S.Instance(tuple(tuple(map(float, r)) for r in rows), tuple(map(float, b)), tuple(map(float, c)), (S.LE,) * m, tuple(f"x{j + 1}" for j in range(n)), tuple(f"r{i + 1}" for i in range(m)), "custom")


def beale_like_search(count=5000):
    """Wie viele der zufälligen Beale-artigen LPs kreisen mit Dantzig und Indexrest? Rückgabe (kreisend, geprüft, davon mit Optimum)."""
    cycled = optimal = 0
    for s in range(count):
        r = A.tableau_simplex(beale_like_instance(s))
        cycled += r.cycled
        optimal += r.status == "optimal"
    return cycled, count, optimal


def sweep(param, base, values=None):
    values = SWEEP_VALUES[param] if values is None else values
    if param == "size":
        return [{"value": v, **{rule: run_config(_sized(base, v), rule=rule) for rule in A.RULES}} for v in values]
    return [{"value": v, **{rule: run_config(base, rule=rule, **{param: v}) for rule in A.RULES}} for v in values]
