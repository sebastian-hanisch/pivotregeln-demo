"""Simplex mit dichtem Tableau, Zwei-Phasen-Start und wählbarer Pivotregel (nur numpy). Maximiert c·x unter A x (<=|>=|=) b, x >= 0.

Das Tableau ist dicht: m Zeilen der Bedingungen plus eine Zielzeile mit den reduzierten Kosten r_j = z_j - c_j (Optimum, wenn alle r_j >= 0). Ein Pivot tauscht eine Variable in der Basis.
Welche Spalte eintritt (Pivotregel) und welche Zeile bei einem Gleichstand im Quotiententest austritt, ist die Frage dieses Stücks. Aufwand wird in Gleitkomma-Operationen des dichten Modells gezählt:
ein Pivot kostet (Spalten + 1) + 2 · m · (Spalten + 1); die Preisgebung der Regel (Auswahl der Spalte) und der lexikographische Vergleich kommen als eigener Posten dazu.
"""

import random
from dataclasses import dataclass, field

import numpy as np

from piv_scenario import EQ, GE, LE

TOL = 1e-9
MAX_SNAPSHOTS = 80
RULES = ("dantzig", "greatest", "steepest", "bland", "random")
RATIO_RULES = ("index", "lexicographic")


@dataclass
class Pivot:
    k: int                       # laufende Nummer (ab 1)
    phase: int
    enter: int                   # eintretende Spalte
    leave_row: int
    leave_var: int               # Spalte der austretenden Basisvariable
    ratio: float                 # Schrittweite (kleinster Quotient)
    ties: int                    # Zahl der Zeilen mit demselben kleinsten Quotienten (> 1: Gleichstand, die Ecke wird entartet)
    obj: float                   # Zielwert der aktuellen Phase nach dem Pivot (Phase 1: -Summe der künstlichen Variablen)
    degenerate: bool             # Schrittweite null: die Ecke ändert sich nicht
    basis: tuple                 # Basisspalten nach dem Pivot, in Zeilenreihenfolge
    x: tuple                     # Entscheidungsvariablen der Basislösung nach dem Pivot
    feasible: bool               # alle künstlichen Variablen null (Phase 2: immer)
    zero_basics: int             # Zahl der Basisvariablen mit Wert null (entartete Ecke)


@dataclass
class SimplexResult:
    status: str                  # "optimal" | "infeasible" | "unbounded" | "cycled" (eine Basis wiederholt sich) | "limit" (Pivot-Grenze)
    x: tuple = ()
    obj: float = float("nan")
    duals: tuple = ()            # Schattenpreise je Bedingung in der ursprünglichen Orientierung
    pivots: list = field(default_factory=list)
    phase1_pivots: int = 0
    flops: int = 0               # Operationen der Pivots
    col_names: tuple = ()
    snapshots: list = field(default_factory=list)   # {"phase", "T", "basis", "pivot"} vor dem ersten und nach jedem Pivot (nur mit keep=True)
    redundant_rows: int = 0
    m: int = 0
    n_cols: int = 0
    rule: str = "dantzig"
    ratio: str = "index"
    price_flops: int = 0         # Aufwand der Preisgebung bzw. des lexikographischen Vergleichs, zusätzlich zu den Pivot-Operationen
    cycled: bool = False
    cycle_len: int = 0           # Pivots zwischen zwei Vorkommen derselben Basis (bei cycled)
    repeats: int = 0             # bei der Zufallsregel: wie oft eine Basis erneut besucht wurde (kein Beweis für einen Zyklus)
    stall_runs: list = field(default_factory=list)   # Längen aufeinanderfolgender Nullschritte

    @property
    def total_pivots(self):
        return len(self.pivots)

    @property
    def degenerate_pivots(self):
        return sum(p.degenerate for p in self.pivots)

    @property
    def flops_per_pivot(self):
        return self.flops / len(self.pivots) if self.pivots else 0.0

    @property
    def total_flops(self):
        return self.flops + self.price_flops


def standard_form(inst):
    """Gleichungsform mit Schlupf-, Überschuss- und künstlichen Variablen; Zeilen mit negativem b werden mit -1 multipliziert (Sinn dreht sich).
    Gibt (T, basis, info) mit T = [A' | rhs] (m Zeilen, ohne Zielzeile)."""
    A, b, c = inst.arrays()
    m, n = A.shape
    rows, rhs, senses, sign = [], [], [], []
    for i in range(m):
        a, r, s = A[i].copy(), float(b[i]), inst.senses[i]
        sg = 1
        if r < 0:
            a, r, sg = -a, -r, -1
            s = {LE: GE, GE: LE, EQ: EQ}[s]
        rows.append(a), rhs.append(r), senses.append(s), sign.append(sg)
    slack_col, art_col = {}, {}
    ncols = n
    for i, s in enumerate(senses):
        if s in (LE, GE):
            slack_col[i] = ncols
            ncols += 1
    for i, s in enumerate(senses):
        if s in (GE, EQ):
            art_col[i] = ncols
            ncols += 1
    T = np.zeros((m, ncols + 1))
    basis = []
    for i in range(m):
        T[i, :n] = rows[i]
        T[i, -1] = rhs[i]
        if i in slack_col:
            T[i, slack_col[i]] = 1.0 if senses[i] == LE else -1.0
        if i in art_col:
            T[i, art_col[i]] = 1.0
        basis.append(art_col[i] if i in art_col else slack_col[i])
    names = list(inst.names) + [None] * (ncols - n)
    for i, j in slack_col.items():
        names[j] = f"s{i + 1}" if senses[i] == LE else f"e{i + 1}"
    for i, j in art_col.items():
        names[j] = f"a{i + 1}"
    info = {"n": n, "m": m, "ncols": ncols, "slack_col": slack_col, "art_col": art_col, "sign": sign, "senses": senses, "names": names, "c": c}
    return T, basis, info


def _col_label(names, n, j):
    return names[j] if j >= n else f"x{j + 1}"


def _pivot(T, row, col):
    T[row] /= T[row, col]
    for i in range(T.shape[0]):
        if i != row and T[i, col] != 0.0:
            T[i] -= T[i, col] * T[row]


def _pivot_flops(m, ncols):
    return (ncols + 1) + 2 * m * (ncols + 1)


def tableau_simplex(inst, rule="dantzig", ratio="index", keep=False, seed="0"):
    """Löst `inst` mit Pivotregel `rule` (dantzig: kleinste reduzierte Kosten; greatest: größter Zielzuwachs je Kandidat; steepest: reduzierte Kosten durch Kantenlänge; bland: kleinster Index; random: zufälliger
    Kandidat) und Quotiententest `ratio` (index: kleinster Index der Basisvariable; lexicographic: lexikographisch kleinste Zeile, beweisbar zyklenfrei). Wiederholt sich bei einer deterministischen Regel
    eine Basis, endet der Lauf mit Status "cycled". Mit `keep` werden Tableau-Schnappschüsse abgelegt."""
    if rule not in RULES or ratio not in RATIO_RULES:
        raise ValueError((rule, ratio))
    T, basis, info = standard_form(inst)
    m, n, ncols = info["m"], info["n"], info["ncols"]
    art = set(info["art_col"].values())
    names = info["names"]
    res = SimplexResult(status="optimal", col_names=tuple(_col_label(names, n, j) for j in range(ncols)), m=m, n_cols=ncols, rule=rule, ratio=ratio)
    full = np.zeros((m + 1, ncols + 1))
    full[:m] = T
    T = full
    init_basis = list(basis)
    rng = random.Random(f"piv-{seed}-{rule}")

    def snapshot(phase, pivot):
        if keep and len(res.snapshots) < MAX_SNAPSHOTS:
            res.snapshots.append({"phase": phase, "T": T.copy(), "basis": tuple(basis), "pivot": pivot})

    def x_of_basis():
        xf = np.zeros(ncols)
        for i, j in enumerate(basis):
            xf[j] = T[i, -1]
        return xf

    def choose_entering(cand, r):
        if rule == "bland":
            return cand[0]
        if rule == "dantzig":
            return min(cand, key=lambda j: (r[j], j))
        if rule == "random":
            return cand[rng.randrange(len(cand))]
        if rule == "steepest":
            res.price_flops += (2 * m + 2) * len(cand)
            return min(cand, key=lambda j: (r[j] / (1.0 + float(np.dot(T[:m, j], T[:m, j]))) ** 0.5, j))
        best, best_gain = None, -1.0
        res.price_flops += 2 * m * len(cand)
        for j in cand:
            col = T[:m, j]
            pos = [i for i in range(m) if col[i] > TOL]
            gain = float("inf") if not pos else -r[j] * min(T[i, -1] / col[i] for i in pos)
            if gain > best_gain + 1e-12:
                best, best_gain = j, gain
        return best

    def choose_leaving(tied, enter):
        if ratio == "index" or len(tied) == 1:
            return min(tied, key=lambda i: basis[i])
        left = list(tied)
        for c in [-1] + init_basis:
            vals = {i: T[i, c] / T[i, enter] for i in left}
            vmin = min(vals.values())
            res.price_flops += len(left)
            left = [i for i in left if vals[i] <= vmin + TOL * max(1.0, abs(vmin))]
            if len(left) == 1:
                break
        return min(left, key=lambda i: basis[i])

    def run(cvec, allowed, phase):
        cB = cvec[basis]
        T[m, :] = cB @ T[:m, :]
        T[m, :-1] -= cvec
        snapshot(phase, None)
        seen = {frozenset(basis): len(res.pivots)}
        zero_run = 0
        limit = 200 * (m + ncols) + 1000
        deterministic = rule != "random"
        while True:
            r = T[m, :-1]
            cand = [j for j in allowed if r[j] < -TOL]
            if not cand:
                if zero_run:
                    res.stall_runs.append(zero_run)
                return "optimal"
            enter = choose_entering(cand, r)
            col = T[:m, enter]
            pos = [i for i in range(m) if col[i] > TOL]
            if not pos:
                if zero_run:
                    res.stall_runs.append(zero_run)
                return "unbounded"
            ratios = {i: T[i, -1] / col[i] for i in pos}
            rmin = min(ratios.values())
            tied = [i for i in pos if ratios[i] <= rmin + TOL * max(1.0, abs(rmin))]
            leave = choose_leaving(tied, enter)
            leave_var = basis[leave]
            step = max(rmin, 0.0)
            _pivot(T, leave, enter)
            res.flops += _pivot_flops(m, ncols)
            basis[leave] = enter
            xf = x_of_basis()
            degenerate = step <= TOL
            if degenerate:
                zero_run += 1
            elif zero_run:
                res.stall_runs.append(zero_run)
                zero_run = 0
            art_sum = float(sum(xf[j] for j in art)) if art else 0.0
            piv = Pivot(len(res.pivots) + 1, phase, enter, leave, leave_var, float(step), len(tied), float(T[m, -1]), degenerate, tuple(basis), tuple(float(v) for v in xf[:n]), art_sum <= 1e-7,
                        sum(1 for i, j in enumerate(basis) if xf[j] <= TOL))
            res.pivots.append(piv)
            snapshot(phase, piv.k)
            key = frozenset(basis)
            if key in seen:
                if deterministic:
                    res.cycled, res.cycle_len = True, piv.k - seen[key]
                    if zero_run:
                        res.stall_runs.append(zero_run)
                    return "cycled"
                res.repeats += 1
            else:
                seen[key] = piv.k
            if len(res.pivots) > limit:
                return "limit"

    if art:
        c1 = np.zeros(ncols)
        for j in art:
            c1[j] = -1.0
        status1 = run(c1, list(range(ncols)), 1)
        res.phase1_pivots = len(res.pivots)
        if status1 in ("cycled", "limit"):
            res.status = status1
            return res
        if T[m, -1] < -1e-7:
            res.status = "infeasible"
            return res
        for i in range(m):
            if basis[i] in art:
                cand = [j for j in range(ncols) if j not in art and abs(T[i, j]) > TOL]
                if cand:
                    enter = max(cand, key=lambda j: (abs(T[i, j]), -j))
                    leave_var = basis[i]
                    _pivot(T, i, enter)
                    res.flops += _pivot_flops(m, ncols)
                    basis[i] = enter
                    xf = x_of_basis()
                    res.pivots.append(Pivot(len(res.pivots) + 1, 1, enter, i, leave_var, 0.0, 1, float(T[m, -1]), True, tuple(basis), tuple(float(v) for v in xf[:n]), True,
                                            sum(1 for k, j in enumerate(basis) if xf[j] <= TOL)))
                    snapshot(1, res.pivots[-1].k)
                else:
                    res.redundant_rows += 1
        res.phase1_pivots = len(res.pivots)
    c2 = np.zeros(ncols)
    c2[:n] = info["c"]
    status = run(c2, [j for j in range(ncols) if j not in art], 2)
    if status != "optimal":
        res.status = status
        return res
    xf = x_of_basis()
    res.x = tuple(float(v) for v in xf[:n])
    res.obj = float(np.dot(info["c"], xf[:n]))
    duals = []
    for i in range(m):
        if i in info["art_col"]:
            y = T[m, info["art_col"][i]]
        elif info["senses"][i] == LE:
            y = T[m, info["slack_col"][i]]
        else:
            y = -T[m, info["slack_col"][i]]
        duals.append(float(y * info["sign"][i]))
    res.duals = tuple(duals)
    return res


# --- Prüfgrößen (für Tests und Auswertung) ---------------------------------------------------------------------------------------------------------


def primal_violation(inst, x):
    """Größte Verletzung der Bedingungen und der Nichtnegativität durch x (0 = zulässig)."""
    A, b, _c = inst.arrays()
    x = np.asarray(x, dtype=float)
    worst = max(0.0, float(-x.min())) if len(x) else 0.0
    for i in range(inst.m):
        lhs = float(A[i] @ x)
        s = inst.senses[i]
        v = max(0.0, lhs - b[i]) if s == LE else (max(0.0, b[i] - lhs) if s == GE else abs(lhs - b[i]))
        worst = max(worst, v)
    return worst


def dual_violation(inst, y):
    """Größte Verletzung der Dual-Zulässigkeit A^T y >= c (max-Problem, x >= 0) und der Vorzeichen (LE: y >= 0, GE: y <= 0)."""
    A, _b, c = inst.arrays()
    y = np.asarray(y, dtype=float)
    worst = float(max(0.0, (c - A.T @ y).max())) if inst.n else 0.0
    for i in range(inst.m):
        s = inst.senses[i]
        worst = max(worst, max(0.0, -y[i]) if s == LE else (max(0.0, y[i]) if s == GE else 0.0))
    return worst


def vertices_2d(inst):
    """Alle zulässigen Ecken einer Instanz mit zwei Variablen (Schnittpunkte je zweier Randgeraden, geprüft gegen alle Bedingungen), gegen den Uhrzeigersinn um ihren Schwerpunkt geordnet."""
    if inst.n != 2:
        raise ValueError("nur für zwei Variablen")
    A, b, _c = inst.arrays()
    lines = [(A[i, 0], A[i, 1], b[i]) for i in range(inst.m)] + [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    pts = []
    for p in range(len(lines)):
        for q in range(p + 1, len(lines)):
            a1, b1, c1 = lines[p]
            a2, b2, c2 = lines[q]
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-12:
                continue
            x = np.array([(c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det])
            if primal_violation(inst, x) <= 1e-9 and not any(np.allclose(x, q_, atol=1e-9) for q_ in pts):
                pts.append(x)
    if not pts:
        return []
    ctr = np.mean(pts, axis=0)
    pts.sort(key=lambda v: float(np.arctan2(v[1] - ctr[1], v[0] - ctr[0])))
    return [(float(v[0]), float(v[1])) for v in pts]
