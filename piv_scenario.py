"""Instanzen der Demo Tableau-Simplex: Auslastungsplanung eines Distributionszentrums als LP  max c·x,  A x (<=|>=|=) b,  x >= 0."""

import random
from dataclasses import dataclass

import numpy as np

KINDS = ("textbook", "fix_degenerate", "beale", "infeasible", "unbounded", "random", "mixed", "degenerate", "transport")
FIXTURE_KINDS = ("textbook", "fix_degenerate", "beale", "infeasible", "unbounded")
KIND_LABELS = {"textbook": "Lehrbuchbeispiel (2 Dienste)", "fix_degenerate": "Entartete Ecke (2 Dienste, 4 Ressourcen)", "beale": "Beales Zyklus (4 Dienste, 3 Ressourcen)", "infeasible": "Unzulässig (Widerspruch)",
               "unbounded": "Unbeschränkt (kein Ende)", "random": "Zufall (alle Ressourcen begrenzt)", "mixed": "Mischung (mit Mindest- und Gleichungs-Bedingungen)",
               "degenerate": "Zufall mit Gleichständen (Ressourcen an einer Ecke)", "transport": "Transportproblem (Lager und Kunden, stark entartet)"}
LE, GE, EQ = "<=", ">=", "="


@dataclass(frozen=True)
class Instance:
    A: tuple                      # m Zeilen mit je n Koeffizienten (Tupel, damit die Instanz hashbar bleibt)
    b: tuple
    c: tuple
    senses: tuple                 # je Zeile "<=", ">=" oder "="
    names: tuple                  # Namen der n Entscheidungsvariablen (Dienste)
    row_names: tuple              # Namen der m Bedingungen (Ressourcen)
    kind: str = "custom"

    @property
    def m(self):
        return len(self.b)

    @property
    def n(self):
        return len(self.c)

    def arrays(self):
        return np.array(self.A, dtype=float).reshape(self.m, self.n), np.array(self.b, dtype=float), np.array(self.c, dtype=float)


def _inst(A, b, c, senses, names, row_names, kind):
    return Instance(tuple(tuple(float(v) for v in row) for row in A), tuple(float(v) for v in b), tuple(float(v) for v in c), tuple(senses), tuple(names), tuple(row_names), kind)


def textbook_instance():
    """Zwei Dienste, drei Ressourcen: max 3 x1 + 5 x2;  x1 <= 4;  2 x2 <= 12;  3 x1 + 2 x2 <= 18. Optimum (2, 6) mit Wert 36, von Hand in zwei Pivots erreichbar."""
    return _inst([[1, 0], [0, 2], [3, 2]], [4, 12, 18], [3, 5], [LE] * 3, ["Express-Pakete", "Palettenversand"], ["Rampenzeit", "Kommissionierstunden", "Lagerfläche"], "textbook")


def degenerate_instance():
    """Das Lehrbuchbeispiel mit einer vierten Ressource x1 + x2 <= 8, die durch das Optimum (2, 6) läuft: an der Ecke sind drei Bedingungen zugleich bindend (entartet)."""
    return _inst([[1, 0], [0, 2], [3, 2], [1, 1]], [4, 12, 18, 8], [3, 5], [LE] * 4, ["Express-Pakete", "Palettenversand"], ["Rampenzeit", "Kommissionierstunden", "Lagerfläche", "Fahrzeugkapazität"], "fix_degenerate")


def beale_instance():
    """Beales Beispiel (1955) für Zyklen, in Maximierungsform: max 3/4 x1 - 20 x2 + 1/2 x3 - 6 x4 unter 1/4 x1 - 8 x2 - x3 + 9 x4 <= 0, 1/2 x1 - 12 x2 - 1/2 x3 + 3 x4 <= 0, x3 <= 1.
    Beide ersten Ressourcen haben Bestand 0 (voll entartet); Optimum 1/20 bei (1, 0, 1, 0)."""
    return _inst([[0.25, -8, -1, 9], [0.5, -12, -0.5, 3], [0, 0, 1, 0]], [0, 0, 1], [0.75, -20, 0.5, -6], [LE] * 3, ["Dienst 1", "Dienst 2", "Dienst 3", "Dienst 4"],
                 ["Ressource 1 (Bestand 0)", "Ressource 2 (Bestand 0)", "Ressource 3"], "beale")


def transport_instance(k, l, seed):
    """Transportproblem mit k Lagern und l Kunden, Angebot gleich Nachfrage (ganzzahlig, dadurch entartet): min Summe Länge · Menge, als max -Länge; Lager: Summe <= Angebot, Kunden: Summe >= Nachfrage."""
    rng = random.Random(f"piv-transport-{k}-{l}-{seed}")
    depots = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(k)]
    customers = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(l)]
    demand = [rng.randint(5, 30) for _ in range(l)]
    total = sum(demand)
    supply = [1] * k
    for _ in range(total - k):
        supply[rng.randrange(k)] += 1
    c, names = [], []
    for i in range(k):
        for j in range(l):
            c.append(-round(((depots[i][0] - customers[j][0]) ** 2 + (depots[i][1] - customers[j][1]) ** 2) ** 0.5, 1))
            names.append(f"Lager {i + 1} → Kunde {j + 1}")
    rows, b, senses, row_names = [], [], [], []
    for i in range(k):
        rows.append([1.0 if idx // l == i else 0.0 for idx in range(k * l)])
        b.append(supply[i])
        senses.append(LE)
        row_names.append(f"Lager {i + 1} (Angebot {supply[i]})")
    for j in range(l):
        rows.append([1.0 if idx % l == j else 0.0 for idx in range(k * l)])
        b.append(demand[j])
        senses.append(GE)
        row_names.append(f"Kunde {j + 1} (Nachfrage {demand[j]})")
    return _inst(rows, b, c, senses, names, row_names, "transport")


def infeasible_instance():
    """Widerspruch: x1 <= 4 (Rampenzeit) und x1 >= 6 (Mindestmenge) zugleich."""
    return _inst([[1, 0], [0, 2], [1, 0]], [4, 12, 6], [3, 5], [LE, LE, GE], ["Express-Pakete", "Palettenversand"], ["Rampenzeit", "Kommissionierstunden", "Mindestmenge Express"], "infeasible")


def unbounded_instance():
    """Kein Ende in Sicht: max x1 + x2 unter x1 - x2 <= 2 und -x1 + x2 <= 3 wächst entlang der Richtung (1, 1) ohne Grenze."""
    return _inst([[1, -1], [-1, 1]], [2, 3], [1, 1], [LE, LE], ["Express-Pakete", "Palettenversand"], ["Bedingung 1", "Bedingung 2"], "unbounded")


def _names(m, n):
    return [f"Dienst {j + 1}" for j in range(n)], [f"Ressource {i + 1}" for i in range(m)]


def _coefficients(rng, m, n, density):
    """Nichtnegative Verbrauchskoeffizienten mit Dichte `density`; jede Zeile und Spalte hat mindestens einen Eintrag, Zeile 0 ist dicht (Gesamtkapazität), damit alles beschränkt bleibt."""
    A = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if rng.random() < density:
                A[i][j] = round(rng.uniform(0.5, 5.0), 2)
    for j in range(n):
        A[0][j] = round(rng.uniform(0.5, 5.0), 2)
    for i in range(m):
        if not any(A[i]):
            A[i][rng.randrange(n)] = round(rng.uniform(0.5, 5.0), 2)
    return A


def generate(kind, m, n, density, seed):
    """Deterministische Instanz je (kind, m, n, density, seed); Mersenne-Twister mit Zeichenketten-Seed, plattformstabil."""
    if kind in FIXTURE_KINDS:
        return {"textbook": textbook_instance, "fix_degenerate": degenerate_instance, "beale": beale_instance, "infeasible": infeasible_instance, "unbounded": unbounded_instance}[kind]()
    if kind == "transport":
        return transport_instance(m, n, seed)
    rng = random.Random(f"piv-{kind}-{m}-{n}-{density}-{seed}")
    names, row_names = _names(m, n)
    c = [round(rng.uniform(1.0, 10.0), 2) for _ in range(n)]
    A = _coefficients(rng, m, n, density)
    if kind == "random":
        b = [round(rng.uniform(40.0, 120.0), 1) for _ in range(m)]
        return _inst(A, b, c, [LE] * m, names, row_names, kind)
    if kind == "mixed":
        x0 = [rng.uniform(1.0, 8.0) for _ in range(n)]
        senses, b = [LE], []
        eq_left = n // 2
        for i in range(1, m):
            r = rng.random()
            sense = LE if r < 0.55 else (GE if r < 0.85 else EQ)
            if sense == EQ:
                eq_left -= 1
                if eq_left < 0:
                    sense = GE
            senses.append(sense)
        for i in range(m):
            act = sum(A[i][j] * x0[j] for j in range(n))
            slack = rng.uniform(0.5, 25.0)
            if senses[i] == EQ:
                b.append(act)                                                                    # nicht runden: sonst können Gleichungen mit gleichem Träger unvereinbar werden
                continue
            b.append(round(act + slack if senses[i] == LE else act - slack, 2))
            if senses[i] == GE and b[-1] < 0:
                b[-1] = 0.0
        return _inst(A, b, c, senses, names, row_names, kind)
    if kind == "degenerate":
        base = max(1, (m + 1) // 2)
        b = [round(rng.uniform(40.0, 120.0), 1) for _ in range(base)]
        for i in range(base, m):
            p, q = rng.randrange(base), rng.randrange(base)
            A[i] = [A[p][j] + (A[q][j] if q != p else 0.0) for j in range(n)]
            b.append(round(b[p] + (b[q] if q != p else 0.0), 1))
        A = [[round(v, 2) for v in row] for row in A]
        return _inst(A, b, c, [LE] * m, names, row_names, kind)
    raise ValueError(kind)
