"""Unabhängiges Orakel: jeder Pivot jedes Laufs wird in exakter Bruchrechnung (fractions) aus der Basis neu berechnet.

Ohne Tableau-Update und ohne Gleitkomma: aus der Basis vor dem Pivot wird B^-1 [A | b] exakt gelöst, daraus die reduzierten Kosten, die Kandidatenmenge, die Regelwahl
(Dantzig, größter Zuwachs, Steepest Edge, Bland), die Zeilenwahl des Quotiententests (Index bzw. lexikographisch, mit exakten Gleichständen), Schrittweite, Zielwert und Entartung.
Zusätzlich Status und Optimalwert gegen HiGHS auf zufälligen kleinen LPs mit Gleichständen und allen drei Status; lexikographisch und Bland terminieren immer.
"""

import math
import random
from fractions import Fraction as F

import pytest

import piv_algorithm as A
import piv_scenario as S


def _exact_form(inst):
    """Gleichungsform wie in der Demo beschrieben (Zeilen mit b < 0 werden gedreht, Schlupf-, dann künstliche Spalten), aber in Brüchen aus den Dezimalwerten."""
    m, n = inst.m, inst.n
    rows, rhs, sen = [], [], []
    for i in range(m):
        a, r, s = [F(repr(float(v))) for v in inst.A[i]], F(repr(float(inst.b[i]))), inst.senses[i]
        if r < 0:
            a, r, s = [-v for v in a], -r, {S.LE: S.GE, S.GE: S.LE, S.EQ: S.EQ}[s]
        rows.append(a), rhs.append(r), sen.append(s)
    nc, slack, art = n, {}, {}
    for i, s in enumerate(sen):
        if s in (S.LE, S.GE):
            slack[i], nc = nc, nc + 1
    for i, s in enumerate(sen):
        if s in (S.GE, S.EQ):
            art[i], nc = nc, nc + 1
    M, init = [[F(0)] * (nc + 1) for _ in range(m)], []
    for i in range(m):
        M[i][:n], M[i][-1] = rows[i], rhs[i]
        if i in slack:
            M[i][slack[i]] = F(1) if sen[i] == S.LE else F(-1)
        if i in art:
            M[i][art[i]] = F(1)
        init.append(art[i] if i in art else slack[i])
    return M, init, set(art.values()), n, nc, [F(repr(float(v))) for v in inst.c]


def _tableau(M, basis, m, nc):
    aug = [[M[i][basis[j]] for j in range(m)] + [F(int(i == j)) for j in range(m)] for i in range(m)]
    for c in range(m):
        p = next(i for i in range(c, m) if aug[i][c] != 0)
        aug[c], aug[p] = aug[p], aug[c]
        aug[c] = [v / aug[c][c] for v in aug[c]]
        for i in range(m):
            if i != c and aug[i][c] != 0:
                f = aug[i][c]
                aug[i] = [a - f * b for a, b in zip(aug[i], aug[c])]
    binv = [row[m:] for row in aug]
    return [[sum(binv[i][k] * M[k][j] for k in range(m)) for j in range(nc + 1)] for i in range(m)]


def _check_pivots(inst, res, rule, ratio):
    M, init, art, n, nc, cvec = _exact_form(inst)
    m = len(M)
    bases = [tuple(init)] + [p.basis for p in res.pivots]
    for k, p in enumerate(res.pivots):
        pre, post = list(bases[k]), list(bases[k + 1])
        T = _tableau(M, pre, m, nc)
        cph = [F(-1) if j in art else F(0) for j in range(nc)] if p.phase == 1 else [cvec[j] if j < n else F(0) for j in range(nc)]
        r = [sum(cph[pre[i]] * T[i][j] for i in range(m)) - cph[j] for j in range(nc)]
        allowed = range(nc) if p.phase == 1 else [j for j in range(nc) if j not in art]
        cand = [j for j in allowed if r[j] < 0]
        if p.phase == 1 and not cand:                                                  # Austausch einer künstlichen Basisvariablen nach Phase 1
            assert pre[p.leave_row] in art and p.enter not in art and T[p.leave_row][p.enter] != 0
            continue
        e = p.enter
        assert e in cand, (k, "Eintretende ist kein Kandidat")
        if rule == "bland":
            assert e == cand[0]
        elif rule == "dantzig":
            assert float(r[e] - min(r[j] for j in cand)) <= 1e-9
        elif rule == "steepest":
            def score(j):
                return float(r[j]) / math.sqrt(1.0 + float(sum(T[i][j] ** 2 for i in range(m))))
            assert score(e) <= min(score(j) for j in cand) + 1e-9
        elif rule == "greatest":
            def gain(j):
                pos = [i for i in range(m) if T[i][j] > 0]
                return math.inf if not pos else float(-r[j] * min(T[i][-1] / T[i][j] for i in pos))
            assert gain(e) == pytest.approx(max(gain(j) for j in cand), abs=1e-9) or gain(e) == math.inf
        pos = [i for i in range(m) if T[i][e] > 0]
        ratios = {i: T[i][-1] / T[i][e] for i in pos}
        rmin = min(ratios.values())
        tied = [i for i in pos if ratios[i] == rmin]
        assert p.ties == len(tied)
        if ratio == "index":
            want = min(tied, key=lambda i: pre[i])
        else:
            want = min(pos, key=lambda i: [T[i][-1] / T[i][e]] + [T[i][c] / T[i][e] for c in init])
        assert p.leave_row == want and p.leave_var == pre[want] and p.ratio == pytest.approx(float(rmin))
        expected_post = pre[:]
        expected_post[want] = e
        assert post == expected_post
        T2 = _tableau(M, post, m, nc)
        before = sum(cph[pre[i]] * T[i][-1] for i in range(m))
        after = sum(cph[post[i]] * T2[i][-1] for i in range(m))
        assert after >= before and p.obj == pytest.approx(float(after)) and p.degenerate == (after == before)
        assert p.zero_basics == sum(1 for i in range(m) if T2[i][-1] == 0)


def _random_lp(rng):
    m, n = rng.randint(1, 5), rng.randint(1, 5)
    rows = [[float(rng.choice([-2, -1, 0, 0, 1, 2, 3])) for _ in range(n)] for _ in range(m)]
    b = [float(rng.choice([0, 0, 1, 2, 4, 6, -1, -2])) for _ in range(m)]
    c = [float(rng.choice([-2, -1, 0, 1, 2, 3])) for _ in range(n)]
    sen = tuple(rng.choice([S.LE, S.LE, S.LE, S.GE, S.EQ]) for _ in range(m))
    return S.Instance(tuple(map(tuple, rows)), tuple(b), tuple(c), sen, tuple(f"x{j}" for j in range(n)), tuple(f"r{i}" for i in range(m)), "custom")


def _instances():
    rng = random.Random(99)
    out = [_random_lp(rng) for _ in range(45)]
    out += [S.generate("degenerate", 8, 6, 0.5, s) for s in range(6)] + [S.generate("transport", 3, 4, 0.5, s) for s in range(4)]
    out += [S.beale_instance(), S.textbook_instance(), S.degenerate_instance(), S.infeasible_instance(), S.unbounded_instance()]
    return out


@pytest.mark.parametrize("rule", A.RULES)
@pytest.mark.parametrize("ratio", A.RATIO_RULES)
def test_every_pivot_is_what_the_rule_prescribes_in_exact_arithmetic(rule, ratio):
    checked = 0
    for inst in _instances():
        res = A.tableau_simplex(inst, rule=rule, ratio=ratio)
        _check_pivots(inst, res, rule, ratio)
        checked += len(res.pivots)
    assert checked > 100


def test_status_and_optimum_agree_with_highs_and_safe_rules_always_terminate():
    pytest.importorskip("scipy")
    from tests.test_scenario import _highs, reference_status
    rng = random.Random(7)
    seen = set()
    for inst in [_random_lp(rng) for _ in range(120)] + [S.beale_instance()]:
        ref = reference_status(inst)
        seen.add(ref)
        for rule in A.RULES:
            for ratio in A.RATIO_RULES:
                res = A.tableau_simplex(inst, rule=rule, ratio=ratio)
                if res.status == "cycled":
                    assert rule != "bland" and ratio == "index"                      # Bland und der lexikographische Test terminieren immer
                    continue
                assert res.status == ref
                if ref == "optimal":
                    assert res.obj == pytest.approx(-_highs(inst).fun, rel=1e-7, abs=1e-7)
    assert seen == {"optimal", "infeasible", "unbounded"}
