"""Korrektheitskette: jede Regel == HiGHS, Beales Zyklus, jede Regel einzeln unabhängig nachgerechnet, lexikographischer Quotiententest, Zykluserkennung, Buchführung, Sonderfälle."""

import math

import numpy as np
import pytest

import piv_algorithm as A
import piv_scenario as S
from tests.test_scenario import _highs, reference_status


def _solve(inst, **kw):
    return A.tableau_simplex(inst, **kw)


def _custom(rows, b, c, senses):
    n = len(c)
    return S.Instance(tuple(tuple(float(v) for v in r) for r in rows), tuple(float(v) for v in b), tuple(float(v) for v in c), tuple(senses), tuple(f"x{j}" for j in range(n)), tuple(f"r{i}" for i in range(len(b))), "custom")


def _families():
    for seed in range(12):
        yield S.generate("random", 8, 6, 0.5, seed)
        yield S.generate("mixed", 8, 6, 0.5, seed)
        yield S.generate("degenerate", 10, 8, 0.5, seed)
        yield S.generate("transport", 3, 4, 0.5, seed)


@pytest.mark.parametrize("rule", A.RULES)
@pytest.mark.parametrize("ratio", A.RATIO_RULES)
def test_every_rule_reaches_the_highs_optimum_with_certificates(rule, ratio):
    count = 0
    for inst in list(_families()) + [S.textbook_instance(), S.degenerate_instance(), S.beale_instance()]:
        r, h = _solve(inst, rule=rule, ratio=ratio), _highs(inst)
        if r.status == "cycled":
            assert rule != "random" and ratio == "index" and rule != "bland"          # nur die naive Kombination darf kreisen
            continue
        assert r.status == "optimal" and h.status == 0
        assert r.obj == pytest.approx(-h.fun, rel=1e-7, abs=1e-7)
        assert A.primal_violation(inst, r.x) < 1e-7 and A.dual_violation(inst, r.duals) < 1e-6
        assert float(np.dot(r.duals, np.array(inst.b))) == pytest.approx(r.obj, rel=1e-7, abs=1e-7)
        count += 1
    assert count >= 48


def test_all_rules_agree_on_infeasible_and_unbounded():
    for rule in A.RULES:
        for ratio in A.RATIO_RULES:
            assert _solve(S.infeasible_instance(), rule=rule, ratio=ratio).status == "infeasible"
            assert _solve(S.unbounded_instance(), rule=rule, ratio=ratio).status == "unbounded"


def test_random_lps_with_mixed_signs_agree_with_highs_for_every_rule():
    rng = np.random.default_rng(21)
    seen = set()
    for _ in range(120):
        m, n = int(rng.integers(1, 6)), int(rng.integers(1, 5))
        inst = _custom(np.round(rng.uniform(-3, 4, (m, n)), 1), np.round(rng.uniform(-5, 20, m), 1), np.round(rng.uniform(-3, 6, n), 1),
                       [str(s) for s in rng.choice([S.LE, S.GE, S.EQ], size=m, p=[0.55, 0.3, 0.15])])
        ref = reference_status(inst)
        for rule in ("dantzig", "greatest", "steepest", "bland"):
            r = _solve(inst, rule=rule, ratio="lexicographic")
            assert r.status == ref, (rule, inst)
            if ref == "optimal":
                assert r.obj == pytest.approx(-_highs(inst).fun, rel=1e-6, abs=1e-6)
        seen.add(ref)
    assert seen == {"optimal", "infeasible", "unbounded"}


def test_beale_cycles_with_dantzig_and_the_index_rule():
    inst = S.beale_instance()
    assert -_highs(inst).fun == pytest.approx(1.25)
    r = _solve(inst, keep=True)
    assert r.status == "cycled" and r.cycled and r.cycle_len == 6 and r.total_pivots == 6
    assert [p.basis for p in r.pivots] == [(0, 5, 6), (0, 1, 6), (2, 1, 6), (2, 3, 6), (4, 3, 6), (4, 5, 6)]
    assert frozenset(r.pivots[-1].basis) == frozenset((4, 5, 6)) and all(p.degenerate and p.obj == 0.0 for p in r.pivots)   # zurück an der Start-Basis, der Zielwert bewegt sich nie
    assert r.stall_runs == [6] and r.obj != r.obj and r.x == ()


def test_beale_terminates_with_bland_and_with_the_lexicographic_ratio_test():
    inst = S.beale_instance()
    bl = _solve(inst, rule="bland")
    assert bl.status == "optimal" and bl.total_pivots == 6 and bl.obj == pytest.approx(1.25) and bl.x == pytest.approx((1.0, 0.0, 1.0, 0.0))
    lex = _solve(inst, ratio="lexicographic")
    assert lex.status == "optimal" and lex.total_pivots == 2 and lex.obj == pytest.approx(1.25)
    for rule in ("greatest", "steepest", "random"):
        r = _solve(inst, rule=rule)
        assert r.status == "optimal" and r.obj == pytest.approx(1.25) and r.total_pivots <= 3


def test_dantzig_cycling_needs_the_index_tiebreak_of_the_ratio_test():
    inst = S.beale_instance()
    assert _solve(inst, rule="dantzig", ratio="index").cycled and not _solve(inst, rule="dantzig", ratio="lexicographic").cycled


def _first_entering(inst, **kw):
    r = _solve(inst, **kw)
    return r.pivots[0].enter if r.pivots else None


def test_steepest_edge_picks_the_smallest_ratio_of_reduced_cost_to_edge_length():
    for seed in range(40):
        inst = S.generate("random", 8, 6, 0.6, seed)
        A_, _b, c = inst.arrays()
        want = min((j for j in range(inst.n) if c[j] > 0), key=lambda j: (-c[j] / math.sqrt(1.0 + float(A_[:, j] @ A_[:, j])), j))       # Startbasis = Schlupf: Kante = (e_j, -A_j)
        assert _first_entering(inst, rule="steepest") == want


def test_greatest_improvement_picks_the_largest_objective_gain_of_its_own_step():
    for seed in range(40):
        inst = S.generate("random", 8, 6, 0.6, seed)
        A_, b, c = inst.arrays()

        def gain(j):
            rows = [b[i] / A_[i, j] for i in range(inst.m) if A_[i, j] > 1e-9]
            return c[j] * min(rows) if rows else math.inf
        want = max((j for j in range(inst.n) if c[j] > 0), key=lambda j: (gain(j), -j))
        assert _first_entering(inst, rule="greatest") == want


def test_dantzig_bland_and_random_choose_as_defined():
    for seed in range(30):
        inst = S.generate("random", 8, 6, 0.6, seed)
        _A, _b, c = inst.arrays()
        pos = [j for j in range(inst.n) if c[j] > 0]
        assert _first_entering(inst, rule="dantzig") == max(pos, key=lambda j: (c[j], -j)) and _first_entering(inst, rule="bland") == pos[0]
        assert _first_entering(inst, rule="random") in pos
    firsts = {_first_entering(S.generate("random", 10, 8, 0.6, 5), rule="random", seed=str(s)) for s in range(30)}
    assert len(firsts) > 1
    a = _solve(S.generate("random", 10, 8, 0.6, 5), rule="random", seed="7")
    b = _solve(S.generate("random", 10, 8, 0.6, 5), rule="random", seed="7")
    assert [p.enter for p in a.pivots] == [p.enter for p in b.pivots]


def test_lexicographic_matches_the_index_rule_without_ties_and_never_cycles_with_ties():
    for seed in range(20):
        inst = S.generate("random", 10, 8, 0.5, seed)
        assert [p.leave_row for p in _solve(inst).pivots] == [p.leave_row for p in _solve(inst, ratio="lexicographic").pivots]
    for seed in range(120):
        for inst in (S.generate("degenerate", 12, 9, 0.5, seed), S.generate("transport", 3, 5, 0.5, seed)):
            r = _solve(inst, ratio="lexicographic")
            assert r.status == "optimal" and not r.cycled
    ties_seen = sum(any(p.ties > 1 for p in _solve(S.generate("degenerate", 12, 9, 0.5, s), ratio="lexicographic").pivots) for s in range(120))
    assert ties_seen > 20                                                                        # der lexikographische Zweig wird wirklich ausgeführt


def test_lexicographic_branch_is_executed_and_can_choose_differently_from_the_index_rule():
    differs = 0
    for seed in range(200):
        inst = S.generate("transport", 3, 5, 0.5, seed)
        idx, lex = _solve(inst), _solve(inst, ratio="lexicographic")
        if [(p.enter, p.leave_row) for p in idx.pivots] != [(p.enter, p.leave_row) for p in lex.pivots]:
            differs += 1
            assert lex.price_flops > 0
    assert differs > 0


def test_cycle_detection_means_a_repeated_basis_and_no_false_alarm():
    for inst in list(_families()):
        r = _solve(inst, keep=True)
        if r.status == "optimal":
            bases = [frozenset(s["basis"]) for s in r.snapshots]
            assert len(set(bases)) == len(bases) or any(s["phase"] == 2 for s in r.snapshots)
            assert not r.cycled and r.cycle_len == 0
    inst = S.beale_instance()
    r = _solve(inst, keep=True)
    bases = [frozenset(s["basis"]) for s in r.snapshots]
    assert bases[0] == bases[-1] and len(set(bases)) == 6


def test_random_rule_never_reports_a_cycle_but_counts_revisits():
    for seed in range(30):
        r = _solve(S.beale_instance(), rule="random", seed=str(seed))
        assert not r.cycled and r.status == "optimal" and r.repeats >= 0


def test_bookkeeping_price_operations_and_stall_runs():
    one = _custom([[2.0]], [10.0], [3.0], [S.LE])
    m = one.m
    assert _solve(one, rule="dantzig").price_flops == 0 and _solve(one, rule="bland").price_flops == 0 and _solve(one, rule="random").price_flops == 0
    assert _solve(one, rule="greatest").price_flops == 2 * m and _solve(one, rule="steepest").price_flops == 2 * m + 2
    for inst in list(_families()):
        for rule in A.RULES:
            r = _solve(inst, rule=rule, ratio="lexicographic")
            assert r.total_flops == r.flops + r.price_flops and r.flops == r.total_pivots * ((r.n_cols + 1) + 2 * r.m * (r.n_cols + 1))
            assert sum(r.stall_runs) == r.degenerate_pivots and all(v >= 1 for v in r.stall_runs)
    assert _solve(S.textbook_instance(), rule="dantzig").price_flops == 0


def test_transport_instances_are_balanced_and_highly_degenerate():
    inst = S.generate("transport", 4, 6, 0.5, 3)
    A_, b, _c = inst.arrays()
    supply = sum(b[i] for i in range(4))
    demand = sum(b[i] for i in range(4, 10))
    assert supply == demand and (inst.m, inst.n) == (10, 24) and inst.senses[:4] == (S.LE,) * 4 and inst.senses[4:] == (S.GE,) * 6
    zero = sum(_solve(S.generate("transport", 4, 6, 0.5, s)).degenerate_pivots for s in range(20))
    ties = sum(sum(p.ties > 1 for p in _solve(S.generate("transport", 4, 6, 0.5, s)).pivots) for s in range(20))
    assert zero > 10 and ties > 10                                                               # Nullschritte und Gleichstände kommen hier wirklich vor
    assert all(_solve(S.generate("transport", 4, 6, 0.5, s)).redundant_rows >= 0 for s in range(5))


def test_unknown_rule_is_rejected_and_special_cases_hold():
    with pytest.raises(ValueError):
        _solve(S.textbook_instance(), rule="nope")
    with pytest.raises(ValueError):
        _solve(S.textbook_instance(), ratio="nope")
    zero_c = _custom([[1.0, 1.0]], [4.0], [0.0, 0.0], [S.LE])
    for rule in A.RULES:
        r = _solve(zero_c, rule=rule)
        assert r.obj == 0.0 and r.total_pivots == 0 and r.price_flops == 0
    dup = _custom([[1.0, 2.0], [1.0, 2.0], [2.0, 4.0]], [6.0, 6.0, 12.0], [1.0, 1.0], [S.EQ, S.EQ, S.EQ])
    for rule in ("dantzig", "steepest", "greatest", "bland"):
        assert _solve(dup, rule=rule).obj == pytest.approx(6.0)
