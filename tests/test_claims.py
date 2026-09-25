"""Jede Zahl in den Grenzen der App und im README, gegen die echten Auswertungsfunktionen (feste Seeds, plattformstabil dank random.Random; Bänder bei Zählgrößen)."""

from functools import lru_cache

import pytest

import piv_algorithm as A
import piv_constants as C
import piv_evaluation as ev
import piv_scenario as S


@lru_cache(maxsize=None)
def _curve(kind):
    return ev.rule_curve(ev.Settings(kind, 10, 10, 0.5, 0))


def _at(kind, size):
    return {r: next(x for x in _curve(kind)[r] if x["size"] == size) for r in A.RULES}


def _text():
    return open("app.py", encoding="utf-8").read()


def _readme():
    return open("README.md", encoding="utf-8").read()


def test_steepest_edge_saves_pivots_on_three_families():
    rnd, mix, tr = _at("random", 40), _at("mixed", 40), _at("transport", 40)
    assert (tr["dantzig"]["m"], tr["dantzig"]["n"]) == (6, 10)
    for got, want in ((rnd["steepest"]["pivots"], 15.5), (rnd["dantzig"]["pivots"], 21.5), (mix["steepest"]["pivots"], 51.0), (mix["dantzig"]["pivots"], 95.5), (tr["steepest"]["pivots"], 32.0), (tr["dantzig"]["pivots"], 53.5),
                      (mix["greatest"]["pivots"], 66.0)):
        assert got == pytest.approx(want, abs=max(1.5, 0.05 * want))
    assert mix["steepest"]["flops"] / mix["dantzig"]["flops"] == pytest.approx(0.63, abs=0.02)
    assert all(_at(k, 40)["steepest"]["pivots"] < _at(k, 40)["dantzig"]["pivots"] for k in ("random", "mixed", "transport"))
    _t = _text()
    for v in ("15.5 statt 21.5 Pivots", "51 statt 95.5", "32 statt 53.5", "0.63 der Operationen", "66 Pivots auf Mischinstanzen"):
        assert v in _t, v


def test_the_smart_rules_lose_on_the_textbook_in_operations():
    c = ev.compare_rules(ev.Settings())
    assert (c["dantzig"].total_flops, c["greatest"].total_flops, c["steepest"].total_flops) == (84, 102, 108)
    assert "84 Operationen für Dantzig, 102 für den größten Zuwachs, 108 für Steepest Edge" in _text()


def test_bland_and_random_need_many_more_pivots():
    rnd, mix, tr = _at("random", 40), _at("mixed", 40), _at("transport", 40)
    assert rnd["bland"]["pivots"] == pytest.approx(80.0, abs=4) and mix["bland"]["pivots"] == pytest.approx(230.0, abs=12) and tr["bland"]["pivots"] == pytest.approx(81.5, abs=4)
    for got, want in ((rnd["bland"]["pivots"] / rnd["dantzig"]["pivots"], 3.7), (mix["bland"]["pivots"] / mix["dantzig"]["pivots"], 2.4), (tr["bland"]["pivots"] / tr["dantzig"]["pivots"], 1.5)):
        assert got == pytest.approx(want, abs=0.15)
    assert rnd["random"]["pivots"] > 2 * rnd["dantzig"]["pivots"] and mix["random"]["pivots"] > 1.5 * mix["dantzig"]["pivots"]
    _t = _text()
    for v in ("3.7-fache auf Zufallsinstanzen (80 gegen 21.5)", "2.4-fache auf Mischinstanzen (230 gegen 95.5)", "1.5-fache im Transportproblem (81.5 gegen 53.5)"):
        assert v in _t, v


def test_no_cycles_on_generated_instances_or_beale_like_lps_but_beale_cycles():
    found = ev.cycle_search(400)
    assert sum(v[0] for v in found.values()) == 0 and sum(v[1] for v in found.values()) == 2000
    cyc, tot, opt = ev.beale_like_search(5000)
    assert (cyc, tot) == (0, 5000) and 2000 < opt < 3000
    beale = A.tableau_simplex(S.beale_instance())
    assert beale.cycled and beale.total_pivots == 6
    _t = _text()
    for v in ("nach 6 Pivots", "In 2000 erzeugten Instanzen", "5000 zufälligen kleinen LPs"):
        assert v in _t, v
    assert "Nur auf konstruierten Instanzen" in _t


def test_stalling_only_happens_in_the_transport_problem():
    rows = {(r["kind"], r["m"], r["n"]): r for r in ev.stall_table()}
    tr = [rows[("transport", 4, 8)], rows[("transport", 8, 12)]]
    values = [100 * r[rule] for r in tr for rule in A.RULES]
    assert min(values) == pytest.approx(10.7, abs=0.6) and max(values) == pytest.approx(19.6, abs=0.6)
    assert 100 * rows[("transport", 4, 8)]["dantzig"] == pytest.approx(12.7, abs=0.6) and 100 * rows[("transport", 8, 12)]["dantzig"] == pytest.approx(13.9, abs=0.6)
    for key in (("random", 20, 20), ("mixed", 20, 20), ("degenerate", 20, 20)):
        assert all(rows[key][rule] == 0.0 for rule in A.RULES)
    ties = sum(any(p.ties > 1 for p in A.tableau_simplex(S.generate("degenerate", 12, 9, 0.5, s)).pivots) for s in range(60))
    assert ties > 10                                                                              # Gleichstände gibt es dort trotzdem
    _t = _text()
    for v in ("10.7 bis 19.6 %", "Dantzig 12.7 % bei 4 Lagern und 8 Kunden, 13.9 % bei 8 und 12", "in Zufalls-, Misch- und Gleichstands-Instanzen sind es 0 %"):
        assert v in _t, v


def test_the_lexicographic_test_is_nearly_free():
    base = ev.Settings("transport", 4, 8, 0.5, 0)
    a = ev.run_config(base, C.FEAS_SEEDS)
    b = ev.run_config(ev.Settings("transport", 4, 8, 0.5, 0, "dantzig", "lexicographic"), C.FEAS_SEEDS)
    assert (a["pivots"], b["pivots"]) == (33.0, 31.0) and b["price_share"] == pytest.approx(0.0004, abs=0.0003) and a["price_share"] == 0.0
    assert "31 statt 33 Pivots" in _text() and "0.04 % der Operationen" in _text()


def test_readme_numbers_match_the_grenzen_numbers():
    r = _readme()
    for v in ("15.5 statt 21.5", "51 statt 95.5", "32 statt 53.5", "6 Pivots", "10.7 bis 19.6 %", "31 statt 33"):
        assert v in r, v
