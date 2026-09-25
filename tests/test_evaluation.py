"""Auswertung: Analyse, Regelvergleich, Kennzahlen über feste Instanzen, Regeltabelle, Kurven, Stillstand, Zyklensuche, Sweeps."""

import math
import pytest

import piv_algorithm as A
import piv_constants as C
import piv_evaluation as ev
import piv_scenario as S


def test_analyse_of_the_textbook_is_consistent_and_cached():
    a = ev.analyse(ev.Settings())
    assert a.status == "optimal" and a.res.obj == 36.0 and a.res.total_pivots == 2 and len(a.vertices) == 5 and a.certificate == {"primal": 0.0, "dual": 0.0, "gap": 0.0}
    assert ev.analyse(ev.Settings()) is ev.analyse(ev.Settings())
    assert ev.analyse(ev.Settings("beale")).certificate is None


def test_analyse_uses_the_selected_rule_and_ratio():
    a = ev.analyse(ev.Settings("beale"))
    b = ev.analyse(ev.Settings("beale", rule="bland"))
    c = ev.analyse(ev.Settings("beale", ratio="lexicographic"))
    assert (a.status, b.status, c.status) == ("cycled", "optimal", "optimal") and (a.res.rule, b.res.rule, c.res.ratio) == ("dantzig", "bland", "lexicographic")


def test_compare_rules_runs_all_five_and_agrees_on_the_optimum():
    cmp = ev.compare_rules(ev.Settings("mixed", 10, 8, 0.5, 4))
    assert list(cmp) == list(A.RULES) and len({round(r.obj, 6) for r in cmp.values()}) == 1 and all(r.status == "optimal" for r in cmp.values())
    assert cmp["dantzig"].price_flops == 0 and cmp["steepest"].price_flops > 0 and cmp["greatest"].price_flops > 0


def test_metrics_fields_and_relations():
    r = ev.metrics(S.generate("transport", 4, 8, 0.5, 1), "steepest", "lexicographic")
    assert r["status"] == "optimal" and r["flops"] == r["pivot_flops"] + r["price_flops"] and r["price_share"] == pytest.approx(r["price_flops"] / r["flops"]) and 0 <= r["degenerate_share"] <= 1
    assert ev.metrics(S.beale_instance())["cycled"] and not ev.metrics(S.beale_instance(), "bland")["cycled"]


def test_run_config_bands_are_ordered_deterministic_and_count_cycles_separately():
    r = ev.run_config(ev.Settings("random", 10, 10, 0.5, 0, "steepest"))
    assert r["n_runs"] == 5 and r["solved"] == 5 and r["cycled"] == 0
    for key in ev.KEYS:
        assert r[f"{key}_lo"] <= r[key] + 1e-12 and r[key] <= r[f"{key}_hi"] + 1e-12
    assert ev.run_config(ev.Settings("random", 10, 10, 0.5, 0, "steepest")) == r
    b = ev.run_config(ev.Settings("beale"))
    assert b["solved"] == 0 and b["cycled"] == 5 and math.isnan(b["pivots"])


def test_rule_table_ratios_are_relative_to_dantzig():
    tab = ev.rule_table(ev.Settings("mixed", 12, 12, 0.5, 0))
    assert list(tab) == list(A.RULES) and tab["dantzig"]["pivots_vs_dantzig"] == 1.0 and tab["dantzig"]["flops_vs_dantzig"] == 1.0
    assert tab["bland"]["pivots_vs_dantzig"] > 1.0 and tab["steepest"]["price_share"] > 0 and tab["bland"]["price_share"] == 0


def test_rule_curve_shape_and_shared_instances():
    cv = ev.rule_curve(ev.Settings("mixed", 10, 10, 0.5, 0), sizes=(5, 10), seeds=range(200000, 200006))
    assert list(cv) == list(A.RULES) and [x["size"] for x in cv["dantzig"]] == [5, 10] and all(x["solved"] == 6 for r in cv.values() for x in r)
    assert cv["dantzig"][1]["pivots"] > cv["dantzig"][0]["pivots"]
    tr = ev.rule_curve(ev.Settings("transport", 4, 8, 0.5, 0), sizes=(5, 10), seeds=range(200000, 200004))
    assert [(x["m"], x["n"]) for x in tr["dantzig"]] == [C.TRANSPORT_SIZES[5], C.TRANSPORT_SIZES[10]]


def test_stall_table_has_nullschritte_only_in_the_transport_rows():
    rows = ev.stall_table(seeds=range(200000, 200008))
    assert [r["kind"] for r in rows] == ["random", "mixed", "degenerate", "transport", "transport"]
    for r in rows:
        for rule in A.RULES:
            assert (r[rule] > 0) == (r["kind"] == "transport"), (r["kind"], rule)


def test_cycle_search_reports_counts_per_type():
    out = ev.cycle_search(count=12, kinds=(("random", 6, 5), ("transport", 3, 4)))
    assert out == {("random", 6, 5): (0, 12), ("transport", 3, 4): (0, 12)}


def test_sweeps_cover_every_parameter():
    base = ev.Settings("random", 8, 8, 0.5, 0)
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    for param, values in (("size", (5, 10)), ("density", (0.3, 0.7)), ("kind", ("random", "transport"))):
        rows = ev.sweep(param, base, values)
        assert [r["value"] for r in rows] == list(values) and all(set(A.RULES) <= set(r) and r["dantzig"]["solved"] == 5 for r in rows)
