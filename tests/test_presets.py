"""Presets: gültige Werte, Bänder (Median über die 5 festen Instanzen) und jede Zahl im Hilfetext gegen die echten Auswertungsfunktionen."""

import pytest

import piv_algorithm as A
import piv_constants as C
import piv_evaluation as ev
from piv_presets import PRESET_KEYS, SETTING_SPECS


def _settings(p):
    kind = p["kind"]
    m, n = (p["k"], p["l"]) if kind == "transport" else (p["m"], p["n"])
    return ev.Settings(kind, m, n, p["density"], p["seed"], p["rule"], p["ratio"])


def _analysis(name):
    return ev.analyse(_settings(C.PRESETS[name]))


def _cmp(name):
    return ev.compare_rules(_settings(C.PRESETS[name]))


def _has(name, *values):
    text = C.PRESET_HELP[name]
    for v in values:
        assert v in text, (name, v)


def test_every_preset_has_valid_values_and_a_help_text():
    assert list(C.PRESETS) == list(C.PRESET_HELP) and len(C.PRESETS) == 10
    for name, p in C.PRESETS.items():
        assert set(p) == set(PRESET_KEYS), name
        for key, state_key in PRESET_KEYS.items():
            spec = SETTING_SPECS[state_key]
            assert spec.caster(p[key]) == p[key], (name, key)
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
        assert C.PRESET_HELP[name].strip(), name


@pytest.mark.parametrize("name", list(C.PRESET_EXPECTED_BANDS))
def test_preset_bands_over_the_five_fixed_instances(name):
    metric, lo, hi = C.PRESET_EXPECTED_BANDS[name]
    assert lo <= ev.run_config(_settings(C.PRESETS[name]))[metric] <= hi


def test_help_standard():
    c = _cmp("Standardfall (Lehrbuchbeispiel)")
    assert all(r.obj == 36.0 for r in c.values()) and [c[r].total_pivots for r in A.RULES] == [2, 2, 2, 3, 2]
    assert (c["dantzig"].total_flops, c["greatest"].total_flops, c["steepest"].total_flops) == (84, 102, 108)
    _has("Standardfall (Lehrbuchbeispiel)", "Optimum 36", "Bland 3", "84 Rechenoperationen", "102", "108")


def test_help_beale_cycle():
    a = _analysis("Beales Zyklus")
    assert a.status == "cycled" and a.res.total_pivots == 6 and a.res.cycle_len == 6 and a.res.degenerate_pivots == 6
    _has("Beales Zyklus", "3 Ressourcen, 4 Dienste", "6 Nullschritte", "nach 6 Pivots wieder in der Start-Basis")
    b = _analysis("Bland rettet Beale")
    assert (b.status, b.res.total_pivots, b.res.degenerate_pivots, round(b.res.obj, 6)) == ("optimal", 6, 4, 1.25) and b.res.x == pytest.approx((1.0, 0.0, 1.0, 0.0))
    _has("Bland rettet Beale", "6 Pivots, davon 4 Nullschritte", "1.25", "(1, 0, 1, 0)")
    c = _analysis("Lexikographisch rettet Beale")
    assert (c.status, c.res.total_pivots, round(c.res.obj, 6)) == ("optimal", 2, 1.25)
    _has("Lexikographisch rettet Beale", "nur 2 Pivots", "1.25")


def test_help_steepest_edge_saves_pivots():
    c = _cmp("Steepest Edge spart Pivots")
    d, s = c["dantzig"], c["steepest"]
    assert (d.total_pivots, d.total_flops, s.total_pivots, s.total_flops) == (102, 743580, 52, 454028) and s.total_pivots / d.total_pivots == pytest.approx(0.51, abs=0.005) and s.total_flops / d.total_flops == pytest.approx(0.61, abs=0.005)
    tab = ev.rule_table(_settings(C.PRESETS["Steepest Edge spart Pivots"]))
    assert (tab["dantzig"]["pivots"], tab["steepest"]["pivots"]) == (112.0, 61.0)
    _has("Steepest Edge spart Pivots", "40 Ressourcen und 40 Diensten", "102 Pivots und 743 580", "52 Pivots und 454 028", "0.51 der Pivots, 0.61 der Operationen", "112 gegen 61")


def test_help_bland_is_slow():
    c = _cmp("Bland ist langsam")
    assert (c["bland"].total_pivots, c["dantzig"].total_pivots) == (95, 41) and c["bland"].total_pivots / c["dantzig"].total_pivots == pytest.approx(2.3, abs=0.05)
    tab = ev.rule_table(_settings(C.PRESETS["Bland ist langsam"]))
    assert (tab["bland"]["pivots"], tab["dantzig"]["pivots"]) == (61.0, 17.0) and tab["bland"]["pivots"] / tab["dantzig"]["pivots"] == pytest.approx(3.6, abs=0.05)
    _has("Bland ist langsam", "95 Pivots", "Dantzig 41 (2.3-fach)", "61 gegen 17 Pivots (3.6-fach)")


def test_help_transport_stalls():
    a = _analysis("Transport: Stillstand")
    assert (a.res.total_pivots, a.res.degenerate_pivots) == (33, 3) and 100 * a.res.degenerate_pivots / a.res.total_pivots == pytest.approx(9.1, abs=0.05)
    row = next(r for r in ev.stall_table() if (r["kind"], r["m"], r["n"]) == ("transport", 4, 8))
    assert 100 * row["dantzig"] == pytest.approx(12.7, abs=0.05)
    _has("Transport: Stillstand", "4 Lagern und 8 Kunden", "33 Pivots, davon 3 Nullschritte (9 %)", "12.7 %")
    b = _analysis("Große Transportinstanz")
    s = _cmp("Große Transportinstanz")["steepest"]
    assert (b.res.total_pivots, b.res.degenerate_pivots, s.total_pivots, s.degenerate_pivots) == (85, 15, 35, 3) and 100 * b.res.degenerate_pivots / b.res.total_pivots == pytest.approx(17.6, abs=0.05)
    _has("Große Transportinstanz", "8 Lager, 12 Kunden", "85 Pivots, davon 15 Nullschritte (17.6 %)", "35 Pivots mit 3 Nullschritten")


def test_help_no_stalling_in_random():
    c = _cmp("Kein Stillstand im Zufall")
    assert c["dantzig"].total_pivots == 7 and all(r.degenerate_pivots == 0 and all(p.ties == 1 for p in r.pivots) for r in c.values())
    _has("Kein Stillstand im Zufall", "Dantzig 7 Pivots", "wie alle fünf Regeln")


def test_help_cost_over_size():
    c = _cmp("Aufwand über die Größe")
    assert (c["steepest"].total_pivots, c["dantzig"].total_pivots) == (24, 37)
    cv = ev.rule_curve(_settings(C.PRESETS["Aufwand über die Größe"]))
    row = {r: next(x for x in cv[r] if x["size"] == 20) for r in A.RULES}
    assert (row["steepest"]["pivots"], row["dantzig"]["pivots"]) == (22.5, 34.0)
    _has("Aufwand über die Größe", "24 Pivots gegen 37", "22.5 gegen 34")
