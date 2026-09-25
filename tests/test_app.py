"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt für jeden Instanztyp und jede Regel, Randwerte, Würfel, Permalink-Grenzen, Instanzwechsel, Experimente und Sweeps auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import piv_algorithm as A
import piv_constants as C
import piv_scenario as S

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(step=1, **state):
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if step != 1:
        at.select_slider(key="piv_step").set_value(step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m for m in at.metric if m.label.startswith(label))


def test_default_run_shows_the_textbook_numbers():
    at = _run()
    _ok(at)
    assert {"Pivots", "Nullschritte", "Operationen", "Ergebnis"} <= {m.label for m in at.metric}
    assert _metric(at, "Pivots").value == "2" and _metric(at, "Operationen").value == "84" and _metric(at, "Ergebnis").value == "36.00" and _metric(at, "Nullschritte").value == "0"
    assert at.dataframe and at.get("plotly_chart")


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    ss = at.session_state
    assert (ss["kind_select"], ss["piv_step"], ss["rule_select"], ss["ratio_select"]) == (p["kind"], p["step"], p["rule"], p["ratio"])
    if p["kind"] == "transport":
        assert (ss["k_slider"], ss["l_slider"]) == (p["k"], p["l"])
    elif p["kind"] not in S.FIXTURE_KINDS:
        assert (ss["m_slider"], ss["n_slider"]) == (p["m"], p["n"])


@pytest.mark.parametrize("step", [1, 2, 3, 4])
@pytest.mark.parametrize("kind", list(S.KINDS))
def test_every_step_runs_for_every_kind(step, kind):
    at = _run(kind_select=kind, m_slider=8, n_slider=5, k_slider=3, l_slider=4, piv_step=step)
    _ok(at)
    assert at.session_state["piv_step"] == step


@pytest.mark.parametrize("rule", A.RULES)
@pytest.mark.parametrize("ratio", A.RATIO_RULES)
def test_every_rule_and_ratio_runs_on_beale_and_a_random_instance(rule, ratio):
    for kind in ("beale", "mixed"):
        for step in (1, 2, 3):
            _ok(_run(step=step, kind_select=kind, m_slider=10, n_slider=8, rule_select=rule, ratio_select=ratio))


def test_beale_cycle_is_reported_and_the_rescues_are_shown():
    at = _run(step=2, kind_select="beale")
    _ok(at)
    assert any("**Zyklus:**" in e.value and "alle 6 Pivots" in e.value for e in at.error)
    assert len(at.dataframe) == 2
    for kw, pivots in ((dict(rule_select="bland"), 6), (dict(ratio_select="lexicographic"), 2)):
        at2 = _run(step=2, kind_select="beale", **kw)
        _ok(at2)
        assert any("**Terminiert:**" in s.value and f"{pivots} Pivots" in s.value for s in at2.success)


def test_pivot_comparison_table_has_all_five_rules():
    at = _run(kind_select="mixed", m_slider=10, n_slider=10)
    _ok(at)
    table = at.dataframe[0].value
    assert list(table["Regel"]) == ["Dantzig", "Größter Zuwachs", "Steepest Edge", "Bland", "Zufall"] and (table["Pivots"] > 0).all()


def test_two_variable_instance_draws_the_paths():
    at = _run(kind_select="random", m_slider=6, n_slider=2, seed_input=3)
    _ok(at)
    assert len(at.get("plotly_chart")) == 3


@pytest.mark.parametrize("kw", [
    dict(kind_select="random", m_slider=C.M_MIN, n_slider=C.N_MIN), dict(kind_select="random", m_slider=C.M_MAX, n_slider=C.N_MAX), dict(kind_select="mixed", m_slider=C.M_MAX, n_slider=C.N_MIN),
    dict(kind_select="degenerate", m_slider=C.M_MIN, n_slider=C.N_MAX), dict(kind_select="transport", k_slider=C.K_MIN, l_slider=C.L_MIN), dict(kind_select="transport", k_slider=C.K_MAX, l_slider=C.L_MAX),
    dict(kind_select="random", density_select=C.DENSITY_OPTIONS[0]), dict(kind_select="random", density_select=C.DENSITY_OPTIONS[-1]),
])
def test_extreme_settings_run(kw):
    for step in (1, 2, 3, 4):
        _ok(_run(step=step, **kw))


def test_dice_button_changes_the_seed():
    at = _run(kind_select="random")
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_clamped_and_invalid_choices_fall_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(m="999", n="1", k="1", l="99", density="0.42", step="9", kind="nope", rule="nope", ratio="nope", seed="-4").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["m_slider"], ss["n_slider"], ss["k_slider"], ss["l_slider"], ss["density_select"], ss["piv_step"], ss["kind_select"], ss["rule_select"], ss["ratio_select"], ss["seed_input"]) == (
        C.M_MAX, C.N_MIN, C.K_MIN, C.L_MAX, C.DEFAULT_DENSITY, 1, "textbook", C.DEFAULT_RULE, C.DEFAULT_RATIO, 0)


def test_permalink_accepts_valid_values():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(kind="transport", k="5", l="9", seed="7", rule="steepest", ratio="lexicographic", step="3").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["kind_select"], ss["k_slider"], ss["l_slider"], ss["seed_input"], ss["rule_select"], ss["ratio_select"], ss["piv_step"]) == ("transport", 5, 9, 7, "steepest", "lexicographic", 3)


def test_sidebar_shows_the_controls_that_belong_to_the_instance():
    fixed = _run()
    assert not any(w.key in ("m_widget", "k_widget") for w in fixed.slider) and not any(n.key == "seed_widget" for n in fixed.number_input)
    assert any(w.key == "rule_select" for w in fixed.selectbox) and any(r.key == "ratio_select" for r in fixed.radio)
    rnd = _run(kind_select="random")
    assert any(w.key == "m_widget" for w in rnd.slider) and any(w.key == "density_widget" for w in rnd.select_slider) and not any(w.key == "k_widget" for w in rnd.slider)
    tr = _run(kind_select="transport")
    assert any(w.key == "k_widget" for w in tr.slider) and any(w.key == "l_widget" for w in tr.slider) and not any(w.key == "m_widget" for w in tr.slider)


def test_changing_the_instance_and_rule_while_on_later_steps_does_not_crash():
    for step in (2, 3, 4):
        at = _run(step=step, kind_select="random")
        _ok(at)
        for kw in (dict(kind_select="beale"), dict(kind_select="transport"), dict(rule_select="steepest"), dict(ratio_select="lexicographic"), dict(kind_select="infeasible"), dict(kind_select="unbounded"),
                   dict(kind_select="mixed", m_slider=6, n_slider=2)):
            for k, v in kw.items():
                at.session_state[k] = v
            at.run()
            _ok(at)


def test_fixed_instance_table_and_curve_and_stall_and_cycles_on_demand():
    at = _run(kind_select="mixed", m_slider=8, n_slider=8)
    next(b for b in at.button if b.key == "table_start").click().run()
    _ok(at)
    assert any(c.value.startswith("Median über 5 feste Instanzen") for c in at.caption)
    at = _run(step=4, kind_select="mixed", m_slider=8, n_slider=8)
    next(b for b in at.button if b.key == "curve_start").click().run()
    _ok(at)
    assert any("Größte Instanz" in c.value for c in at.caption)
    at.radio(key="curve_metric").set_value("flops").run()
    _ok(at)
    at2 = _run()
    next(b for b in at2.button if b.key == "stall_start").click().run()
    _ok(at2)
    next(b for b in at2.button if b.key == "cycle_start").click().run()
    _ok(at2)
    assert _metric(at2, "Zyklen (erzeugt)").value == "0" and _metric(at2, "Zyklen (Beale-artig)").value == "0"


@pytest.mark.parametrize("param", ["size", "density", "kind"])
@pytest.mark.parametrize("metric", ["pivots", "flops", "degenerate_share", "price_share"])
def test_sweeps_run_on_demand_for_every_metric(param, metric):
    at = _run(kind_select="random", m_slider=8, n_slider=8, sweep_metric=metric)
    at.selectbox(key="sweep_select").set_value(param).run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_footer_limits_and_literature_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Bland, R. G. (1977)" in m.value and "Forrest, J. J., & Goldfarb, D. (1992)" in m.value and "Beale, E. M. L. (1955)" in m.value for m in at.markdown)
