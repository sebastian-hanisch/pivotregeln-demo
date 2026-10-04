"""Pivotregeln und Entartung – wie wählt der Simplex, und was, wenn er stillsteht? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zweites Stück der Lineare-Programmierung-Reihe der "Konzepte"-Reihe: Der Tableau-Simplex hat in jedem Pivot die Wahl, welche Spalte eintritt (Pivotregel: Dantzig, größter Zuwachs, Steepest Edge, Bland, Zufall) und
welche Zeile bei einem Gleichstand austritt (Quotiententest: Index oder lexikographisch). Die Demo vergleicht die Regeln, zeigt Beales Zyklus und misst, wie oft der Simplex stillsteht.

Lauffähig mit: streamlit run app.py
"""

from dataclasses import replace

import pandas as pd
import streamlit as st

import piv_algorithm as A
import piv_constants as C
import piv_evaluation as ev
import piv_scenario as S
from piv_evaluation import SWEEP_LABELS, Settings, analyse, compare_rules
from piv_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    store_from_widget,
    sync_query_params,
)
from piv_visualization import (
    build_ops_bars,
    build_paths,
    build_rule_bars,
    build_rule_curves,
    build_stall_bars,
    build_step_sizes,
    build_sweep,
    pivot_frame,
    style_pivots,
)

st.set_page_config(page_title="Pivotregeln und Entartung – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _rule_table(base):
    return ev.rule_table(base)


@st.cache_data(show_spinner=False)
def _curve(base):
    return ev.rule_curve(base)


@st.cache_data(show_spinner=False)
def _stall(ratio):
    return ev.stall_table(ratio=ratio)


@st.cache_data(show_spinner=False)
def _cycles():
    return ev.cycle_search(), ev.beale_like_search()


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return ev.sweep(param, base)


def num(x, digits=2):
    return f"{x:.{digits}f}"


def thousands(x):
    return f"{int(round(x)):,}".replace(",", " ")


STATUS_TEXT = {"optimal": "Optimum", "infeasible": "unzulässig", "unbounded": "unbeschränkt", "cycled": "kreist (Basis wiederholt sich)", "limit": "Pivot-Grenze"}

st.title("🧭 Pivotregeln und Entartung")
st.markdown(
    """
**Zweites Stück der Lineare-Programmierung-Reihe.** Im [Tableau-Simplex](https://github.com/sebastian-hanisch/tableau-simplex-demo) war die Wahl in jedem Pivot festgelegt: die Spalte mit den kleinsten reduzierten Kosten tritt ein (**Dantzig-Regel**).
Andere Regeln wählen anders: die mit dem **größten Zuwachs** des Zielwerts, die mit dem besten Verhältnis von Verbesserung zu **Kantenlänge** (**Steepest Edge**), die mit dem **kleinsten Index** (**Bland**) oder eine **zufällige**.
Dazu kommt die Zeilenwahl bei einem Gleichstand im Quotiententest: nach **kleinstem Index** oder **lexikographisch**. Bei einem Gleichstand steht der Simplex an einer entarteten Ecke, und dort kann er stillstehen (Nullschritte) oder sogar
**kreisen**. Vier Fragen, alle gemessen: **(1) Regelvergleich** - wie viele Pivots und wie viel Rechnung braucht jede Regel? **(2) Beales Zyklus** - kreist der Simplex wirklich, und was verhindert es? **(3) Stillstand** - wie oft steht er still?
**(4) Aufwand** - wie wachsen die Unterschiede mit der Größe?
"""
)
st.caption("Kind der Wurzel der Reihe. Folgestücke (Klee-Minty und der schlimmste Fall, Revised Simplex, Dualität, Innere Punkte, PDLP) gibt es als eigene Demos der Reihe.")

with st.expander("So funktionieren die Regeln", expanded=True):
    st.markdown(
        """
1. **Dantzig:** die Spalte mit dem kleinsten (am stärksten negativen) r. Billig zu wählen, aber blind für die Länge des Schritts.
2. **Größter Zuwachs:** für jede Kandidaten-Spalte den Quotiententest probeweise rechnen, die Spalte mit dem größten Zuwachs des Zielwerts (r mal Schrittweite) wählen. Kostet je Pivot eine Rechnung über alle Kandidaten.
3. **Steepest Edge:** r geteilt durch die Länge der Kante, entlang der sich die Basislösung bewegt (Wurzel aus 1 + Summe der Quadrate der Spalte im Tableau): die Richtung mit dem steilsten Anstieg je zurückgelegter Strecke.
4. **Bland:** die Spalte mit dem kleinsten Index unter allen Kandidaten. Braucht keine Rechnung und kann nach einem Satz von Bland (1977) nicht kreisen, bewegt sich aber oft in sehr kleinen Schritten.
5. **Zufall:** ein zufälliger Kandidat (fester Seed).
6. **Quotiententest:** bei einem Gleichstand verlässt die Zeile mit dem **kleinsten Index** der Basisvariable die Basis (einfach), oder die **lexikographisch kleinste** Zeile (Vergleich der Zeilen geteilt durch das Pivotelement, beginnend mit der rechten Seite): beweisbar zyklenfrei.
7. **Zyklus:** kehrt bei einer deterministischen Regel eine Basis zurück, wiederholt sich der Lauf für immer; die Demo bricht dann ab und zeigt den Zyklus.
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:4], preset_names[4:7], preset_names[7:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

ss = st.session_state
with st.sidebar:
    st.header("⚙️ Einstellungen")
    kind = st.selectbox("Instanz", options=list(S.KINDS), format_func=lambda v: S.KIND_LABELS[v], key="kind_select",
                        help="Lehrbuchbeispiel, Beale und die Sonderfälle sind fest; Zufall, Mischung, Gleichstände und Transport sind regelbar.")
    random_kind = kind not in S.FIXTURE_KINDS
    if random_kind and kind == "transport":
        m = st.slider("Lager", *bounds("k_slider"), value=int(ss["k_slider"]), key="k_widget", on_change=store_from_widget, args=("k_slider",), help="Lager (Angebot). Angebot = Nachfrage: das macht die Instanz entartet.")
        n = st.slider("Kunden", *bounds("l_slider"), value=int(ss["l_slider"]), key="l_widget", on_change=store_from_widget, args=("l_slider",), help="Kunden (Nachfrage); Lager mal Kunden sind die Variablen.")
        density = C.DEFAULT_DENSITY
    elif random_kind:
        m = st.slider("Ressourcen m", *bounds("m_slider"), value=int(ss["m_slider"]), key="m_widget", on_change=store_from_widget, args=("m_slider",), help="Zahl der Bedingungen (knappe Ressourcen).")
        n = st.slider("Dienste n", *bounds("n_slider"), value=int(ss["n_slider"]), key="n_widget", on_change=store_from_widget, args=("n_slider",),
                      help="Zahl der Entscheidungsvariablen. Bei n = 2 gibt es die Zeichnung der Pfade.")
        density = st.select_slider("Dichte der Matrix", options=list(C.DENSITY_OPTIONS), value=float(ss["density_select"]), key="density_widget", on_change=store_from_widget, args=("density_select",),
                                   help="Anteil der Dienste, die eine Ressource verbrauchen.")
    else:
        m, n, density = C.DEFAULT_M, C.DEFAULT_N, C.DEFAULT_DENSITY
    if random_kind:
        seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), value=int(ss["seed_input"]), key="seed_widget", step=1, on_change=store_from_widget, args=("seed_input",))
        st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    else:
        seed = C.DEFAULT_SEED
    rule = st.selectbox("Pivotregel", options=list(C.RULE_LABELS), format_func=lambda v: C.RULE_LABELS[v], key="rule_select",
                        help="Wirkt auf Schritt 2 und 3 und die Kennzahlen des gewählten Laufs; der Regelvergleich in Schritt 1 und die Kurven in Schritt 4 zeigen immer alle fünf.")
    ratio = st.radio("Quotiententest bei Gleichstand", options=list(C.RATIO_LABELS), format_func=lambda v: C.RATIO_LABELS[v], key="ratio_select",
                     help="Wirkt bei Gleichständen im Quotiententest, also vor allem bei Beale und beim Transportproblem.")

sync_query_params({"kind_select": kind, "m_slider": int(ss["m_slider"]), "n_slider": int(ss["n_slider"]), "k_slider": int(ss["k_slider"]), "l_slider": int(ss["l_slider"]), "density_select": float(ss["density_select"]),
                   "seed_input": int(ss["seed_input"]), "rule_select": rule, "ratio_select": ratio, "piv_step": int(ss["piv_step"])})

settings = Settings(kind, int(m), int(n), float(density), int(seed), rule, ratio)
with st.spinner("Rechne..."):
    a = analyse(settings)
    cmp = compare_rules(settings)
inst, res = a.inst, a.res
names = list(res.col_names)

# --- In Aktion ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Regeln im Vergleich")
step = st.select_slider("Schritt", options=list(C.STEPS), key="piv_step", format_func=lambda s: C.STEPS[s])

if step == 1:
    rows = []
    d = cmp["dantzig"]
    for r in A.RULES:
        x = cmp[r]
        ok = x.status == "optimal"
        rows.append({"Regel": C.RULE_SHORT[r], "Ergebnis": STATUS_TEXT[x.status] + (f" {num(x.obj)}" if ok else ""), "Pivots": x.total_pivots, "Nullschritte": x.degenerate_pivots, "Operationen Pivots": thousands(x.flops),
                     "Operationen Preisgebung": thousands(x.price_flops), "Gesamt": thousands(x.total_flops),
                     "Gesamt / Dantzig": (f"{x.total_flops / d.total_flops:.2f}" if ok and d.status == "optimal" and d.total_flops else "-")})
    st.markdown(f"**Alle fünf Regeln auf derselben Instanz** ({S.KIND_LABELS[kind].split(' (')[0]}, {inst.m} Ressourcen, {inst.n} Variablen, Quotiententest: {C.RATIO_LABELS[ratio].split(' (')[0].lower()}):")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_rule_bars(cmp), width="stretch", key="s1_pivots")
    with c2:
        st.plotly_chart(build_ops_bars(cmp), width="stretch", key="s1_ops")
    st.caption("Links: Pivots je Regel. Rechts: Rechenoperationen im dichten Tableau, geteilt in die Pivots selbst und die Preisgebung (Auswahl der eintretenden Spalte); Bland, Dantzig und Zufall brauchen keine Preisgebung.")
    if inst.n == 2 and inst.m >= 1 and a.vertices:
        st.plotly_chart(build_paths(inst, cmp, a.vertices), width="stretch", key="s1_paths")
        st.caption("Pfade der Regeln von Ecke zu Ecke (leicht gegeneinander versetzt gezeichnet, damit gemeinsame Kanten sichtbar bleiben).")
    if st.button("Über 5 feste Instanzen vergleichen", key="table_start"):
        ss["table_done"] = ss.get("table_done", set()) | {replace(settings, seed=0)}
    if replace(settings, seed=0) in ss.get("table_done", set()):
        with st.spinner("Rechne..."):
            tab = _rule_table(replace(settings, seed=0))
        st.dataframe(pd.DataFrame([{"Regel": C.RULE_SHORT[r], "Pivots (Median)": tab[r]["pivots"], "Pivots / Dantzig": round(tab[r]["pivots_vs_dantzig"], 2), "Operationen (Median)": thousands(tab[r]["flops"]),
                                    "Operationen / Dantzig": round(tab[r]["flops_vs_dantzig"], 2), "kreisend": tab[r]["cycled"]} for r in A.RULES]), hide_index=True, width="stretch")
        st.caption("Median über 5 feste Instanzen (Seeds 100000–100004) desselben Typs und derselben Größe; die Fixtures haben nur eine Instanz.")
elif step == 2:
    if res.status == "cycled":
        first = res.total_pivots - res.cycle_len
        st.error(f"**Zyklus:** Nach Pivot {res.total_pivots} ist eine Basis erreicht, die schon {'am Start' if first == 0 else f'nach Pivot {first}'} vorkam - der Lauf wiederholt sich alle {res.cycle_len} Pivots für immer "
                 f"(Zielwert bewegt sich nicht, {res.degenerate_pivots} Nullschritte). Die Demo bricht hier ab.")
    elif res.status == "optimal":
        st.success(f"**Terminiert:** {res.total_pivots} Pivots bis zum Optimum {num(res.obj)}; keine Basis kam zweimal vor" + (f" (Nullschritte: {res.degenerate_pivots})." if res.degenerate_pivots else "."))
    else:
        st.info(f"Ergebnis: {STATUS_TEXT[res.status]} nach {res.total_pivots} Pivots.")
    df, first_repeat = pivot_frame(inst, res)
    if len(df):
        st.dataframe(style_pivots(df, first_repeat), hide_index=True, width="stretch", height=min(600, 40 + 35 * len(df)))
        st.caption("Basis = die Variablen in der Basis nach dem Pivot (x = Dienste, s = Schlupf, e = Überschuss, a = künstlich). Rot markiert: Pivots, die eine schon gesehene Basis wiederholen.")
    else:
        st.info("Diese Instanz ist ohne Pivot fertig.")
    combos = []
    for rl in A.RULES:
        for rt in A.RATIO_RULES:
            x = A.tableau_simplex(inst, rule=rl, ratio=rt, seed="0")
            combos.append({"Regel": C.RULE_SHORT[rl], "Quotiententest": C.RATIO_LABELS[rt].split(" (")[0], "Ergebnis": STATUS_TEXT[x.status] + (f" {num(x.obj)}" if x.status == "optimal" else ""), "Pivots": x.total_pivots,
                           "Nullschritte": x.degenerate_pivots})
    st.markdown("**Alle Kombinationen auf dieser Instanz:**")
    st.dataframe(pd.DataFrame(combos), hide_index=True, width="stretch")
elif step == 3:
    if res.status == "optimal" or res.status == "cycled":
        st.markdown(f"**{res.degenerate_pivots} von {res.total_pivots} Pivots** ({(100 * res.degenerate_pivots / res.total_pivots) if res.total_pivots else 0:.1f} %) sind **Nullschritte**: die Basis wechselt, die Ecke nicht, der Zielwert bleibt gleich. "
                    f"Gleichstände im Quotiententest gab es in {sum(p.ties > 1 for p in res.pivots)} Pivots" + (f"; die längste Stillstandsphase dauert {max(res.stall_runs)} Pivots." if res.stall_runs else "."))
        if res.total_pivots:
            st.plotly_chart(build_step_sizes(res), width="stretch", key="s3_steps")
            st.caption("Schrittweite je Pivot mit der gewählten Regel; rot = Nullschritt.")
    else:
        st.info(f"Ergebnis: {STATUS_TEXT[res.status]}; kein Optimum, also kein Stillstand zu messen.")
    zs = pd.DataFrame([{"Regel": C.RULE_SHORT[r], "Pivots": cmp[r].total_pivots, "Nullschritte": cmp[r].degenerate_pivots, "Anteil": f"{100 * cmp[r].degenerate_pivots / cmp[r].total_pivots:.1f} %" if cmp[r].total_pivots else "-",
                       "längster Stillstand": max(cmp[r].stall_runs) if cmp[r].stall_runs else 0} for r in A.RULES])
    st.markdown("**Nullschritte je Regel auf dieser Instanz:**")
    st.dataframe(zs, hide_index=True, width="stretch")
else:
    st.markdown("Pivots und Rechenoperationen über die Größe (m = n bzw. Lager und Kunden) für alle fünf Regeln, 30 Instanzen je Größe und dieselben Instanzen für alle Regeln (🔬 auf Abruf).")
    curve_base = replace(settings, seed=0, rule="dantzig") if random_kind else replace(settings, kind="mixed", m=10, n=10, seed=0, rule="dantzig")
    if not random_kind:
        st.caption("Die festen Instanzen haben keine Größe; die Kurve zeigt die Mischinstanzen.")
    if st.button("Kurven über die Größe berechnen (dauert einige Sekunden)", key="curve_start"):
        ss["curve_done"] = ss.get("curve_done", set()) | {curve_base}
    if curve_base in ss.get("curve_done", set()):
        with st.spinner("Rechne..."):
            cv = _curve(curve_base)
        metric = st.radio("Kennzahl", ["pivots", "flops"], format_func=lambda v: {"pivots": "Pivots", "flops": "Gesamtoperationen"}[v], horizontal=True, key="curve_metric")
        st.plotly_chart(build_rule_curves(cv, metric, "Pivots (Median)" if metric == "pivots" else "Gesamtoperationen (Median)"), width="stretch", key=f"curve_chart_{metric}")
        big = {r: cv[r][-1] for r in A.RULES}
        st.caption(f"Größte Instanz ({big['dantzig']['m']} x {big['dantzig']['n']}): Dantzig {big['dantzig']['pivots']:.1f} Pivots, Größter Zuwachs {big['greatest']['pivots']:.1f}, Steepest Edge {big['steepest']['pivots']:.1f}, "
                   f"Bland {big['bland']['pivots']:.1f}, Zufall {big['random']['pivots']:.1f}.")

st.markdown("---")

# --- Kennzahlen --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ⚙️ Der gewählte Lauf")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Pivots", str(res.total_pivots), delta=C.RULE_SHORT[rule], delta_color="off")
m2.metric("Nullschritte", str(res.degenerate_pivots), delta=f"{sum(p.ties > 1 for p in res.pivots)} Gleichstände", delta_color="off")
m3.metric("Operationen", thousands(res.total_flops), delta=f"Preisgeb. {thousands(res.price_flops)}", delta_color="off")
m4.metric("Ergebnis", num(res.obj) if res.status == "optimal" else "-", delta=STATUS_TEXT[res.status].split(" (")[0], delta_color="off")

st.markdown("---")

# --- Experimente auf Abruf ---------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Stillstand je Instanztyp")
st.caption("Anteil der Nullschritte an allen Pivots, 30 Instanzen je Typ und Größe, alle fünf Regeln, mit dem gewählten Quotiententest.")
if st.button("Stillstand-Tabelle berechnen (dauert einige Sekunden)", key="stall_start"):
    ss["stall_done"] = ss.get("stall_done", set()) | {ratio}
if ratio in ss.get("stall_done", set()):
    with st.spinner("Rechne..."):
        st_rows = _stall(ratio)
    st.plotly_chart(build_stall_bars(st_rows), width="stretch", key=f"stall_chart_{ratio}")
    st.caption("Nullschritte gibt es nur im Transportproblem (Angebot gleich Nachfrage); Zufalls-, Misch- und Gleichstands-Instanzen laufen ohne Stillstand.")
st.markdown("---")

st.subheader("🔍 Kreist Dantzig auf erzeugten Instanzen?")
st.caption("Dantzig mit Indexrest im Quotiententest auf 400 erzeugten Instanzen je Typ (Zufall, Mischung, Gleichstände, Transport in zwei Größen) und auf 5000 zufälligen kleinen LPs im Stil von Beale - Beales Zyklus selbst ist ein konstruiertes Gegenbeispiel.")
if st.button("Zyklen suchen (dauert einige Sekunden)", key="cycle_start"):
    ss["cycle_done"] = True
if ss.get("cycle_done"):
    with st.spinner("Rechne..."):
        found, (b_cyc, b_tot, b_opt) = _cycles()
    q1, q2 = st.columns(2)
    q1.metric("Zyklen (erzeugt)", str(sum(v[0] for v in found.values())), delta=f"in {sum(v[1] for v in found.values())} Instanzen", delta_color="off")
    q2.metric("Zyklen (Beale-artig)", str(b_cyc), delta=f"in {b_tot} kleinen LPs", delta_color="off")
    st.caption(f"Kein einziger Zyklus, weder auf den erzeugten Instanzen (Zufall, Mischung, Gleichstände, Transport) noch auf {b_tot} zufälligen kleinen LPs im Stil von Beale ({b_opt} davon mit Optimum, der Rest unbeschränkt): "
               "kreisen kann der Simplex, aber zufällige Instanzen stoßen ihn nicht in einen Zyklus.")
st.markdown("---")

st.subheader("📐 Sweeps")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda v: SWEEP_LABELS[v], key="sweep_select")
metric_opts = {"pivots": "Pivots", "flops": "Gesamtoperationen", "degenerate_share": "Nullschritte", "price_share": "Anteil Preisgebung"}
if ss.get("sweep_metric") not in metric_opts:
    ss.pop("sweep_metric", None)
metric = st.radio("Kennzahl", options=list(metric_opts), format_func=lambda v: metric_opts[v], key="sweep_metric", horizontal=True)
sweep_base = replace(settings, seed=0, rule="dantzig")
if st.button("Sweep über 5 feste Instanzen berechnen (kann einige Sekunden dauern)", key="sweep_start"):
    ss["sweep_done"] = ss.get("sweep_done", set()) | {(sweep_param, sweep_base)}
if (sweep_param, sweep_base) in ss.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Instanzen..."):
        rows_w = _sweep(sweep_param, sweep_base)
    tick = (lambda v: v.capitalize()) if sweep_param == "kind" else None
    st.plotly_chart(build_sweep(rows_w, SWEEP_LABELS[sweep_param], metric, metric_opts[metric], tick=tick), width="stretch", key="sweep_chart")
    st.caption("Median über 5 feste Instanzen (Seeds 100000–100004), Band = 10. bis 90. Perzentil, eine Linie je Regel; Quotiententest wie in der Seitenleiste.")
st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die klügere Regel gewinnt immer** | Meist, nicht überall. Bei m = n = 40 (Median über 30 Instanzen) braucht Steepest Edge 15.5 statt 21.5 Pivots auf Zufallsinstanzen, 51 statt 95.5 auf Mischinstanzen und 32 statt 53.5 im Transportproblem (6 Lager, 10 Kunden); auf Mischinstanzen kostet das mit Preisgebung 0.63 der Operationen von Dantzig. Der größte Zuwachs ist fast so gut (66 Pivots auf Mischinstanzen), Zufall und Bland sind deutlich schlechter. Auf dem Lehrbuchbeispiel kippt es: 84 Operationen für Dantzig, 102 für den größten Zuwachs, 108 für Steepest Edge. | Folgestück: Revised Simplex (dort müssen die Steepest-Edge-Längen mitgeführt werden) |
| **Bland ist die sichere Wahl** | Sicher gegen Zyklen, aber teuer: Pivots im Median gegenüber Dantzig bei m = n = 40 das 3.7-fache auf Zufallsinstanzen (80 gegen 21.5), das 2.4-fache auf Mischinstanzen (230 gegen 95.5) und das 1.5-fache im Transportproblem (81.5 gegen 53.5). | - |
| **Der Simplex kreist** | Nur auf konstruierten Instanzen: Beales Beispiel kreist mit Dantzig und Indexrest nach 6 Pivots. In 2000 erzeugten Instanzen (Zufall, Mischung, Gleichstände, Transport) und 5000 zufälligen kleinen LPs im Stil von Beale trat kein einziger Zyklus auf. | Folgestück: der schlimmste Fall |
| **Stillstand ist häufig** | Nur bei struktureller Entartung: im Transportproblem (Angebot gleich Nachfrage) sind je nach Regel 10.7 bis 19.6 % der Pivots Nullschritte (Dantzig 12.7 % bei 4 Lagern und 8 Kunden, 13.9 % bei 8 und 12); in Zufalls-, Misch- und Gleichstands-Instanzen sind es 0 %, obwohl es dort Gleichstände im Quotiententest gibt. | - |
| **Der lexikographische Test kostet etwas** | Kaum: im Transportproblem (4 Lager, 8 Kunden) 31 statt 33 Pivots im Median, der Vergleichsaufwand macht 0.04 % der Operationen aus. Er verhindert Zyklen beweisbar, ändert aber ohne Gleichstand nichts. | - |
| **Preisgebung ist ein Näherungsmaß** | Gezählt werden Multiplikationen und Vergleiche im dichten Tableau, wo alle Spalten vorliegen; keine Laufzeit, keine Referenzrahmen (Devex), keine Störung nach Wolfe. | Folgestück: Revised Simplex |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Reduzierte Kosten und Kandidaten.** Für die Basis $B$ ist $r_j = c_B^\top B^{-1} a_j - c_j$; Kandidaten für den Eintritt sind die Nichtbasisspalten mit $r_j < 0$. Mit $t_j = B^{-1} a_j$ ist $d_j = (e_j, -t_j)$ die Kante, entlang der sich die Basislösung
bewegt, wenn $j$ eintritt. Der Quotiententest $\theta_j = \min_{i:\,t_{ij} > 0} \beta_i / t_{ij}$ (mit $\beta = B^{-1} b$) ist die Schrittweite.

**Regeln.** Dantzig: $\arg\min_j r_j$. Größter Zuwachs: $\arg\max_j (-r_j)\,\theta_j$. Steepest Edge: $\arg\min_j r_j / \lVert d_j \rVert$ mit $\lVert d_j \rVert^2 = 1 + \lVert t_j \rVert^2$. Bland: kleinster Index unter den Kandidaten.

**Entartung und Zyklen.** Sind mehrere $\beta_i / t_{ij}$ gleich klein, hat die nächste Basislösung eine Basisvariable mit Wert 0 (entartete Ecke); ein Pivot mit $\theta = 0$ ändert die Ecke nicht. Eine deterministische Regel, die eine Basis wieder besucht, wiederholt sich für immer (Zyklus).
Bland (1977) verhindert das über feste Indizes, der lexikographische Quotiententest (Dantzig, Orden, Wolfe 1955) über die lexikographische Ordnung der Zeilen $(\beta_i, B^{-1}_i)$.

**Aufwand.** Ein Pivot des dichten Tableaus kostet $(N + 1) + 2 m (N + 1)$ Operationen; die Preisgebung kostet bei größtem Zuwachs $2 m$, bei Steepest Edge $2 m + 2$ Operationen je Kandidat (Näherungsmaß, keine Laufzeit).

**Literatur.** Bland, R. G. (1977). *New finite pivoting rules for the simplex method.* Mathematics of Operations Research 2(2), 103-107. Forrest, J. J., & Goldfarb, D. (1992). *Steepest-edge simplex algorithms for linear programming.* Mathematical Programming 57, 341-374.
Beale, E. M. L. (1955). *Cycling in the dual simplex algorithm.* Naval Research Logistics Quarterly 2(4), 269-275 (Beispiel des Zyklus, hier in der Maximierungsform). Dantzig, G. B., Orden, A., & Wolfe, P. (1955). *The generalized simplex method for minimizing a linear form under linear inequality restraints.* Pacific Journal of Mathematics 5, 183-195.

Implementiert in `piv_algorithm.py` (Simplex mit fünf Regeln, zwei Quotienten-Regeln, Zykluserkennung), `piv_scenario.py` (Instanzen inkl. Beale und Transport), `piv_evaluation.py` (Regelvergleich, Stillstand, Kurven).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Lineare Programmierung: vom Tableau zum Crossover](https://sebastianhanisch.net/konzepte-lineare-programmierung.html)."
)
