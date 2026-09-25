"""Plotly-Abbildungen: Regelvergleich (Pivots, Operationen), Pfade der Regeln im Zwei-Dienste-Bild, Zyklus-Tabelle, Schrittweiten, Stillstand je Instanztyp, Kurven über die Größe und Sweeps.
Die Geometrie hat feste Achsenbereiche aus der Instanz (ohne scaleanchor, damit nichts einfriert); Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import pandas as pd
import plotly.graph_objects as go

import piv_scenario as S

TEAL, ORANGE, RED, BLUE, GREY = "#2F6B65", "#e8a13a", "#d62728", "#1f4e9c", "#8a8f98"
LINE_COLORS = ["#4c78a8", "#7b3fbf", "#d95f9b", "#2ca02c", "#8c564b", "#17becf", "#bcbd22", "#e377c2"]
RULE_COLORS = {"dantzig": "#2F6B65", "greatest": "#e8a13a", "steepest": "#7b3fbf", "bland": "#d62728", "random": "#8a8f98"}
RULE_NAMES = {"dantzig": "Dantzig", "greatest": "Größter Zuwachs", "steepest": "Steepest Edge", "bland": "Bland", "random": "Zufall"}
RULES = ("dantzig", "greatest", "steepest", "bland", "random")


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _clip(poly, a, b_, c):
    """Halbebene a x + b y <= c auf ein konvexes Polygon anwenden (Sutherland-Hodgman)."""
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        fp, fq = a * p[0] + b_ * p[1] - c, a * q[0] + b_ * q[1] - c
        if fp <= 1e-12:
            out.append(p)
        if (fp < -1e-12 and fq > 1e-12) or (fp > 1e-12 and fq < -1e-12):
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def plot_box(vertices, status):
    """Sichtbarer Bereich [0, X] x [0, Y]: 1,35-fache größte Ecke, bei offenem Gebiet größer."""
    if not vertices:
        return 10.0, 10.0
    scale = 1.35 if status != "unbounded" else 2.4
    return max(1.0, scale * max(v[0] for v in vertices)), max(1.0, scale * max(v[1] for v in vertices))


def _line_in_box(a, b_, c, X, Y):
    pts = []
    if abs(b_) > 1e-12:
        for x in (0.0, X):
            y = (c - a * x) / b_
            if -1e-9 <= y <= Y + 1e-9:
                pts.append((x, y))
    if abs(a) > 1e-12:
        for y in (0.0, Y):
            x = (c - b_ * y) / a
            if -1e-9 <= x <= X + 1e-9:
                pts.append((x, y))
    uniq = []
    for p in pts:
        if not any(abs(p[0] - q[0]) < 1e-9 and abs(p[1] - q[1]) < 1e-9 for q in uniq):
            uniq.append(p)
    return uniq[:2]


def _region(inst, vertices, status):
    """Grundbild bei zwei Diensten: Zulässigkeitsmenge, Randgeraden, Ecken. Gibt (Abbildung, X, Y)."""
    A, b, _c = inst.arrays()
    X, Y = plot_box(vertices, status)
    poly = [(0.0, 0.0), (X, 0.0), (X, Y), (0.0, Y)]
    for i in range(inst.m):
        if inst.senses[i] == S.LE:
            poly = _clip(poly, A[i, 0], A[i, 1], b[i])
        elif inst.senses[i] == S.GE:
            poly = _clip(poly, -A[i, 0], -A[i, 1], -b[i])
    fig = go.Figure()
    if len(poly) >= 3:
        fig.add_trace(go.Scatter(x=[p[0] for p in poly] + [poly[0][0]], y=[p[1] for p in poly] + [poly[0][1]], mode="lines", fill="toself", fillcolor="rgba(47,107,101,0.13)", line=dict(width=0),
                                 name="zulässig", hoverinfo="skip"))
    for i in range(inst.m):
        seg = _line_in_box(A[i, 0], A[i, 1], b[i], X, Y)
        if len(seg) == 2:
            sense = {S.LE: "≤", S.GE: "≥", S.EQ: "="}[inst.senses[i]]
            fig.add_trace(go.Scatter(x=[seg[0][0], seg[1][0]], y=[seg[0][1], seg[1][1]], mode="lines", line=dict(color=LINE_COLORS[i % len(LINE_COLORS)], width=1.6, dash="dash" if inst.senses[i] != S.LE else "solid"),
                                     name=f"{inst.row_names[i]} {sense} {b[i]:g}", hoverinfo="skip"))
    if vertices:
        fig.add_trace(go.Scatter(x=[v[0] for v in vertices], y=[v[1] for v in vertices], mode="markers", marker=dict(size=8, color="white", line=dict(width=1.5, color=GREY)), name="Ecken", hoverinfo="skip"))
    fig.update_xaxes(title_text=inst.names[0], range=[-0.03 * X, X], zeroline=True)
    fig.update_yaxes(title_text=inst.names[1], range=[-0.03 * Y, Y], zeroline=True)
    return fig, X, Y


def path_points(res):
    """Punkte des Pfads (Start = Ursprung, dann die Basislösung nach jedem Pivot) mit Zulässigkeit."""
    return [(0.0, 0.0)] + [p.x for p in res.pivots], [res.phase1_pivots == 0] + [p.feasible for p in res.pivots]


def build_paths(inst, results, vertices):
    """Zwei Dienste: die Pfade aller Regeln in derselben Zeichnung (jede Regel eine Farbe); Nullschritte bleiben auf der Stelle stehen."""
    status = next(iter(results.values())).status
    fig, X, Y = _region(inst, vertices, status)
    offset = {rule: (i - 2) * 0.012 for i, rule in enumerate(RULES)}
    for rule, res in results.items():
        pts, ok = path_points(res)
        xs = [p[0] + offset[rule] * X for p in pts]
        ys = [p[1] + offset[rule] * Y for p in pts]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=RULE_COLORS[rule], width=2.6), marker=dict(size=6), name=f"{RULE_NAMES[rule]} ({res.total_pivots} Pivots)", hoverinfo="skip"))
    return _base(fig, 470, legend_y=-0.4)


def build_rule_bars(results):
    """Pivots und Gesamtoperationen (Pivots + Preisgebung) je Regel."""
    rules = [r for r in RULES if r in results]
    fig = go.Figure(go.Bar(x=[RULE_NAMES[r] for r in rules], y=[results[r].total_pivots for r in rules], marker_color=[RULE_COLORS[r] for r in rules], text=[results[r].total_pivots for r in rules], textposition="outside"))
    fig.update_yaxes(title_text="Pivots")
    return _base(fig, 300, legend_y=-0.3)


def build_ops_bars(results):
    rules = [r for r in RULES if r in results]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[RULE_NAMES[r] for r in rules], y=[results[r].flops for r in rules], name="Pivots (Zeilen eliminieren)", marker_color=TEAL))
    fig.add_trace(go.Bar(x=[RULE_NAMES[r] for r in rules], y=[results[r].price_flops for r in rules], name="Preisgebung (Spalte wählen)", marker_color=ORANGE))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Rechenoperationen")
    return _base(fig, 300, legend_y=-0.3)


def pivot_frame(inst, res):
    """Tabelle der Pivots mit eintretender/austretender Variable, Schrittweite, Zielwert und Basis; Zeilen, die zu einem Zyklus gehören, sind markiert."""
    names = list(res.col_names)
    rows = []
    first_repeat = res.total_pivots - res.cycle_len if res.cycled else None
    for p in res.pivots:
        rows.append({"Pivot": p.k, "Phase": p.phase, "tritt ein": names[p.enter], "verlässt": names[p.leave_var], "Schrittweite": p.ratio, "Zielwert": p.obj,
                     "Basis": ", ".join(names[j] for j in p.basis), "Zyklus": "◀" if first_repeat is not None and p.k > first_repeat else ""})
    df = pd.DataFrame(rows)
    return df, first_repeat


def style_pivots(df, first_repeat):
    def hi(_):
        css = pd.DataFrame("", index=df.index, columns=df.columns)
        if first_repeat is not None:
            for i in df.index:
                if df.loc[i, "Pivot"] > first_repeat:
                    css.loc[i, :] = "background-color: rgba(214,39,40,0.16)"
        return css
    return df.style.apply(hi, axis=None).format({"Schrittweite": "{:.3f}", "Zielwert": "{:.3f}"})


def build_step_sizes(res):
    """Schrittweite (kleinster Quotient) je Pivot; rot = Nullschritt (entartet)."""
    fig = go.Figure(go.Bar(x=list(range(1, res.total_pivots + 1)), y=[p.ratio for p in res.pivots], marker_color=[RED if p.degenerate else TEAL for p in res.pivots]))
    fig.update_xaxes(title_text="Pivot", dtick=1 if res.total_pivots <= 20 else None)
    fig.update_yaxes(title_text="Schrittweite (kleinster Quotient)")
    return _base(fig, 300, legend_y=-0.3)


def build_stall_bars(rows):
    """Anteil der Nullschritte je Instanztyp (Gruppen) und Regel (Balken)."""
    labels = [f"{r['kind']} {r['m']}x{r['n']}" for r in rows]
    fig = go.Figure()
    for rule in RULES:
        fig.add_trace(go.Bar(x=labels, y=[100 * r[rule] for r in rows], name=RULE_NAMES[rule], marker_color=RULE_COLORS[rule]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Nullschritte (% der Pivots)")
    return _base(fig, 340, legend_y=-0.3)


def build_rule_curves(curve, key, y_label):
    """Pivots bzw. Operationen über die Größe, eine Linie je Regel."""
    fig = go.Figure()
    for rule in RULES:
        rows = curve[rule]
        fig.add_trace(go.Scatter(x=[str(r["size"]) for r in rows], y=[r[key] for r in rows], mode="lines+markers", line=dict(color=RULE_COLORS[rule], width=2.5), name=RULE_NAMES[rule]))
    fig.update_xaxes(title_text="Größe (m = n)", type="category")
    fig.update_yaxes(title_text=y_label)
    return _base(fig, 340, legend_y=-0.3)


def build_sweep(rows, param_label, key, y_label, tick=None):
    """`rows` = [{"value": v, Regel: {Kennzahlen}}]: je Regel Median als Linie, 10. bis 90. Perzentil als Band."""
    xs = [tick(r["value"]) if tick else str(r["value"]) for r in rows]
    fig = go.Figure()
    for rule in RULES:
        ys = [None if r[rule][key] != r[rule][key] else r[rule][key] for r in rows]
        lo = [None if r[rule][f"{key}_lo"] != r[rule][f"{key}_lo"] else r[rule][f"{key}_lo"] for r in rows]
        hi = [None if r[rule][f"{key}_hi"] != r[rule][f"{key}_hi"] else r[rule][f"{key}_hi"] for r in rows]
        color = RULE_COLORS[rule]
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        if all(v is not None for v in lo + hi):
            fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], mode="lines", fill="toself", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.10)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=color, width=2.5), name=RULE_NAMES[rule], connectgaps=False))
    fig.update_xaxes(title_text=param_label, type="category")
    fig.update_yaxes(title_text=y_label)
    return _base(fig, 360, legend_y=-0.3)

