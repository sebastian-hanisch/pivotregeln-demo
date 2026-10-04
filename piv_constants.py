"""Konstanten der Demo Pivotregeln und Entartung: Regler-Bereiche, feste Instanzen, Presets (Werte aus der Vormessung)."""
M_MIN, M_MAX, DEFAULT_M = 2, 60, 10
N_MIN, N_MAX, DEFAULT_N = 2, 60, 10
K_MIN, K_MAX, DEFAULT_K = 2, 8, 4                    # Lager im Transportproblem
L_MIN, L_MAX, DEFAULT_L = 2, 12, 8                   # Kunden im Transportproblem
DENSITY_OPTIONS = (0.1, 0.3, 0.5, 0.7, 1.0)
DEFAULT_DENSITY = 0.5
SEED_MAX = 999999
DEFAULT_SEED = 35
RULE_LABELS = {"dantzig": "Dantzig (kleinste reduzierte Kosten)", "greatest": "Größter Zuwachs", "steepest": "Steepest Edge", "bland": "Bland (kleinster Index)", "random": "Zufall"}
RULE_SHORT = {"dantzig": "Dantzig", "greatest": "Größter Zuwachs", "steepest": "Steepest Edge", "bland": "Bland", "random": "Zufall"}
RATIO_LABELS = {"index": "Kleinster Index (einfach)", "lexicographic": "Lexikographisch (zyklenfrei)"}
DEFAULT_RULE, DEFAULT_RATIO = "dantzig", "index"
STEPS = {1: "1 · Regelvergleich", 2: "2 · Beales Zyklus", 3: "3 · Stillstand", 4: "4 · Aufwand"}
SWEEP_SEEDS = tuple(range(100000, 100005))
FEAS_SEEDS = tuple(range(200000, 200030))
SIZE_SWEEP = (5, 10, 20, 40)
TRANSPORT_SIZES = {5: (2, 3), 10: (3, 5), 20: (4, 8), 40: (6, 10)}
_BASE = {"kind": "textbook", "m": 10, "n": 10, "k": 4, "l": 8, "density": 0.5, "seed": 35, "rule": "dantzig", "ratio": "index", "step": 1}
PRESETS = {
    "Standardfall (Lehrbuchbeispiel)": dict(_BASE),
    "Beales Zyklus": {**_BASE, "kind": "beale", "step": 2},
    "Bland rettet Beale": {**_BASE, "kind": "beale", "rule": "bland", "step": 2},
    "Lexikographisch rettet Beale": {**_BASE, "kind": "beale", "ratio": "lexicographic", "step": 2},
    "Steepest Edge spart Pivots": {**_BASE, "kind": "mixed", "m": 40, "n": 40, "rule": "steepest", "step": 1},
    "Bland ist langsam": {**_BASE, "kind": "random", "m": 40, "n": 40, "rule": "bland", "step": 1},
    "Transport: Stillstand": {**_BASE, "kind": "transport", "k": 4, "l": 8, "step": 3},
    "Große Transportinstanz": {**_BASE, "kind": "transport", "k": 8, "l": 12, "step": 3},
    "Kein Stillstand im Zufall": {**_BASE, "kind": "random", "m": 20, "n": 20, "step": 3},
    "Aufwand über die Größe": {**_BASE, "kind": "mixed", "m": 20, "n": 20, "rule": "steepest", "step": 4},
}
PRESET_HELP = {
    "Standardfall (Lehrbuchbeispiel)": "Alle fünf Regeln finden das Optimum 36. Dantzig, größter Zuwachs, Steepest Edge und Zufall brauchen 2 Pivots, Bland 3. Dantzig kostet 84 Rechenoperationen, größter Zuwachs 102 und Steepest Edge 108 (die Preisgebung ist mitgezählt): auf so kleiner Instanz lohnt sich die klügere Regel nicht.",
    "Beales Zyklus": "Beales Beispiel (3 Ressourcen, 4 Dienste, zwei Ressourcen mit Bestand 0): Dantzig mit dem Indexrest im Quotiententest bewegt den Zielwert nie (6 Nullschritte) und steht nach 6 Pivots wieder in der Start-Basis: ein Zyklus, der Lauf würde ewig weiterkreisen. Die Demo bricht bei der ersten wiederholten Basis ab.",
    "Bland rettet Beale": "Dieselbe Instanz mit der Bland-Regel (kleinster Index): 6 Pivots, davon 4 Nullschritte, dann das Optimum 1.25 bei (1, 0, 1, 0). Bland kann nicht kreisen, zahlt aber mit Pivots.",
    "Lexikographisch rettet Beale": "Dantzig mit lexikographischem Quotiententest: bei einem Gleichstand gewinnt die lexikographisch kleinste Zeile. Das ist beweisbar zyklenfrei und braucht hier nur 2 Pivots bis zum Optimum 1.25.",
    "Steepest Edge spart Pivots": "Mischinstanz mit 40 Ressourcen und 40 Diensten: Dantzig 102 Pivots und 743 580 Operationen, Steepest Edge 52 Pivots und 454 028 Operationen (0.51 der Pivots, 0.61 der Operationen); im Median der fünf festen Instanzen 112 gegen 61 Pivots.",
    "Bland ist langsam": "Zufallsinstanz mit 40 Ressourcen und 40 Diensten: Bland braucht 95 Pivots, Dantzig 41 (2.3-fach); im Median der fünf festen Instanzen 61 gegen 17 Pivots (3.6-fach).",
    "Transport: Stillstand": "Transportproblem mit 4 Lagern und 8 Kunden (Angebot gleich Nachfrage): Dantzig braucht 33 Pivots, davon 3 Nullschritte (9 %); über 30 Instanzen sind es zusammen 12.7 % der Pivots (Nullschritte durch alle Pivots, nicht der Mittelwert der Anteile je Instanz). Die Ecke ändert sich bei einem Nullschritt nicht, nur die Basis.",
    "Große Transportinstanz": "8 Lager, 12 Kunden: Dantzig 85 Pivots, davon 15 Nullschritte (17.6 %); Steepest Edge nur 35 Pivots mit 3 Nullschritten. Zyklen kommen auch hier nicht vor.",
    "Kein Stillstand im Zufall": "Zufallsinstanz 20 x 20: Dantzig 7 Pivots ohne Gleichstand und ohne Nullschritt, wie alle fünf Regeln. Nullschritte gibt es hier nie; Stillstand ist ein Merkmal strukturierter Instanzen wie des Transportproblems.",
    "Aufwand über die Größe": "Mischinstanz 20 x 20 mit Steepest Edge: 24 Pivots gegen 37 bei Dantzig (Median über 30 Instanzen: 22.5 gegen 34). Die 🔬-Kurve zeigt Pivots und Operationen über die Größe für alle Regeln (30 Instanzen je Größe).",
}
# Beobachtete Spannweite der Kennzahl (MEDIAN über die 5 festen Instanzen Seeds 100000-100004) je Preset und Regel des Presets, mit Sicherheitsabstand: (Kennzahl, untere, obere Grenze).
PRESET_EXPECTED_BANDS = {
    "Standardfall (Lehrbuchbeispiel)": ("pivots", 2, 2),
    "Steepest Edge spart Pivots": ("pivots", 45, 80),
    "Bland ist langsam": ("pivots", 40, 85),
    "Transport: Stillstand": ("degenerate_share", 0.05, 0.25),
    "Große Transportinstanz": ("degenerate_share", 0.05, 0.25),
    "Kein Stillstand im Zufall": ("degenerate_share", 0.0, 0.0),
}
