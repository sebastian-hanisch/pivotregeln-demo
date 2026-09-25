# Pivotregeln und Entartung – wie wählt der Simplex, und was, wenn er stillsteht? – Streamlit-Demo

Zweites Stück der **Lineare-Programmierung-Reihe** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Kind der Wurzel [tableau-simplex-demo](https://github.com/sebastian-hanisch/tableau-simplex-demo). Dort war die Wahl in jedem Pivot festgelegt: die Spalte mit den kleinsten reduzierten Kosten tritt ein (**Dantzig-Regel**). Andere Regeln wählen anders: der **größte Zuwachs** des Zielwerts, **Steepest Edge** (Verbesserung je Kantenlänge), **Bland** (kleinster Index) oder eine **zufällige** Spalte. Dazu kommt die Zeilenwahl bei einem Gleichstand im Quotiententest: nach **kleinstem Index** oder **lexikographisch**. Bei einem Gleichstand steht der Simplex an einer entarteten Ecke, und dort kann er stillstehen (Nullschritte) oder sogar **kreisen**. Die Demo stellt vier Fragen, alle gemessen: **(1) Regelvergleich** – wie viele Pivots und wie viel Rechnung braucht jede Regel? **(2) Beales Zyklus** – kreist der Simplex wirklich, und was verhindert es? **(3) Stillstand** – wie oft steht er still? **(4) Aufwand** – wie wachsen die Unterschiede mit der Größe?

**Einordnung in die Reihe:** geplant sind zwölf Stücke, dies ist das zweite (Details in `lp-planung/PLAN.md` des Portfolio-Ordners):

```
Tableau-Simplex (Wurzel)                                                                  [gebaut: tableau-simplex-demo]
 ├─ Pivotregeln & Entartung  ─ Simplex im schlimmsten und im typischen Fall (Klee-Minty)  [DIESES STÜCK]  →  [nicht gebaut]
 ├─ Revised Simplex ─ Präsolve, Skalierung & Numerik                                     [nicht gebaut]
 ├─ Dualität & Sensitivität ─ Dualer Simplex & Neuoptimierung                            [nicht gebaut]
 ├─ Ellipsoid-Methode (Kontrast: polynomial in der Theorie)                              [nicht gebaut]
 └─ Innere Punkte ─ PDLP (Verfahren erster Ordnung) ─ Crossover & Simplex gegen Innere Punkte gegen PDLP  [nicht gebaut]
```

Ergebnis in Kürze: **Die klügere Regel spart Pivots und meist auch Rechnung – der Simplex kreist aber nur auf dem konstruierten Gegenbeispiel, und Stillstand gibt es nur bei struktureller Entartung.** Bei m = n = 40 braucht Steepest Edge im Median (30 Instanzen) 15.5 statt 21.5 Pivots auf Zufallsinstanzen, 51 statt 95.5 auf Mischinstanzen und 32 statt 53.5 im Transportproblem; mit Preisgebung kostet das auf Mischinstanzen 0.63 der Operationen von Dantzig, auf dem Lehrbuchbeispiel aber mehr (84 gegen 108). Bland braucht das 1.5- bis 3.7-fache der Pivots von Dantzig. **Beales Zyklus** kehrt mit Dantzig und Indexrest nach 6 Pivots zur Start-Basis zurück; in 2000 erzeugten Instanzen und 5000 zufälligen kleinen LPs im Stil von Beale trat **kein einziger Zyklus** auf. Nullschritte gibt es nur im Transportproblem (10.7 bis 19.6 % der Pivots je nach Regel), obwohl auch Zufalls-Instanzen mit Gleichständen im Quotiententest vorkommen.

| Frage | Ergebnis (Distributionszentrum, Standard-LP max c·x; **Median** über 30 Instanzen, Seeds 200000–200029, bzw. über 5 feste Instanzen, Seeds 100000–100004; vollständig deterministisch, `random.Random` mit Zeichenketten-Seeds; Aufwand in Gleitkomma-Operationen des dichten Tableaus, Preisgebung mitgezählt) |
|---|---|
| **Stimmt jede Regel?** | ✅ Alle fünf Regeln mit beiden Quotiententests erreichen das Optimum von HiGHS (Tests gegen `scipy.optimize.linprog`; Zufall, Mischung, Gleichstände, Transport, Lehrbuch, Beale, 120 kleine Zufalls-LPs mit gemischten Vorzeichen in allen drei Status), primal und dual zulässig, starke Dualität. Nur Dantzig, größter Zuwachs, Steepest Edge und Zufall mit Indexrest dürfen kreisen, und nur Beale tut es |
| **Regeln einzeln nachgerechnet** | Die erste Wahl jeder Regel wird unabhängig aus den Instanzdaten berechnet: Steepest Edge minimiert −c_j / √(1 + ‖A_j‖²) (40 Instanzen), der größte Zuwachs maximiert c_j mal den kleinsten Quotienten (40), Dantzig das größte c_j, Bland den kleinsten Index (30) |
| **Beales Zyklus** | Beales Beispiel (3 Ressourcen, 4 Dienste, Bestand 0 in zwei Ressourcen): Dantzig mit Indexrest: die Basisfolge (0, 5, 6) → (0, 1, 6) → (2, 1, 6) → (2, 3, 6) → (4, 3, 6) → (4, 5, 6) kehrt nach **6 Pivots** zur Start-Basis zurück, der Zielwert bleibt 0 (6 Nullschritte). **Bland**: 6 Pivots (4 Nullschritte) bis zum Optimum 1.25; **lexikographisch**: 2 Pivots; größter Zuwachs, Steepest Edge und Zufall mit Indexrest terminieren ebenfalls (höchstens 3 Pivots) |
| **Wer gewinnt Pivots?** | Median bei m = n = 40: Zufallsinstanz Dantzig 21.5, größter Zuwachs 15.5, Steepest Edge **15.5**, Bland 80, Zufall 59.5; Mischung 95.5 / 66 / **51** / 230 / 167.5; Transport (6 Lager, 10 Kunden) 53.5 / 35 / **32** / 81.5 / 57.5. Steepest Edge liegt überall unter Dantzig |
| **… und Operationen?** | Mit Preisgebung braucht Steepest Edge auf Mischinstanzen bei m = n = 40 **0.63** der Operationen von Dantzig. Auf dem Lehrbuchbeispiel kippt es: Dantzig 84, größter Zuwachs 102, Steepest Edge 108 Operationen (gleiche 2 Pivots, aber Preisgebung) |
| **Preis von Bland** | Pivots bei m = n = 40 im Verhältnis zu Dantzig: **3.7** (Zufall), **2.4** (Mischung), **1.5** (Transport). Bland braucht keine Preisgebung, spart also dort nichts, wo die Pivots so viel mehr werden |
| **Stillstand** | Nullschritte nur im **Transportproblem** (Angebot gleich Nachfrage): 12.7 % der Pivots bei Dantzig (4 Lager, 8 Kunden), 13.9 % bei 8 Lagern und 12 Kunden; je Regel zwischen **10.7 und 19.6 %** (Zufall und größter Zuwachs am meisten). In Zufalls-, Misch- und Gleichstands-Instanzen: 0 %, obwohl es in den Gleichstands-Instanzen in mehr als jedem sechsten Lauf einen Gleichstand im Quotiententest gibt |
| **Zyklen in der Praxis** | **0** Zyklen in 2000 erzeugten Instanzen (je 400 Zufall, Mischung, Gleichstände, Transport in zwei Größen, Dantzig + Indexrest) und in 5000 zufälligen kleinen LPs im Stil von Beale (Koeffizienten aus Vierteln, Bestand meist 0; 2518 davon mit Optimum, der Rest unbeschränkt) |
| **Preis der lexikographischen Regel** | Kaum: im Transportproblem (4 Lager, 8 Kunden) **31 statt 33 Pivots**, der Vergleichsaufwand macht 0.04 % der Operationen aus; ohne Gleichstand wählt er dieselbe Zeile wie der Indextest (Test auf 20 Instanzen) |

## Was die Demo zeigt

1. **Vier Schritte** (Schritt-Slider): **Regelvergleich** (alle fünf Regeln auf derselben Instanz: Tabelle mit Pivots, Nullschritten und Operationen, Balken für Pivots und Operationen (Pivots und Preisgebung getrennt), bei zwei Diensten die Pfade aller Regeln in einer Zeichnung, auf Abruf der Vergleich über 5 feste Instanzen) → **Beales Zyklus** (Tabelle aller Pivots mit Basis, Wiederholung rot markiert, "kreist" oder "terminiert", dazu alle zehn Kombinationen aus Regel und Quotiententest) → **Stillstand** (Schrittweite je Pivot, Nullschritte je Regel) → **Aufwand** (Kurven über die Größe je Regel).
2. **Kennzahlen des gewählten Laufs:** Pivots, Nullschritte, Operationen, Ergebnis.
3. **🔬 Auf Abruf:** Stillstand je Instanztyp (30 Instanzen), Zyklensuche (2000 erzeugte Instanzen, 5000 Beale-artige LPs), Sweeps über Größe / Dichte / Typ für Pivots, Operationen, Nullschritte und Anteil der Preisgebung.

Presets (10): Standardfall (Lehrbuchbeispiel), Beales Zyklus, Bland rettet Beale, Lexikographisch rettet Beale, Steepest Edge spart Pivots, Bland ist langsam, Transport: Stillstand, Große Transportinstanz, Kein Stillstand im Zufall, Aufwand über die Größe.

## Messwerte der Presets

| Preset | Einstellungen | Ergebnis |
|---|---|---|
| **Standardfall** | Lehrbuchbeispiel, alle Regeln | Optimum 36; Pivots 2/2/2/3/2 (Dantzig, größter Zuwachs, Steepest Edge, Bland, Zufall); Operationen 84/102/108 |
| **Beales Zyklus** | Beale, Dantzig, Indexrest | Zyklus nach 6 Pivots, 6 Nullschritte, Lauf würde ewig kreisen |
| **Bland rettet Beale** | Beale, Bland | 6 Pivots, 4 Nullschritte, Optimum 1.25 bei (1, 0, 1, 0) |
| **Lexikographisch rettet Beale** | Beale, Dantzig, lexikographisch | 2 Pivots, Optimum 1.25 |
| **Steepest Edge spart Pivots** | Mischung 40 x 40, Seed 35 | Dantzig 102 Pivots / 743 580 Operationen, Steepest Edge 52 / 454 028 (0.51 der Pivots, 0.61 der Operationen); Median über 5 feste Instanzen 112 gegen 61 Pivots |
| **Bland ist langsam** | Zufall 40 x 40, Seed 35 | Bland 95 Pivots, Dantzig 41 (2.3-fach); Median 61 gegen 17 (3.6-fach) |
| **Transport: Stillstand** | 4 Lager, 8 Kunden, Seed 35 | Dantzig 33 Pivots, 3 Nullschritte (9 %); über 30 Instanzen im Mittel 12.7 % |
| **Große Transportinstanz** | 8 Lager, 12 Kunden | Dantzig 85 Pivots, 15 Nullschritte (17.6 %); Steepest Edge 35 Pivots mit 3 Nullschritten |
| **Kein Stillstand im Zufall** | Zufall 20 x 20 | Dantzig 7 Pivots ohne Gleichstand, keine Regel hat einen Nullschritt |
| **Aufwand über die Größe** | Mischung 20 x 20, Steepest Edge | 24 Pivots gegen 37 bei Dantzig (Median über 30 Instanzen 22.5 gegen 34) |

## Modell und Verfahren

- **Instanz** (`piv_scenario.py`): die Auslastungsplanung aus dem Tableau-Simplex (Lehrbuchbeispiel, Fixtures, Zufall, Mischung, Zufall mit Gleichständen) plus **Beale** (Maximierungsform von Beales Beispiel von 1955) und **Transport** (Lager und Kunden auf einer Karte, ganzzahliges Angebot gleich Nachfrage, Kosten = Länge; Lager: Summe ≤ Angebot, Kunden: Summe ≥ Nachfrage; ganzzahlige Salden machen die Instanz stark entartet).
- **Simplex** (`piv_algorithm.py`): dichtes Tableau mit Zwei-Phasen-Start wie in Stück 1, dazu fünf Regeln (`dantzig`, `greatest`, `steepest`, `bland`, `random`) und zwei Quotiententests (`index`, `lexicographic`). **Zykluserkennung:** wiederholt sich bei einer deterministischen Regel eine Basis (Menge der Basisspalten), endet der Lauf mit Status "cycled". Steepest Edge nutzt die exakte Kantenlänge aus dem Tableau: √(1 + Σ t_ij²).
- **Aufwand:** ein Pivot kostet (Spalten + 1) + 2 · m · (Spalten + 1) Operationen; die Preisgebung kostet beim größten Zuwachs 2 m, bei Steepest Edge 2 m + 2 Operationen je Kandidat, der lexikographische Vergleich eine Operation je verglichene Zeile und Spalte. Ein Näherungsmaß, keine Laufzeit.
- **Notbremse aus Stück 1 entfernt:** der Wechsel zu Bland nach langem Stillstand ist hier keine versteckte Rettung, sondern eine eigene Regel.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Hypothese "Steepest Edge gewinnt nicht unbedingt bei den Operationen" – auf diesen Instanzen widerlegt** (0.63 der Operationen von Dantzig auf Mischinstanzen bei m = n = 40); auf kleinen Instanzen kippt es.
- **Vorab-Hypothese "Stillstand häufig bei Transport-LPs, Zyklen praktisch nie" – bestätigt** (10.7 bis 19.6 % Nullschritte, 0 Zyklen in 7000 Instanzen), aber nur für diese erzeugten Transportinstanzen mit ganzzahligem Angebot.
- **Zyklen sind konstruiert.** Beale kreist, zufällige Instanzen nicht. Das beweist nicht, dass es in der Praxis keine gibt, nur dass Zufall sie nicht findet.
- **Preisgebung im dichten Tableau.** Dort liegen alle Spalten vor, die Steepest-Edge-Längen kosten nur eine Summe je Spalte. Im Revised Simplex müssen sie mitgeführt und aktualisiert werden: die Rechnung sähe dort anders aus (Folgestück).
- **Keine Referenzrahmen (Devex), keine Störung nach Wolfe, kein Klee-Minty.** Der exponentielle schlimmste Fall ist Thema des nächsten Stücks.
- **Synthetische Instanzen, kleine Größen.** Bis 60 x 60; echte LPs sind größer, dünner und schlechter skaliert.

## Verifikation

- `tests/test_algorithm.py` (25 Tests): jede Regel × Quotiententest gegen HiGHS (Zielwert, Zulässigkeit, Dualität), alle Regeln bei Unzulässig/Unbeschränkt, 120 kleine LPs mit gemischten Vorzeichen; **Beales Zyklus mit der genauen Basisfolge**, Bland und lexikographisch terminieren; Steepest Edge, größter Zuwachs, Dantzig, Bland und Zufall einzeln unabhängig nachgerechnet; lexikographisch gleich Index ohne Gleichstand und nie ein Zyklus auf 240 Instanzen (der Zweig wird nachweislich ausgeführt); Zykluserkennung ohne Fehlalarm; Buchführung der Preisgebung; Transport ausgeglichen und entartet; Sonderfälle.
- `tests/test_scenario.py`, `test_evaluation.py`, `test_presets.py` (Bänder + jede Zahl der Hilfetexte), `test_claims.py` (jede Zahl aus README und App über die echten `ev.*`-Funktionen), `test_app.py` (Streamlit-AppTest: Voreinstellung, jedes Preset, jeder Schritt für jeden Instanztyp, jede Regel und jeder Quotiententest, Randwerte, Würfel, Permalink-Grenzen, Instanzwechsel, Experimente und Sweeps auf Abruf, Footer).
- Für die Prüfung genügt **pytest**; `scipy` dient nur als Gegenprobe (`requirements-dev.txt`), die App braucht nur numpy.

## Lokal starten

```bash
python -m venv venv && venv/Scripts/activate  # Windows; Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -W error::SyntaxWarning`.

## Literatur

- Bland, R. G. (1977). *New finite pivoting rules for the simplex method.* Mathematics of Operations Research 2(2), 103–107.
- Forrest, J. J., & Goldfarb, D. (1992). *Steepest-edge simplex algorithms for linear programming.* Mathematical Programming 57, 341–374.
- Beale, E. M. L. (1955). *Cycling in the dual simplex algorithm.* Naval Research Logistics Quarterly 2(4), 269–275.
- Dantzig, G. B., Orden, A., & Wolfe, P. (1955). *The generalized simplex method for minimizing a linear form under linear inequality restraints.* Pacific Journal of Mathematics 5, 183–195.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
