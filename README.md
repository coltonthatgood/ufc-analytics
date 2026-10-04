# UFC Analytics (C++)

Fight-metric projections, winner prediction, and fighter style/movement profiles built from the
Kaggle dataset [UFC Complete Dataset (All events 1996-2024)](https://www.kaggle.com/datasets/maksbasher/ufc-complete-dataset-all-events-1996-2024)
(7,439 fights, 1994-03 to 2024-03; CC0-1.0, scraped from ufcstats.com). `data/` holds the two files
used: `large_dataset.csv` and `completed_events_small.csv` (event dates).

## Build & run

```sh
g++ -std=c++20 -O2 main.cpp -o ufc      # or: cmake -B build && cmake --build build
g++ -std=c++20 -O2 test.cpp -o ufc_test && ./ufc_test

./ufc evaluate                                   # backtest on the latest 20% of fights
./ufc profile "Islam Makhachev"                  # metrics, percentiles, style, movement habits
./ufc predict "Sean O'Malley" "Merab Dvalishvili" 5
```

Run from this folder (it looks for `data/`). Partial names work if unique (`profile adesanya`).

## How it works

- **No leakage.** The dataset's `*_total` career columns are scrape-time snapshots, so they're ignored.
  Every fighter's stats are rebuilt fight by fight in date order; each fight is predicted from what was
  known before it.
- **Rates** (sig strikes/min, TD/15, accuracy, defense, control share, ...) are shrunk toward the league
  average by pseudo-counts, so a debutant isn't scored on one 40-second fight.
- **Winner model:** L2 logistic regression on A-minus-B differences of 20 pre-fight features (Elo, striking,
  grappling, age, reach, experience, layoff, streak, finish rates). Training data is mirrored, so
  P(A beats B) = 1 - P(B beats A). `predict` ignores corner color.
- **Metric projection:** attacker's rate x sqrt(defender's conceded rate / league rate).
- **Method & length:** winner's win-method mix blended with loser's loss-method mix; expected minutes from
  the decision probability and the league's average finish time.
- **Style:** archetype (Striker / Wrestler / Submission grappler / All-rounder) from striking vs grappling
  percentiles, plus tags (volume striker, knockout power, hard to hit, chain wrestler, sprawl-and-brawl, ...).

## Backtest (`./ufc evaluate`, test = 1,488 fights from 2021-04 on)

| Winner prediction | Accuracy | Log loss |
|---|---|---|
| Always pick red corner | 57.9% | 0.700 |
| Elo only | 56.2% | 0.683 |
| Logistic, corner-blind | **62.0%** | **0.651** |
| ...both fighters 3+ UFC fights | 63.6% | 0.646 |

| Metric (MAE per fighter-fight) | Matchup model | Own career avg | League avg |
|---|---|---|---|
| Sig strikes / min | **1.673** | 1.736 | 1.829 |
| Sig strike accuracy | **0.088** | 0.090 | 0.094 |
| Takedowns / 15 min | **1.336** | 1.406 | 1.566 |
| Control time share | **0.148** | 0.161 | 0.177 |

The sqrt in the projection was chosen by looking at this same test split, so those numbers are slightly
optimistic.

## Limitations

- **Movement habits are inferred.** The dataset has fight totals only: no distance/clinch/ground split, no
  head/body/leg targets, no per-round data. Pace, position time, close-range work, level changes and
  scrambles are derived from strike/takedown/control totals. `fight_urls.txt` in the original download
  links to ufcstats.com pages that do have positional and per-round splits.
- Fighters are keyed by name (no IDs), so namesakes merge (e.g. the two Bruno Silvas).
- Data ends 2024-03-23; predictions use each fighter's record up to then.
