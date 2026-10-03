# Dashboard data — Premier League Match Outcome Prediction

Key results from `Final_Project.ipynb` (repo root), exported as small CSV/JSON
files so an interactive dashboard can be built from this folder alone.

## Provenance

Two kinds of files live here:

- **Published** — copied verbatim from the notebook's result tables
  (`model_ranking.csv`, `poisson_coefficients.csv`, `matchday38_predictions.csv`,
  and the headline numbers in `dataset_summary.json`).
- **Recomputed** — produced by re-running the notebook's own code (data
  download, feature engineering, Poisson walk-forward, calibration cells)
  against the repo's saved 1130-match feature set (`epl_features_2324.csv`),
  with static per-match columns (Bet365 odds, in-game stats) joined from
  football-data.co.uk. Reproduction check: market log-loss 0.9639 (exact),
  Poisson log-loss 1.0251 (notebook: 1.0250).

One caveat: the recomputed Poisson confusion matrix lands at 475/940 correct
(50.5%) versus the notebook's published 474/940 (50.4%). The draw column is
all zeros in both, and the log-loss matches, so this is one borderline
prediction flipping between library versions — the structure is faithful.

Rows are in a canonical (Date, HomeTeam, AwayTeam) order so the 5-fold
walk-forward splits are fully reproducible. Outcome labels everywhere:
`home_win` / `draw` / `away_win`.

## Files

- `dataset_summary.json` — dataset facts: 1130 matches, seasons 2023/24–2025/26,
  date range, feature matrix shape (1130, 12), the 12 feature names, outcome
  counts/shares (488 home / 366 away / 276 draw), naive always-pick-home
  accuracy (43.2%), goals per game (1.62 home / 1.38 away), shots per game
  (14.3 home / 11.8 away), Bet365 log-loss 0.9639, Poisson log-loss 1.0250 and
  out-of-sample accuracy 50.4% (474/940), evaluation method.
- `model_ranking.csv` — all nine models: `model`, `log_loss` (lower is better).
  Bet365 market baseline unbeaten at 0.9639; Poisson best model at 1.0250.
- `outcome_distribution.csv` — `outcome`, `count`, `share` over all 1130 matches.
- `poisson_confusion_matrix.csv` — 3x3, rows actual / columns predicted.
  The predicted-draw column is all zeros: the model never predicts a draw.
- `poisson_calibration.csv` — reliability-diagram points for the Poisson model:
  `outcome`, `bin`, `mean_predicted_prob`, `fraction_actual`, `n`.
  Quantile bins (8 per outcome), same settings as the notebook.
- `market_calibration.csv` — same layout for the Bet365 market baseline.
  Uniform bins (10 per outcome), same settings as the notebook. The market is
  very well calibrated (points sit near the diagonal).
- `shot_difference_curve.csv` — empirical P(home win) vs shot dominance:
  `Bin_Center`, `Bin_Left`, `Bin_Right` (home shots minus away shots),
  `empirical_home_win_prob`, `n`. 10 quantile bins.
- `poisson_walkforward_folds.csv` — per-fold `log_loss`, `accuracy`, `f1_macro`
  for the Poisson model (`fold`, `n_test` = 188 each).
- `poisson_coefficients.csv` — Poisson regression coefficients from the
  notebook's published table: `feature`, `display_name`, `home_goals_coef`,
  `away_goals_coef`. Venue-split form carries the most weight; travel fatigue
  barely registers.
- `matchday38_predictions.csv` — Poisson vs market probabilities (percent) for
  the ten final fixtures of 2025/26: `home_team`, `away_team`,
  `poisson_home_pct`, `poisson_draw_pct`, `poisson_away_pct`,
  `market_home_pct`, `market_draw_pct`, `market_away_pct`.
- `man_city_rolling_form.csv` — the team-level example from the notebook:
  every Man City match with `Date`, `Match_No`, `HomeTeam`, `AwayTeam`,
  `Team_Shots`, `Opp_Shots`, `Rolling_Shots`, `Rolling_Opp_Shots`
  (5-game rolling averages).
- `logistic_coefficients.csv` — logistic regression coefficients per feature
  (from the earlier repo export): `feature`, `Coefficient`.
- `feature_summary_stats.csv` — describe() output for the 12 features
  (from the earlier repo export). Note: 380 rows = 2023/24 season only.
- `venue_form_delta.csv` — per-team home goals minus away goals form
  (`HomeTeam`, `Home_GF_minus_Away_GF`), top 20.
- `home_travel_fatigue_leaders.csv` / `away_travel_fatigue_leaders.csv` —
  top-5 matches by travel fatigue: `Date`, `HomeTeam`, `AwayTeam`,
  `H_Travel_Fatigue_5` / `A_Travel_Fatigue_5` (miles over last 5 matches).
