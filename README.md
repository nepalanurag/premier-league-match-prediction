# Premier League Match Prediction

Walkthrough with all plots: https://nepalanurag.github.io/premier-league-match-prediction/

I built models to predict English Premier League match outcomes from three seasons (2023-24 through 2025-26, 1,130 matches).

I started with raw match data from football-data.co.uk and engineered my own features: rolling form for goals, shots, and shots on target, home/away splits, and a travel fatigue measure computed from stadium GPS coordinates and rest days. I compared logistic regression, Poisson regression for team goal rates, XGBoost, and LightGBM, and used the final Poisson model to generate win/draw/away probabilities for upcoming fixtures.

I also stress-tested the engineered features themselves with a transparent multinomial logistic model: the signal holds up across seasons (CV accuracy 0.50, 0.50, 0.46 for 2023-24, 2024-25, 2025-26, and a forward test training on the first two seasons and testing on 2025-26 at 0.48) and beats the naive baselines (0.51 vs 0.43 for always-home), but the travel fatigue features do not earn their keep: dropping them slightly improves both accuracy and log-loss. The numbers are in `analysis/metrics.json`, and the code that produced them is `analysis/season_and_ablation_shuffled.py` (the original script, preserved as the reference shuffled-CV version).

### Follow-up: ablation under the time-respecting protocol

The main notebook evaluates with `TimeSeriesSplit`, so as a follow-up robustness check I re-ran the season-by-season stability and ablation analysis under the same time-respecting protocol: `analysis/season_and_ablation.py` now uses forward-chaining splits (training only on matches that precede the evaluation matches). Both sets of numbers are reported; the notebook's main results are untouched.

Time-respecting results (`analysis/metrics_timeseries.json`, plots in `figures/epl_seasons_timeseries.png` and `figures/epl_ablation_timeseries.png`):
- Season-by-season forward-chain accuracy: 0.50 (2023-24), 0.45 (2024-25), 0.42 (2025-26); forward test (train on first two seasons, test on 2025-26): 0.48, same as the shuffled-CV figure.
- Ablation on all 12 features: accuracy 0.500, log-loss 1.035. Dropping travel fatigue: log-loss improves to 1.029 while accuracy moves slightly to 0.498. So under the time-respecting splits the travel-fatigue null on log-loss agrees with the shuffled-CV run (1.011 vs 1.014 with travel kept), while the accuracy side is now essentially flat instead of slightly favoring the drop (0.519 vs 0.511 in the shuffled run).
- The model still clears the naive baselines comfortably (always-home 0.42, class priors 0.38 on accuracy).

## Files

- `Final_Project.ipynb` - the full analysis notebook
- `requirements.txt` - Python package versions used to run the notebook
- `Final_Project.pdf` / `Final_Project.html` - rendered versions of the notebook
- `PROJECT REPORT.docx` / `PROJECT REPORT.pdf` - the written report
- `MATH_748_PROJECT_PROPOSAL.docx` / `.pdf` - the original project proposal
- `MATH_748_PROGRESS_REPORT_1.docx` / `.pdf`, `MATH_748_PROGRESS_REPORT_2.pdf` - progress updates
- `epl_features_2324.csv` - the engineered feature set
- `feature_summary_stats.csv`, `logistic_coefficients.csv`, `venue_form_delta.csv`, `away_travel_fatigue_leaders.csv`, `home_travel_fatigue_leaders.csv` - supporting outputs
- `analysis/season_and_ablation.py` - season-by-season stability and feature ablation under time-respecting splits (follow-up)
- `analysis/season_and_ablation_shuffled.py` - reference: original shuffled-CV version of the ablation
- `analysis/metrics_timeseries.json` - numbers from the time-respecting ablation run
