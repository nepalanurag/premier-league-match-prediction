# Premier League Match Prediction

Walkthrough with all plots: https://nepalanurag.github.io/premier-league-match-prediction/

I built models to predict English Premier League match outcomes from three seasons (2023-24 through 2025-26, 1,130 matches).

I started with raw match data from football-data.co.uk and engineered my own features: rolling form for goals, shots, and shots on target, home/away splits, and a travel fatigue measure computed from stadium GPS coordinates and rest days. I compared logistic regression, Poisson regression for team goal rates, XGBoost, and LightGBM, and used the final Poisson model to generate win/draw/away probabilities for upcoming fixtures.

## Files

- `Final_Project.ipynb` - the full analysis notebook
- `requirements.txt` - Python package versions used to run the notebook
- `Final_Project.pdf` / `Final_Project.html` - rendered versions of the notebook
- `PROJECT REPORT.docx` / `PROJECT REPORT.pdf` - the written report
- `MATH_748_PROJECT_PROPOSAL.docx` / `.pdf` - the original project proposal
- `MATH_748_PROGRESS_REPORT_1.docx` / `.pdf`, `MATH_748_PROGRESS_REPORT_2.pdf` - progress updates
- `epl_features_2324.csv` - the engineered feature set
- `feature_summary_stats.csv`, `logistic_coefficients.csv`, `venue_form_delta.csv`, `away_travel_fatigue_leaders.csv`, `home_travel_fatigue_leaders.csv` - supporting outputs

## Follow-up: season stability and ablation

I stress-tested the engineered features with a transparent multinomial logistic model: [expansion analysis](https://nepalanurag.github.io/premier-league-match-prediction/expansion.html). The signal holds across seasons (CV accuracy 0.50, 0.50, 0.47) and beats always-home (0.51 vs 0.43), but the travel fatigue features do not earn their keep: dropping them slightly improves both accuracy and log-loss. Code in `analysis/`.
