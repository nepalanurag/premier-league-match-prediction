# Premier League Match Prediction

I built models to predict English Premier League match outcomes from the 2023-24 season.

I started with raw match data from football-data.co.uk and engineered my own features: rolling form for goals, shots, and shots on target, home/away splits, and a travel fatigue measure computed from stadium GPS coordinates and rest days. I compared logistic regression, Poisson regression for team goal rates, XGBoost, and LightGBM, and used the final Poisson model to generate win/draw/away probabilities for upcoming fixtures.

## Files

- `Final_Project.ipynb` - the full analysis notebook
- `Final_Project.pdf` / `Final_Project.html` - rendered versions of the notebook
- `PROJECT REPORT.docx` / `PROJECT REPORT.pdf` - the written report
- `MATH_748_PROJECT_PROPOSAL.docx` / `.pdf` - the original project proposal
- `MATH_748_PROGRESS_REPORT_1.docx` / `.pdf`, `MATH_748_PROGRESS_REPORT_2.pdf` - progress updates
- `epl_features_2324.csv` - the engineered feature set
- `feature_summary_stats.csv`, `logistic_coefficients.csv`, `venue_form_delta.csv`, `away_travel_fatigue_leaders.csv`, `home_travel_fatigue_leaders.csv` - supporting outputs
