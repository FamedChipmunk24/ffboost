# FFBoost: Next-Season NFL Fantasy Projections with XGBoost

FFBoost projects next-season NFL fantasy points (PPR) and positional rankings for QBs, RBs, WRs, and TEs. It is trained on player-season data from 2002–2025 and evaluated with rolling, time-aware backtests over 2016–2025.

## How it works

- **Model:** per-position XGBoost ensembles (regressor, classifier, and ranker) behind scikit-learn preprocessing pipelines (`ColumnTransformer`, imputation, one-hot encoding).
- **Features:** multi-year usage and efficiency metrics, snap shares, team Elo, schedule rest, and draft capital.
- **Leakage-safe evaluation:** for each test season, the model trains only on earlier seasons. Cohort features use only information available before that season.

## Backtest results (rolling, 2016–2025)

Share of each position's relevant next-season finishers (QB top 24, RB top 60, WR top 80, TE top 36) correctly identified within the draft pool:

| Position | Hit rate | Avg. rank MAE |
|----------|---------:|--------------:|
| QB       | 79%      | 9.5           |
| RB       | 76%      | 19.6          |
| WR       | 76%      | 26.1          |
| TE       | 72%      | 16.2          |

## Repository layout

```
nfl_ffboost/      Model code: backtesting, prediction, feature merges, ADP blending
nfl_data/         Training dataset and supporting team/snap/schedule data
nfl_backtests/    Backtest metrics and per-player predicted vs. actual results
nfl_predictions/  2026 projections
```

## Run it

```bash
pip install -r requirements.txt

# Rolling backtest (2016-2025); trains and saves models to nfl_backtests/ffboost/models
python -m nfl_ffboost.neo_backtest --data nfl_data/dataset_v7_signals_2002_2025.csv --out_dir nfl_backtests/ffboost

# 2026 projections using the saved models
python -m nfl_ffboost.neo_predict --data nfl_data/dataset_v7_signals_2002_2025.csv --models_dir nfl_backtests/ffboost/models --out_csv nfl_predictions/predictions_2026_ffboost.csv
```

## Data

Player and team data come from public sources: [nflverse](https://github.com/nflverse) (snap counts, schedules, draft picks) and FootballDB (season stats). The model can also use historical ADP (average draft position), but that data came from a commercial export and isn't included. Without it, the model trains without ADP features.

## Author

Stephen Goodwin, Honors B.S. in Computer Science (AI concentration), DePaul University
