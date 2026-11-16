# 🌦️ Météo Clair — India Rainfall & Weather Forecast

Forecasts for today and the next 3 days of **rainfall (category, amount, probability), max/min temperature, humidity and wind**
for 411 Indian locations.

- **Model:** two-layer LSTM, 30-day input window, 7-day multi-horizon output, with an **ordinal rainfall head**
  (P(rain ≥ 2.5 mm), P(≥ 15.6 mm), P(≥ 64.5 mm) + amount) and a separate model for temperature, humidity and wind.
- **Training data (2015–2025):** IMD 0.25° gridded rain-gauge rainfall (mainland target) + NASA POWER daily weather.
- **Live input:** NASA POWER daily API (≈ 3-day delay), so "tomorrow" is lead day 4 of the model.
- The app runs the LSTM in plain NumPy (no PyTorch needed on the server).

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit web app (Live forecast, Validation 2025, About) |
| `forecast_core.py` | NumPy LSTM, feature engineering, NASA POWER download |
| `data/config.json` | Feature list, scaling statistics, rain thresholds, skill by lead |
| `data/live_*.npz` | Trained model weights (mainland/island × rain/continuous) |
| `data/locations_named.csv` | 411 locations with names |
| `data/validation_2025.parquet` | Test-year forecasts vs observations for the Validation tab |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud, free)
1. Create a **public GitHub repository** and upload all files, keeping the `data/` folder.
2. Go to **share.streamlit.io**, sign in with GitHub, click **Create app → Deploy a public app from GitHub**.
3. Choose the repository, branch `main`, main file `app.py`, and click **Deploy**.

*Research prototype — not an official forecast. For warnings, follow the India Meteorological Department.*
