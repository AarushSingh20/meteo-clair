"""
Core logic for the India rainfall & weather forecast app.

- Loads the trained LSTM weights (exported from PyTorch as .npz) and runs them in pure NumPy,
  so the web app does not need PyTorch.
- Downloads the latest daily data for a location from the NASA POWER API.
- Builds exactly the same input features that were used in training.
"""
import json
import os

import numpy as np
import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# NASA POWER parameter -> training column name (verified: identical values, r = 1.000)
NASA_PARAMS = {
    "T2M": "temperature", "T2MDEW": "dewpoint", "RH2M": "humidity", "PS": "pressure",
    "WS10M": "wind_speed", "PRECTOTCORR": "nasa_rain", "T2M_MAX": "t2m_max", "T2M_MIN": "t2m_min",
    "U10M": "u10", "V10M": "v10", "WS10M_MAX": "ws10_max", "QV2M": "qv2m",
}

RAIN_CATS = ["No rain", "Light rain", "Moderate rain", "Heavy rain"]
RAIN_RANGES = ["< 2.5 mm", "2.5–15.5 mm", "15.6–64.4 mm", "≥ 64.5 mm"]
LO = np.array([0, 2.5, 15.6, 64.5])
HI = np.array([0, 15.5, 64.4, np.inf])


# ----------------------------------------------------------------- model in NumPy
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def calibrate(p, pw):
    """Undo the class weighting used in training (positive weight w inflates the odds by w).
    Turns the model's raw score into a real probability: q = p / (p + w (1 - p))."""
    w = np.asarray(pw, dtype=float)
    return p / (p + w * (1.0 - p))


def rain_label(cat, p_rain):
    """Plain-language label for the forecast card."""
    if cat == 0:
        return "No rain" if p_rain < 0.2 else "Mostly dry, chance of a shower"
    if cat == 3:
        return "Heavy rain possible"
    return RAIN_CATS[cat] + (" likely" if p_rain >= 0.5 else " possible")


def rain_icon(cat, p_rain):
    if cat == 0:
        return "☀️" if p_rain < 0.2 else "🌤️"
    return ["☀️", "🌦️", "🌧️", "⛈️"][cat]


class NumpyLiveLSTM:
    """Same architecture as LiveLSTM in training: 2-layer LSTM + rain head + continuous head."""

    def __init__(self, npz_path, H):
        w = np.load(npz_path)
        self.w = {k: w[k].astype(np.float64) for k in w.files}
        self.H = H
        self.layers = sum(1 for k in self.w if k.startswith("lstm.weight_ih_l"))

    def _lstm(self, x):
        # x: (B, T, F). PyTorch gate order: input, forget, cell(g), output
        h_seq = x
        for layer in range(self.layers):
            Wi = self.w[f"lstm.weight_ih_l{layer}"]; Wh = self.w[f"lstm.weight_hh_l{layer}"]
            b = self.w[f"lstm.bias_ih_l{layer}"] + self.w[f"lstm.bias_hh_l{layer}"]
            B, T, _ = h_seq.shape; hid = Wh.shape[1]
            h = np.zeros((B, hid)); c = np.zeros((B, hid)); outs = []
            for t in range(T):
                z = h_seq[:, t] @ Wi.T + h @ Wh.T + b
                i, f, g, o = np.split(z, 4, axis=1)
                c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
                h = _sigmoid(o) * np.tanh(c)
                outs.append(h)
            h_seq = np.stack(outs, 1)
        return h_seq[:, -1]

    def _head(self, h, name):
        z = np.maximum(h @ self.w[f"{name}.0.weight"].T + self.w[f"{name}.0.bias"], 0)
        return z @ self.w[f"{name}.2.weight"].T + self.w[f"{name}.2.bias"]

    def __call__(self, x):
        h = self._lstm(x)
        return (self._head(h, "rain").reshape(-1, self.H, 4),
                self._head(h, "cont").reshape(-1, self.H, 4))


class Forecaster:
    def __init__(self, data_dir=DATA_DIR):
        self.cfg = json.load(open(os.path.join(data_dir, "config.json")))
        self.H, self.L = self.cfg["H"], self.cfg["L"]
        self.feats = self.cfg["LIVE_FEATS"]
        self.models = {}
        for region in self.cfg["regions"]:
            self.models[region] = {
                "rain": NumpyLiveLSTM(os.path.join(data_dir, f"live_{region}_rain.npz"), self.H),
                "cont": NumpyLiveLSTM(os.path.join(data_dir, f"live_{region}_cont.npz"), self.H),
            }

    def predict(self, region, X_window):
        """X_window: DataFrame (L rows) with self.feats columns -> per-lead forecast DataFrame."""
        rc = self.cfg["regions"][region]
        x = (X_window[self.feats].to_numpy(np.float64) - np.array(rc["x_mean"])) / np.array(rc["x_std"])
        x = x[None]
        r_out, _ = self.models[region]["rain"](x)
        _, c_out = self.models[region]["cont"](x)
        r_out, c_out = r_out[0], c_out[0]
        cont = c_out * np.array(rc["y_std"]) + np.array(rc["y_mean"])
        thr = np.array(rc["thr"])
        p = _sigmoid(r_out[:, :3])
        cat = np.zeros(self.H, int)
        for k in range(3):
            cat = np.where(p[:, k] >= thr[:, k], k + 1, cat)
        amount = np.where(cat == 0, 0.0, np.clip(np.expm1(r_out[:, 3]), LO[cat], HI[cat]))
        q = calibrate(p, rc.get("pw", [1.0, 1.0, 1.0]))
        return pd.DataFrame({
            "lead": np.arange(1, self.H + 1), "rain_cat": cat, "rain_mm": amount,
            "p_rain": q[:, 0], "p_moderate": q[:, 1], "p_heavy": q[:, 2],
            "tmax": cont[:, 0], "tmin": cont[:, 1], "humidity": np.clip(cont[:, 2], 0, 100),
            "wind": np.clip(cont[:, 3], 0, None)})

    def skill(self, region, lead):
        for r in self.cfg["regions"][region]["skill"]:
            if r["lead"] == lead:
                return r
        return None


# ----------------------------------------------------------------- features
def build_features(raw, latitude, longitude):
    """raw: DataFrame indexed by date with the training column names (see NASA_PARAMS values).
    Returns the same engineered features that were used in training."""
    d = raw.copy()
    d["dtr"] = d.t2m_max - d.t2m_min
    d["dew_dep"] = d.temperature - d.dewpoint
    d["p_change"] = d.pressure.diff().fillna(0)
    d["qv_chg"] = d.qv2m.diff().fillna(0)
    r = d.nasa_rain
    d["n_rain_log"] = np.log1p(r)
    d["n_rain_3d"] = np.log1p(r.rolling(3, min_periods=1).sum().clip(lower=0))
    d["n_rain_7d"] = np.log1p(r.rolling(7, min_periods=1).sum().clip(lower=0))
    d["n_rainy_7d"] = (r >= 2.5).rolling(7, min_periods=1).sum()
    d["n_days_since_rain"] = r.groupby((r >= 2.5).cumsum()).cumcount().clip(upper=60)
    doy = pd.DatetimeIndex(d.index).dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d["latitude"] = latitude
    d["longitude"] = longitude
    return d


# ----------------------------------------------------------------- NASA POWER
def fetch_nasa(latitude, longitude, today, days=100, timeout=90):
    """Download the last `days` days of daily data. Returns DataFrame (complete days only)."""
    start = (pd.Timestamp(today) - pd.Timedelta(days=days)).strftime("%Y%m%d")
    end = pd.Timestamp(today).strftime("%Y%m%d")
    params = dict(parameters=",".join(NASA_PARAMS), community="AG", latitude=latitude,
                  longitude=longitude, start=start, end=end, format="JSON")
    resp = requests.get(NASA_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    raw = pd.DataFrame(resp.json()["properties"]["parameter"])
    raw.index = pd.to_datetime(raw.index, format="%Y%m%d")
    raw = raw.replace(-999.0, np.nan).rename(columns=NASA_PARAMS)
    raw = raw[list(NASA_PARAMS.values())]
    complete = raw.notna().all(axis=1)
    if not complete.any():
        raise ValueError("NASA POWER returned no complete days for this location.")
    raw = raw.loc[: complete[complete].index[-1]]          # drop the trailing not-yet-available days
    raw = raw.interpolate(limit=3).ffill().bfill()          # fill rare isolated gaps
    return raw


# ------------------------------------------------------------------ air quality (Open-Meteo / CAMS, Indian NAQI)
# CPCB National Air Quality Index breakpoints: (pollutant concentration, index) pairs, linear between them.
AQI_IDX = [0, 50, 100, 200, 300, 400, 500]
AQI_BP = {
    "pm2_5": [0, 30, 60, 90, 120, 250, 380],           # µg/m³, 24-h mean
    "pm10": [0, 50, 100, 250, 350, 430, 510],          # µg/m³, 24-h mean
    "nitrogen_dioxide": [0, 40, 80, 180, 280, 400, 520],   # µg/m³, 24-h mean
    "sulphur_dioxide": [0, 40, 80, 380, 800, 1600, 2100],  # µg/m³, 24-h mean
    "ozone": [0, 50, 100, 168, 208, 748, 1000],        # µg/m³, max 8-h mean
    "carbon_monoxide": [0, 1.0, 2.0, 10, 17, 34, 50],  # mg/m³, max 8-h mean
}
AQI_NAMES = {"pm2_5": "PM2.5", "pm10": "PM10", "nitrogen_dioxide": "NO₂", "sulphur_dioxide": "SO₂",
             "ozone": "O₃", "carbon_monoxide": "CO"}
AQI_CATS = [(50, "Good", "#2e9e5b"), (100, "Satisfactory", "#8cc63f"), (200, "Moderate", "#f2c12e"),
            (300, "Poor", "#f08a24"), (400, "Very poor", "#e0402f"), (10_000, "Severe", "#9b1c31")]


def sub_index(pol, c):
    if c is None or not np.isfinite(c):
        return np.nan
    return float(np.interp(c, AQI_BP[pol], AQI_IDX))


def aqi_category(aqi):
    for top, name, color in AQI_CATS:
        if aqi <= top:
            return name, color
    return AQI_CATS[-1][1:]


def indian_aqi(h):
    """AQI from an hourly frame covering one 24-h period. Returns (aqi, main pollutant, {pollutant: concentration})."""
    conc = {}
    for p in ["pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide"]:
        conc[p] = h[p].mean()
    for p in ["ozone", "carbon_monoxide"]:
        conc[p] = h[p].rolling(8, min_periods=6).mean().max()
    conc["carbon_monoxide"] = conc["carbon_monoxide"] / 1000  # µg/m³ → mg/m³
    subs = {p: sub_index(p, c) for p, c in conc.items()}
    subs = {p: s for p, s in subs.items() if np.isfinite(s)}
    if not subs or not ({"pm2_5", "pm10"} & set(subs)):
        return np.nan, None, conc
    main = max(subs, key=subs.get)
    return round(subs[main]), AQI_NAMES[main], conc


def fetch_air(lat, lon, timeout=30):
    """Hourly pollutant forecast (CAMS model via Open-Meteo, no key). Index = IST hour."""
    r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", timeout=timeout, params=dict(
        latitude=round(float(lat), 3), longitude=round(float(lon), 3),
        hourly=",".join(AQI_BP), past_days=1, forecast_days=5, timezone="Asia/Kolkata"))
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({k: v for k, v in h.items() if k != "time"}, index=pd.to_datetime(h["time"])).astype(float)
    return df


def air_summary(df, now):
    """now: naive IST datetime. Returns dict with 'now' (last 24 h, like CPCB's live AQI) and 'days' {date: {aqi, main, pm2_5, pm10}}."""
    past = df[(df.index > now - pd.Timedelta(hours=24)) & (df.index <= now)]
    aqi, main, conc = indian_aqi(past)
    days = {}
    for d, g in df.groupby(df.index.date):
        if len(g) >= 20:
                       d_aqi, d_main, d_conc = indian_aqi(g)  # own names, so the "now" values above are not overwritten
            days[d] = dict(aqi=d_aqi, main=d_main, pm2_5=d_conc["pm2_5"], pm10=d_conc["pm10"])
    best = None
    nxt = df[(df.index > now) & (df.index <= now + pd.Timedelta(hours=24))]
    if len(nxt) >= 6:
        end = nxt.pm2_5.rolling(3).mean().idxmin()  # cleanest 3-hour window in the next 24 h
        best = (end - pd.Timedelta(hours=2), end + pd.Timedelta(hours=1))
    return dict(aqi=aqi, main=main, pm2_5=conc["pm2_5"], pm10=conc["pm10"], days=days, best=best)


AQI_HEALTH = {  # CPCB health statements, shortened
    "Good": "Minimal health impact — a good day to be outside.",
    "Satisfactory": "May cause minor breathing discomfort to sensitive people.",
    "Moderate": "Breathing discomfort for people with asthma, lung or heart disease, children and older adults.",
    "Poor": "Breathing discomfort for most people on long exposure. Limit long outdoor exercise.",
    "Very poor": "Respiratory illness on long exposure. Avoid outdoor exercise; consider a mask outdoors.",
    "Severe": "Affects healthy people too. Stay indoors as much as possible.",
}