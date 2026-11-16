"""
Météo Clair — India rainfall & weather forecast (Streamlit app, "bento" design)
Multi-task LSTM with an ordinal rainfall head, trained on IMD gauge-based rainfall + NASA POWER (2015–2025).
"""
import base64
import datetime as dt
import html
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import forecast_core as fc

try:  # browser location ("Use my location" button); the app still works without this library
    from streamlit_js_eval import get_geolocation
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

APP_NAME = "Météo Clair"
st.set_page_config(page_title=f"{APP_NAME} · India weather forecast", page_icon="🌦️", layout="wide")

INK, MUTED, ORANGE, ORANGE_SOFT, NAVY, BLUE = "#172238", "#5a6578", "#f07a2b", "#f7b27f", "#172238", "#4f93c9"
QUICK_PLACES = ["New Delhi", "Mumbai City", "Chennai", "Kolkata", "Bengaluru"]

# ------------------------------------------------------------------ page state
if "page" not in st.session_state:
    st.session_state.page = "forecast"


def set_page(p):
    st.session_state.page = p


def set_loc(label):
    st.session_state.loc = st.session_state.prev_loc = label


# ------------------------------------------------------------------ styles
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea, select {{font-family: 'Plus Jakarta Sans', Inter, system-ui, sans-serif !important;}}
.stApp {{background: #e3ecec;}}
header[data-testid="stHeader"] {{background: transparent;}}
.block-container {{max-width: 1384px; background: #ffffff; border-radius: 32px; padding: 22px 26px 26px 22px !important;
  margin-top: 28px; margin-bottom: 28px; box-shadow: 0 10px 40px rgba(23,34,56,0.10);}}

/* left icon rail */
.st-key-rail {{background: #f5f7f8; border-radius: 22px; padding: 14px 6px; gap: 8px;}}
.st-key-rail button {{width: 100%; min-height: 64px; border: 0; border-radius: 16px; background: transparent; color: {MUTED};
  padding: 6px 2px; font-weight: 600;}}
.st-key-rail button > div {{flex-direction: column; align-items: center; gap: 2px;}}
.st-key-rail button:hover {{background: #ffffff; color: {INK};}}
.st-key-rail button p {{font-size: 11px; white-space: nowrap;}}
.st-key-place_0 button > div, .st-key-place_1 button > div {{justify-content: flex-start; width: 100%;}}
.st-key-place_0 button p, .st-key-place_1 button p {{text-align: left;}}
.st-key-weekmetric button p {{font-size: 13px;}}
@media (max-width: 640px) {{
  .st-key-rail {{flex-direction: row; justify-content: space-around; padding: 6px;}}
  .st-key-rail > div {{flex: 1 1 0;}}
  .st-key-rail > div:first-child {{display: none;}}
  .st-key-rail button {{min-height: 52px;}}
}}
.st-key-nav_{st.session_state.page} button {{background: #ffe7d6 !important; color: #c4561a !important;}}
.mc-logo {{width: 52px; height: 52px; border-radius: 16px; background: {ORANGE}; display: flex; align-items: center;
  justify-content: center; margin: 2px auto 10px auto;}}

/* header */
.mc-date {{font-size: 13px; color: {MUTED};}}
.mc-name {{font-size: 24px; font-weight: 800; letter-spacing: -0.3px; color: {INK};}}
div[data-testid="stSelectbox"] > div > div {{border-radius: 999px; background: #f2f5f6; border: 0; min-height: 46px;}}

/* cards */
.mc-hero {{position: relative; border-radius: 26px; padding: 22px 24px; min-height: 300px; overflow: hidden; color: {INK};}}
.mc-chip {{display: inline-flex; align-items: center; gap: 8px; background: #fff; border-radius: 999px; padding: 8px 14px;
  font-size: 14px; font-weight: 600;}}
.mc-hero .t1 {{font-size: 26px; font-weight: 700; margin-top: 16px;}}
.mc-hero .t2 {{font-size: 14px; color: #44516a;}}
.mc-hero .big {{font-size: 88px; font-weight: 800; line-height: 1; letter-spacing: -3px; margin-top: 16px;}}
.mc-hero .big span {{font-size: 30px; font-weight: 600; color: #44516a; letter-spacing: 0; margin-left: 8px;}}
.mc-hero .illo {{position: absolute; right: 26px; top: 26px;}}
.mc-tiles {{position: absolute; right: 24px; bottom: 22px; display: flex; gap: 12px; flex-wrap: wrap; justify-content: flex-end;}}
.mc-tile {{background: #fff; border-radius: 18px; padding: 12px 16px; min-width: 104px;}}
.mc-tile.lime {{background: #d7ef9b;}}
.mc-tile .l {{font-size: 12px; color: #44516a; font-weight: 600;}}
.mc-tile.lime .l {{color: #3a4a1a;}}
.mc-tile .v {{font-size: 22px; font-weight: 800;}}
.st-key-mapcard {{border-radius: 26px; overflow: hidden;}}
.st-key-place_0 button, .st-key-place_1 button {{width: 100%; min-height: 72px; border: 0; border-radius: 20px;
  background: #f5f7f8; justify-content: flex-start; text-align: left; padding: 12px 16px;}}
.st-key-place_0 button:hover, .st-key-place_1 button:hover {{background: #eef2f4; color: {INK};}}
.mc-air {{background: {NAVY}; color: #fff; border-radius: 22px; padding: 18px 20px; margin-top: 4px;}}
.mc-air .h {{font-size: 18px; font-weight: 700; margin-bottom: 10px;}}
.mc-air .row {{display: flex; gap: 22px; flex-wrap: wrap;}}
.mc-air .l {{font-size: 12px; color: #b8c2d4;}}
.mc-air .v {{font-size: 24px; font-weight: 800;}}
.mc-air .v small {{font-size: 12px; font-weight: 600; color: #b8c2d4;}}
.mc-air .aqi {{display: flex; align-items: center; justify-content: space-between; gap: 12px;}}
.mc-air .aqi .n {{font-size: 44px; font-weight: 800; line-height: 1.05;}}
.mc-air .aqi .cat {{display: inline-block; padding: 5px 12px; border-radius: 999px; font-weight: 700; font-size: 14px; color: #fff;}}
.mc-air .scale {{position: relative; height: 8px; border-radius: 4px; margin: 10px 0 16px;
  background: linear-gradient(90deg, #2e9e5b 0 10%, #8cc63f 10% 20%, #f2c12e 20% 40%, #f08a24 40% 60%, #e0402f 60% 80%, #9b1c31 80%);}}
.mc-air .scale div {{position: absolute; top: -4px; width: 4px; height: 16px; margin-left: -2px; border-radius: 2px; background: #fff;}}
.st-key-weekcard {{border: 1px solid #e8edf0; border-radius: 22px; padding: 18px 20px;}}
.mc-weekh {{font-size: 20px; font-weight: 700; line-height: 1.25; color: {INK};}}
.mc-bars {{display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; align-items: end; min-height: 190px; margin-top: 6px;}}
.mc-bars > div {{display: flex; flex-direction: column; align-items: center; gap: 8px; border-right: 1px solid #edf1f3;}}
.mc-bars > div:last-child {{border-right: 0;}}
.mc-bars .v {{font-size: 18px; font-weight: 800;}}
.mc-bars .bar {{width: 30px; border-radius: 10px;}}
.mc-bars b {{font-size: 14px;}}
.mc-bars .s {{font-size: 12px; color: {MUTED};}}
.mc-air .tip {{background: rgba(255,255,255,0.08); border-radius: 14px; padding: 10px 12px; font-size: 13px;
  line-height: 1.45; color: #e3e8f0; margin: 0 0 16px;}}
.mc-air .tip .best {{margin-top: 6px; color: #fff;}}
.st-key-geo_btn button {{width: 44px; height: 44px; min-height: 44px; padding: 0; border-radius: 999px; border: 0;
  background: #f5f7f8; color: {ORANGE};}}
.st-key-geo_btn button:hover {{background: {ORANGE}; color: #fff;}}
.mc-insight {{background: #f4f7f9; border-radius: 12px; padding: 8px 12px; font-size: 13px; color: {INK}; margin: 8px 0 0;}}
.st-key-weekcard [data-testid="stPlotlyChart"] {{cursor: pointer;}}
.mc-aqiweek {{display: flex; flex-direction: column; gap: 4px; margin-top: 10px;}}
.mc-aqiweek .r {{display: grid; grid-template-columns: 92px 1fr 58px; align-items: center; gap: 14px;
  padding: 9px 0; border-bottom: 1px solid #edf1f3;}}
.mc-aqiweek .r:last-child {{border-bottom: 0;}}
.mc-aqiweek .d b {{display: block; font-size: 15px; color: {INK};}}
.mc-aqiweek .d span, .mc-aqiweek .s {{font-size: 12px; color: {MUTED};}}
.mc-aqiweek .scale {{position: relative; height: 8px; border-radius: 4px; margin: 4px 0 7px;
  background: linear-gradient(90deg, #2e9e5b 0 10%, #8cc63f 10% 20%, #f2c12e 20% 40%, #f08a24 40% 60%, #e0402f 60% 80%, #9b1c31 80%);}}
.mc-aqiweek .scale div {{position: absolute; top: -5px; width: 6px; height: 18px; margin-left: -3px; border-radius: 3px;
  background: {INK}; border: 2px solid #fff; box-shadow: 0 1px 3px rgba(0,0,0,.25);}}
.mc-aqiweek .n {{color: #fff; font-weight: 800; font-size: 17px; text-align: center; border-radius: 12px; padding: 7px 0;}}
.mc-aqikey {{display: flex; flex-wrap: wrap; align-items: center; row-gap: 4px; font-size: 11px; color: {MUTED}; margin-top: 8px;}}
.mc-aqikey i {{display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin: 0 4px 0 10px;}}
.mc-aqikey i:first-child {{margin-left: 0;}}
.mc-tom {{position: relative; background: #e3f3b4; border-radius: 22px; padding: 18px 20px; min-height: 330px;
  overflow: hidden; display: flex; flex-direction: column; color: {INK};}}
.mc-tom .k {{font-size: 13px; color: #3a4a1a; font-weight: 600;}}
.mc-tom .p {{font-size: 20px; font-weight: 800;}}
.mc-tom .illo {{position: absolute; right: 14px; top: 64px;}}
.mc-tom .big {{margin-top: auto; font-size: 46px; font-weight: 800; letter-spacing: -1px;}}
.mc-tom .c {{font-size: 14px; font-weight: 600;}}
.mc-tom .lbl {{display: flex; justify-content: space-between; font-size: 12px; margin-top: 8px; color: #3a4a1a;}}
.mc-tom .bar {{height: 8px; border-radius: 99px; background: #fff;}}
.mc-tom .bar div {{height: 8px; border-radius: 99px; background: {NAVY};}}
.mc-foot {{font-size: 12px; color: {MUTED}; text-align: center; margin-top: 14px;}}
@media (max-width: 760px) {{
  .mc-hero .big {{font-size: 64px;}} .mc-hero .illo {{transform: scale(0.7); transform-origin: top right;}}
  .mc-tiles {{position: static; justify-content: flex-start; margin-top: 18px;}}
}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data & model
@st.cache_resource
def load_forecaster():
    return fc.Forecaster(fc.DATA_DIR)


@st.cache_data
def load_locations():
    return pd.read_csv(os.path.join(fc.DATA_DIR, "locations_named.csv")).sort_values("label").reset_index(drop=True)


@st.cache_data
def load_validation():
    return pd.read_parquet(os.path.join(fc.DATA_DIR, "validation_2025.parquet"))


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def get_nasa(lat, lon, today_str):
    return fc.fetch_nasa(lat, lon, today_str)


@st.cache_data(ttl=3600, show_spinner=False)
def get_air(lat, lon, hour_str):
    return fc.fetch_air(lat, lon)


F = load_forecaster()
LOCS = load_locations()
LABEL_SET = set(LOCS.label)
today = dt.datetime.now(ZoneInfo("Asia/Kolkata")).date()


def region_of(row):
    return "island" if int(row.is_island) == 1 else "mainland"


def short_name(label):
    return label.split(" — ")[0].split(" (")[0]


def forecast_for(row):
    raw = get_nasa(float(row.latitude), float(row.longitude), str(today))
    feats = fc.build_features(raw, row.latitude, row.longitude)
    window = feats.iloc[-F.L:]
    issue = window.index[-1].date()
    f = F.predict(region_of(row), window)
    f["date"] = [issue + dt.timedelta(days=int(h)) for h in f.lead]
    return raw, issue, f


def find_label(name):
    hits = LOCS[LOCS.label.str.startswith(name)]
    return hits.label.iloc[0] if len(hits) else None


# ------------------------------------------------------------------ search any place in India (free geocoders, no key)
STATE_CODES = {
    "andaman and nicobar": "AN", "andhra pradesh": "AP", "arunachal pradesh": "AR", "assam": "AS", "bihar": "BR",
    "chandigarh": "CH", "chhattisgarh": "CT", "dadra and nagar haveli": "DD", "daman and diu": "DD", "delhi": "DL",
    "goa": "GA", "gujarat": "GJ", "himachal pradesh": "HP", "haryana": "HR", "jammu and kashmir": "JK",
    "jharkhand": "JH", "karnataka": "KA", "kerala": "KL", "ladakh": "LA", "lakshadweep": "LD", "maharashtra": "MH",
    "meghalaya": "ML", "manipur": "MN", "madhya pradesh": "MP", "mizoram": "MZ", "nagaland": "NL", "odisha": "OR",
    "orissa": "OR", "punjab": "PB", "puducherry": "PY", "pondicherry": "PY", "rajasthan": "RJ", "sikkim": "SK",
    "tamil nadu": "TN", "telangana": "TG", "tripura": "TR", "uttar pradesh": "UP", "uttarakhand": "UK",
    "west bengal": "WB",
}
MAX_KM = 250  # farther than this from every training location = outside the model's area


def state_code(name):
    n = (name or "").lower().replace("&", "and").replace("national capital territory of ", "").replace("nct of ", "")
    for k, v in STATE_CODES.items():
        if k in n:
            return v
    return (name or "")[:2].upper()


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def geocode(query):
    """Up to 5 matches in India as dicts (name, state, lat, lon). Open-Meteo first, OpenStreetMap Nominatim as backup."""
    name, _, hint = query.partition(",")
    name, hint = name.strip(), hint.strip().lower()
    out = []
    try:
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search", timeout=10,
                         params=dict(name=name, count=10, language="en", format="json", countryCode="IN"))
        for g in r.json().get("results") or []:
            out.append(dict(name=g["name"], state=g.get("admin1", ""), lat=g["latitude"], lon=g["longitude"]))
    except Exception:
        pass
    if hint:  # "Noida, Uttar Pradesh" → prefer matches in that state
        out.sort(key=lambda g: hint not in g["state"].lower() and hint.upper() != state_code(g["state"]))
    if not out:
        try:
            r = requests.get("https://nominatim.openstreetmap.org/search", timeout=10,
                             headers={"User-Agent": "MeteoClair/1.0 (student research project)"},
                             params=dict(q=query, format="jsonv2", countrycodes="in", limit=5, addressdetails=1))
            for g in r.json():
                a = g.get("address", {})
                nm = g.get("name") or g["display_name"].split(",")[0]
                out.append(dict(name=nm, state=a.get("state", a.get("territory", "")), lat=float(g["lat"]), lon=float(g["lon"])))
        except Exception:
            pass
    return out[:5]


def nearest_loc(lat, lon):
    """Nearest of the 411 training locations and its distance in km."""
    la, lo = np.radians(LOCS.latitude.to_numpy()), np.radians(LOCS.longitude.to_numpy())
    p, q = np.radians(lat), np.radians(lon)
    km = 6371 * 2 * np.arcsin(np.sqrt(np.sin((la - p) / 2) ** 2 + np.cos(p) * np.cos(la) * np.sin((lo - q) / 2) ** 2))
    i = int(np.argmin(km))
    return LOCS.iloc[i], float(km[i])


def place_from_geocode(g):
    near, km = nearest_loc(g["lat"], g["lon"])
    st_code = f", {state_code(g['state'])}" if g.get("state") else ""
    label = f"{g['name']}{st_code} ({g['lat']:.2f}°N, {g['lon']:.2f}°E)"
    row = pd.Series(dict(location_id=f"geo_{g['lat']:.3f}_{g['lon']:.3f}", latitude=round(g["lat"], 4),
                         longitude=round(g["lon"], 4), is_island=near.is_island, label=label,
                         near_id=int(near.location_id), near_label=near.label, near_km=km))
    return row, km


if "custom" not in st.session_state:
    st.session_state.custom = {}  # label -> row (pd.Series) for places found by search


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def reverse_geocode(lat, lon):
    """Place name for a point (OpenStreetMap Nominatim). Falls back to 'My location'."""
    try:
        r = requests.get("https://nominatim.openstreetmap.org/reverse", timeout=10,
                         headers={"User-Agent": "MeteoClair/1.0 (student research project)"},
                         params=dict(lat=lat, lon=lon, format="jsonv2", zoom=12, addressdetails=1, **{"accept-language": "en"}))
        a = r.json().get("address", {})
        name = next((a[k] for k in ["city", "town", "village", "suburb", "city_district", "county", "state_district"]
                     if a.get(k)), "My location")
        return dict(name=name, state=a.get("state", a.get("territory", "")), lat=lat, lon=lon)
    except Exception:
        return dict(name="My location", state="", lat=lat, lon=lon)


def locate_me():
    """Ask the browser for the user's position (shows the browser's permission prompt). Runs only after the button."""
    pos = get_geolocation(component_key=f"geo_{st.session_state.get('geo_n', 0)}")
    if pos is None:  # the browser hasn't answered yet; the component reruns the app when it does
        st.caption("📍 Waiting for your browser — please allow location access…")
        return
    st.session_state.geo_req = False
    st.session_state.geo_n = st.session_state.get("geo_n", 0) + 1  # fresh component next time
    if "coords" not in pos:
        st.session_state.geo_msg = ("Couldn't get your location (permission denied or unavailable). "
                                    "You can still search for your place.")
        st.rerun()
    lat, lon = round(float(pos["coords"]["latitude"]), 4), round(float(pos["coords"]["longitude"]), 4)
    if nearest_loc(lat, lon)[1] > MAX_KM:
        st.session_state.geo_msg = "Forecasts are only available for places in India."
        st.rerun()
    row, _ = place_from_geocode(reverse_geocode(lat, lon))
    st.session_state.custom = {row.label: row, **{k: v for k, v in st.session_state.custom.items() if k != row.label}}
    st.session_state.pending_loc = row.label
    st.session_state.geo_msg = f"📍 Showing your location: {short_name(row.label)}"
    st.rerun()


def on_search():
    q = (st.session_state.loc or "").strip()
    if q in LABEL_SET or q in st.session_state.custom:
        st.session_state.prev_loc = q
        return
    hits = geocode(q) if len(q) >= 2 else []
    hits = [g for g in hits if nearest_loc(g["lat"], g["lon"])[1] <= MAX_KM]
    if not hits:
        st.session_state.loc = st.session_state.prev_loc
        st.session_state.geo_msg = (f"Couldn't find “{q}” in India. Check the spelling, or add the state, "
                                    f"e.g. “Noida, Uttar Pradesh”.")
        return
    for g in reversed(hits):  # keep every match in the list, best one selected
        row, _ = place_from_geocode(g)
        st.session_state.custom = {row.label: row, **{k: v for k, v in st.session_state.custom.items() if k != row.label}}
    best, _ = place_from_geocode(hits[0])
    st.session_state.loc = st.session_state.prev_loc = best.label
    if len(hits) > 1:
        st.session_state.geo_msg = (f"Showing {short_name(best.label)}. Other places called “{q.split(',')[0]}” "
                                    f"are now in the search list.")


# ------------------------------------------------------------------ weather icons (Meteocons by Bas Milius, MIT licence)
ICON_DIR = os.path.join(os.path.dirname(__file__), "assets", "icons")


@st.cache_data
def icon_uri(name):
    with open(os.path.join(ICON_DIR, f"{name}.svg"), "rb") as fh:
        return "data:image/svg+xml;base64," + base64.b64encode(fh.read()).decode()


def condition_icon(cat, p):
    if cat == 3:
        return "thunderstorms-rain"
    if cat == 2:
        return "rain"
    if cat == 1:
        return "partly-cloudy-day-drizzle"
    return "clear-day" if p < 0.2 else "partly-cloudy-day"


def illo(cat, p, size=200):
    return f'<img src="{icon_uri(condition_icon(cat, p))}" width="{size}" height="{size}" alt="">'


def small_icon(name, size=22):
    return f'<img src="{icon_uri(name)}" width="{size}" height="{size}" alt="" style="vertical-align:middle;margin:-4px 2px 0 -4px">'


def hero_bg(cat, p):
    if cat == 3:
        return "#cfd9e6"
    if cat >= 1:
        return "#d8e4ee"
    return "#dceff4" if p < 0.2 else "#e2ebf1"


def amount_text(c, mm):
    if c == 3:
        return f"≥ {mm:.0f} mm if it happens"
    if c > 0:
        return f"≈ {mm:.1f} mm expected"
    return ""


# ------------------------------------------------------------------ "this week" charts (interactive, Plotly)
BEAUFORT = [(2, "Calm"), (6, "Light air"), (12, "Light breeze"), (20, "Gentle breeze"), (29, "Moderate breeze"),
            (39, "Fresh breeze"), (50, "Strong breeze"), (1e9, "Gale")]
RAIN_COL = {0: "#9cc7e6", 1: "#4f93c9", 2: "#2b6cb0", 3: "#3b3f8f"}


def beaufort(kmh):
    return next(n for top, n in BEAUFORT if kmh < top)


def day_names(d):
    return ["Today" if x == today else f"{x:%a}" for x in d.date], [f"{x:%a %d %b}" for x in d.date]


def base_layout(fig, height=250):
    fig.update_layout(height=height, margin=dict(l=4, r=4, t=36, b=4), showlegend=False, bargap=0.45,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", dragmode=False,
                      font=dict(family="Plus Jakarta Sans, sans-serif", color=INK, size=13),
                      hoverlabel=dict(bgcolor="#ffffff", bordercolor="#dfe6ea", font=dict(color=INK, size=13)))
    fig.update_xaxes(showgrid=False, zeroline=False, fixedrange=True, tickfont=dict(size=14, color=INK))
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f4", griddash="dot", zeroline=False, fixedrange=True,
                     tickfont=dict(size=11, color=MUTED))
    return fig


def week_chart(metric, d):
    """Returns (one-line insight, interactive Plotly figure) for Rain / Temp / Wind."""
    x, full = day_names(d)
    fig = go.Figure()
    if metric == "Rain":
        p = d.p_rain * 100
        cats = d.rain_cat.astype(int)
        hover = [f"<b>{f}</b><br>{html.escape(fc.rain_label(int(c), pr))}<br>Chance of rain: <b>{pr*100:.0f}%</b>"
                 f"<br>Moderate (≥15.6 mm): {pm*100:.0f}%<br>Heavy (≥64.5 mm): {ph*100:.0f}%"
                 f"{'<br>Amount: ' + amount_text(int(c), mm) if int(c) > 0 else ''}"
                 for f, c, pr, pm, ph, mm in zip(full, cats, d.p_rain, d.p_moderate, d.p_heavy, d.rain_mm)]
        fig.add_trace(go.Bar(x=x, y=[100] * len(x), marker=dict(color="#f1f5f7", cornerradius=14),
                             hovertext=hover, hoverinfo="text"))
        fig.add_trace(go.Bar(x=x, y=np.maximum(p, 3), marker=dict(color=[RAIN_COL[int(c)] for c in cats], cornerradius=14),
                             hovertext=hover, hoverinfo="text"))
        fig.add_trace(go.Scatter(x=x, y=np.maximum(p, 3) + 9, mode="text", text=[f"<b>{v:.0f}%</b>" for v in p],
                                 textfont=dict(size=15, color=INK), hoverinfo="skip"))
        fig.update_layout(barmode="overlay")
        fig.update_yaxes(range=[0, 104], tickvals=[0, 50, 100], ticksuffix="%")
        for xi, c, pr in zip(x, cats, d.p_rain):
            fig.add_layout_image(source=icon_uri(condition_icon(int(c), pr)), x=xi, y=1.0, xref="x", yref="paper",
                                 sizex=0.55, sizey=0.26, xanchor="center", yanchor="bottom")
        base_layout(fig, 260).update_layout(margin=dict(t=62))
        wet = d[d.p_rain >= 0.5]
        insight = (f"🌧️ Rain likely on <b>{', '.join(day_names(wet)[0])}</b>" if len(wet)
                   else f"☀️ Dry days ahead — highest chance of rain is <b>{p.max():.0f}%</b> ({x[int(p.values.argmax())]})")
        return insight, fig
    if metric == "Temp":
        hover = [f"<b>{f}</b><br>Max: <b>{hi:.0f}°C</b><br>Min: <b>{lo:.0f}°C</b><br>Humidity: {h:.0f}%"
                 for f, hi, lo, h in zip(full, d.tmax, d.tmin, d.humidity)]
        lo, hi = float(d.tmin.min()), float(d.tmax.max())
        fig.add_trace(go.Bar(x=x, y=d.tmax - d.tmin, base=d.tmin, hovertext=hover, hoverinfo="text", width=0.32,
                             marker=dict(color=d.tmax, colorscale=[[0, "#f7c9a4"], [1, ORANGE]],
                                         cmin=hi - 4, cmax=hi, cornerradius=14)))
        fig.add_trace(go.Scatter(x=x, y=d.tmax, mode="markers+text", text=[f"<b>{v:.0f}°</b>" for v in d.tmax],
                                 textposition="top center", textfont=dict(size=16, color=INK),
                                 marker=dict(size=12, color="#fff", line=dict(color=ORANGE, width=3)),
                                 hovertext=hover, hoverinfo="text"))
        fig.add_trace(go.Scatter(x=x, y=d.tmin, mode="markers+text", text=[f"{v:.0f}°" for v in d.tmin],
                                 textposition="bottom center", textfont=dict(size=13, color=MUTED),
                                 marker=dict(size=10, color="#fff", line=dict(color=BLUE, width=3)),
                                 hovertext=hover, hoverinfo="text"))
        fig.update_yaxes(range=[lo - 6, hi + 6], ticksuffix="°")
        base_layout(fig, 260)
        insight = (f"🌡️ Warmest: <b>{x[int(d.tmax.values.argmax())]}, {hi:.0f}°C</b> · coolest night: "
                   f"<b>{lo:.0f}°C</b> · day–night swing ≈ {float((d.tmax - d.tmin).mean()):.0f}°")
        return insight, fig
    # Wind
    k = d.wind * 3.6
    top = max(30.0, float(k.max()) * 1.4)
    for lo_, hi_, nm, col in [(0, 6, "calm", "#fbfcfd"), (6, 20, "light breeze", "#f5f9fb"),
                              (20, 39, "moderate–fresh", "#eef4f8"), (39, 300, "strong", "#e6eff5")]:
        if lo_ < top:
            fig.add_hrect(y0=lo_, y1=min(hi_, top), fillcolor=col, line_width=0, layer="below")
            fig.add_annotation(x=1, xref="paper", xanchor="right", y=min(hi_, top), yanchor="top", showarrow=False,
                               text=nm, font=dict(size=10, color=MUTED))
    hover = [f"<b>{f}</b><br>Wind: <b>{v:.0f} km/h</b> ({v/3.6:.1f} m/s)<br>{beaufort(v)}" for f, v in zip(full, k)]
    fig.add_trace(go.Scatter(x=x, y=k, mode="lines+markers+text", line=dict(shape="spline", color=BLUE, width=4),
                             fill="tozeroy", fillcolor="rgba(79,147,201,0.12)", text=[f"<b>{v:.0f}</b>" for v in k],
                             textposition="top center", textfont=dict(size=15, color=INK),
                             marker=dict(size=13, color="#fff", line=dict(color=BLUE, width=3)),
                             hovertext=hover, hoverinfo="text"))
    fig.update_yaxes(range=[0, top], ticksuffix=" km/h", showgrid=False)
    base_layout(fig, 260)
    fig.update_xaxes(range=[-0.45, len(x) - 0.55])
    insight = (f"🍃 Mostly a <b>{beaufort(float(k.mean())).lower()}</b> · strongest: "
               f"<b>{x[int(k.values.argmax())]}, {k.max():.0f} km/h</b>")
    return insight, fig


# ------------------------------------------------------------------ layout: rail + main
rail, main = st.columns([1, 14], gap="medium")

with rail:
    with st.container(key="rail"):
        st.markdown('<div class="mc-logo"><svg width="30" height="30" viewBox="0 0 24 24" fill="#ffffff" aria-hidden="true">'
                    '<path d="M7 19h10.5a4 4 0 0 0 .6-7.96A6 6 0 0 0 6.6 10.2 4.4 4.4 0 0 0 7 19z"/></svg></div>',
                    unsafe_allow_html=True)
        st.button("Today", icon=":material/space_dashboard:", key="nav_forecast", on_click=set_page, args=("forecast",))
        st.button("Map", icon=":material/location_on:", key="nav_map", on_click=set_page, args=("map",))
        st.button("Check", icon=":material/bar_chart:", key="nav_check", on_click=set_page, args=("check",))
        st.button("About", icon=":material/info:", key="nav_about", on_click=set_page, args=("about",))

with main:
    # ---------------- header
    h1, h2 = st.columns([3, 2], vertical_alignment="center")
    h1.markdown(f'<div class="mc-date">{today:%A, %d %B %Y}</div><div class="mc-name">{APP_NAME}</div>',
                unsafe_allow_html=True)
    if "loc" not in st.session_state:
        st.session_state.loc = st.session_state.prev_loc = find_label("New Delhi") or LOCS.label.iloc[0]
    if pending := st.session_state.pop("pending_loc", None):  # place found by "Use my location"
        st.session_state.loc = st.session_state.prev_loc = pending
    labels = list(st.session_state.custom) + LOCS.label.tolist()
    if st.session_state.loc not in labels:  # safety: unknown value → go back to the last good place
        st.session_state.loc = st.session_state.get("prev_loc", labels[0])
    with h2:
        s1, s2 = st.columns([8, 1], gap="small", vertical_alignment="center") if HAS_GEO else (st.container(), None)
        with s1:
            st.selectbox("Search any city, town or district in India", labels, key="loc", label_visibility="collapsed",
                         accept_new_options=True, on_change=on_search,
                         placeholder="Search any place in India, e.g. Greater Noida…")
        if s2 is not None:
            s2.button("", icon=":material/my_location:", key="geo_btn", help="Use my current location",
                      on_click=lambda: st.session_state.update(geo_req=True))
        if st.session_state.get("geo_req"):
            locate_me()
        if msg := st.session_state.pop("geo_msg", None):
            st.toast(msg, icon=":material/travel_explore:")
    if st.session_state.loc in st.session_state.custom:
        row = st.session_state.custom[st.session_state.loc]
    else:
        row = LOCS[LOCS.label == st.session_state.loc].iloc[0]
    region = region_of(row)
    page = st.session_state.page

    # ================================================================ FORECAST
    if page in ("forecast", "map"):
        try:
            with st.spinner("Getting the latest weather data from NASA POWER…"):
                raw, issue, fcst = forecast_for(row)
        except Exception as e:
            st.error(f"Could not reach NASA POWER right now ({e}). Please try again in a few minutes.")
            st.stop()
        show = fcst[fcst.date >= today].head(4).reset_index(drop=True)
        now_ist = dt.datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None, minute=0, second=0, microsecond=0)
        try:
            air = fc.air_summary(get_air(float(row.latitude), float(row.longitude), str(now_ist)), pd.Timestamp(now_ist))
        except Exception:
            air = None
        if show.empty:
            st.warning("The latest NASA data is too old to forecast today. Please try again later.")
            st.stop()

    # Real street map from plain image tiles. Default: OpenStreetMap (no key). Optional: MapTiler if a key is in secrets.
    try:
        MAPTILER_KEY = st.secrets["MAPTILER_KEY"]
    except Exception:
        MAPTILER_KEY = ""
    if MAPTILER_KEY:
        TILES = [f"https://api.maptiler.com/maps/streets-v2/256/{{z}}/{{x}}/{{y}}@2x.png?key={MAPTILER_KEY}"]
        TILE_ATTR = "© MapTiler © OpenStreetMap contributors"
    else:
        TILES = ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"]
        TILE_ATTR = "© OpenStreetMap contributors"

    def map_figure(height, zoom):
        t = show.iloc[0]
        lat, lon = float(row.latitude), float(row.longitude)
        fig = go.Figure()
        fig.add_trace(go.Scattermap(lat=LOCS.latitude, lon=LOCS.longitude, mode="markers",
                                    marker=dict(size=8, color=BLUE, opacity=0.65), text=LOCS.label, hoverinfo="text"))
        # selected place: white ring + orange dot, always at the centre of the map
        fig.add_trace(go.Scattermap(lat=[lat], lon=[lon], mode="markers", marker=dict(size=30, color="#ffffff"),
                                    hoverinfo="skip"))
        fig.add_trace(go.Scattermap(lat=[lat], lon=[lon], mode="markers", marker=dict(size=20, color=ORANGE),
                                    hovertext=[row.label], hoverinfo="text"))
        fig.add_annotation(x=0.03, y=0.05, xref="paper", yref="paper", xanchor="left", yanchor="bottom", showarrow=False,
                           text=f"<b>{t.tmax:.0f}°C</b>  Today · {html.escape(short_name(row.label))}",
                           bgcolor="#ffffff", borderpad=8, font=dict(size=14, color=INK))
        fig.update_layout(
            map=dict(style="white-bg", center=dict(lat=lat, lon=lon), zoom=zoom,
                     layers=[dict(below="traces", sourcetype="raster", source=TILES,
                                  sourceattribution=TILE_ATTR)]),
            margin=dict(l=0, r=0, t=0, b=0), height=height, showlegend=False)
        return fig

    if page == "forecast":
        t = show.iloc[0]
        c, p = int(t.rain_cat), float(t.p_rain)
        # ---------------- row 1: Today hero + map
        r1a, r1b = st.columns([7, 5], gap="medium")
        r1a.markdown(f"""
<div class="mc-hero" style="background:{hero_bg(c, p)}">
  <span class="mc-chip"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{INK}" stroke-width="2"
    stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 21s7-6.2 7-12a7 7 0 0 0-14 0c0 5.8 7 12 7 12z"/>
    <circle cx="12" cy="9" r="2.5"/></svg>{html.escape(short_name(row.label))}</span>
  <div class="t1">Today</div>
  <div class="t2">{html.escape(fc.rain_label(c, p))}{(' · ' + amount_text(c, t.rain_mm)) if c else ''}</div>
  <div class="big">{t.tmax:.0f}°<span>/ {t.tmin:.0f}°C</span></div>
  <div class="illo">{illo(c, p, 190)}</div>
  <div class="mc-tiles">
    <div class="mc-tile lime"><div class="l">{small_icon("raindrops")}Rain chance</div><div class="v">{p*100:.0f}%</div></div>
    <div class="mc-tile"><div class="l">{small_icon("humidity")}Humidity</div><div class="v">{t.humidity:.0f}%</div></div>
    <div class="mc-tile"><div class="l">{small_icon("wind")}Wind</div><div class="v">{t.wind*3.6:.0f} km/h</div></div>
  </div>
</div>""", unsafe_allow_html=True)
        with r1b:
            with st.container(key="mapcard"):
                st.plotly_chart(map_figure(300, 5.3), width="stretch", key=f"map_card_{row.location_id}",
                                config={"displayModeBar": False, "scrollZoom": True})

        # ---------------- row 2: air | week | tomorrow
        r2a, r2b, r2c = st.columns([4, 5, 3], gap="medium")
        with r2a:
            if air and np.isfinite(air["aqi"]):
                cat, col = fc.aqi_category(air["aqi"])
                aqi_html = (f'<div class="aqi"><div><div class="l">Air quality index (AQI)</div>'
                            f'<div class="n">{air["aqi"]:.0f}</div></div><div><span class="cat" style="background:{col}">'
                            f'{cat}</span><div class="l" style="margin-top:6px">main pollutant: {air["main"]}</div></div></div>'
                            f'<div class="scale"><div style="left:{min(air["aqi"], 500) / 5:.1f}%"></div></div>')
                pm_html = (f'<div><div class="l">PM2.5</div><div class="v">{air["pm2_5"]:.0f}<small> µg/m³</small></div></div>'
                           f'<div><div class="l">PM10</div><div class="v">{air["pm10"]:.0f}<small> µg/m³</small></div></div>')
                tip = fc.AQI_HEALTH.get(cat, "")
                if air.get("best"):
                    s0, s1 = air["best"]
                    hr = lambda t: f"{t.hour % 12 or 12} {'am' if t.hour < 12 else 'pm'}"  # works on Windows too
                    tip += (f'<div class="best">🕒 Cleanest air in the next 24 h: <b>{hr(s0)} – {hr(s1)}'
                            f'{"" if s0.date() == today else " (tomorrow)"}</b></div>')
                aqi_html += f'<div class="tip">{tip}</div>'
            else:
                aqi_html = '<div class="l" style="margin-bottom:10px">Air quality data is not available right now.</div>'
                pm_html = ""
            st.markdown(f"""<div class="mc-air"><div class="h">Air today</div>{aqi_html}<div class="row">{pm_html}
  <div><div class="l">{small_icon("humidity", 20)}Humidity</div><div class="v">{t.humidity:.0f}%</div></div>
  <div><div class="l">{small_icon("wind", 20)}Wind</div><div class="v">{t.wind*3.6:.0f}<small> km/h</small></div></div></div></div>""",
                        unsafe_allow_html=True)

        with r2b:
            with st.container(key="weekcard"):
                wa, wb = st.columns([2, 3], vertical_alignment="top")
                metric = st.session_state.get("weekmetric") or "Rain"
                title = {"Rain": "Will it rain<br>this week?", "Temp": "How warm<br>this week?",
                         "Wind": "How windy<br>this week?", "AQI": "Air quality<br>this week"}[metric]
                wa.markdown(f'<div class="mc-weekh">{title}</div>', unsafe_allow_html=True)
                metric = wb.segmented_control("Show", ["Rain", "Temp", "Wind", "AQI"], default="Rain", key="weekmetric",
                                              label_visibility="collapsed") or "Rain"
                if metric == "AQI":
                    days = (air or {}).get("days", {})
                    rows_html = ""
                    for _, d in show.iterrows():
                        a = days.get(d.date)
                        if d.date == today and air and np.isfinite(air["aqi"]):  # same "now" value as the Air today card
                            a = dict(aqi=air["aqi"], main=air["main"], pm2_5=air["pm2_5"])
                        dn = "Today" if d.date == today else f"{d.date:%A}"
                        if not a or not np.isfinite(a["aqi"]):
                            rows_html += (f'<div class="r"><div class="d"><b>{dn}</b><span>{d.date:%d %b}</span></div>'
                                          f'<div class="g"><span class="s">no data</span></div><div></div></div>')
                            continue
                        cat, col = fc.aqi_category(a["aqi"])
                        rows_html += (
                            f'<div class="r"><div class="d"><b>{dn}</b><span>{d.date:%d %b}</span></div>'
                            f'<div class="g"><div class="scale"><div style="left:{min(a["aqi"], 500) / 5:.1f}%"></div></div>'
                            f'<span class="s">{cat} · main {a["main"]} · PM2.5 {a["pm2_5"]:.0f} µg/m³</span></div>'
                            f'<div class="n" style="background:{col}">{a["aqi"]:.0f}</div></div>')
                    st.markdown(f'<div class="mc-aqiweek">{rows_html}</div>'
                                f'<div class="mc-aqikey"><i style="background:#2e9e5b"></i>Good<i style="background:#8cc63f"></i>Satisfactory'
                                f'<i style="background:#f2c12e"></i>Moderate<i style="background:#f08a24"></i>Poor'
                                f'<i style="background:#e0402f"></i>Very poor<i style="background:#9b1c31"></i>Severe</div>'
                                f'<div class="mc-foot" style="margin-top:6px">India National AQI (CPCB scale) · '
                                f'from the Copernicus CAMS air-quality forecast, not our LSTM</div>', unsafe_allow_html=True)
                else:
                    insight, fig = week_chart(metric, show)
                    st.markdown(f'<div class="mc-insight">{insight}</div>', unsafe_allow_html=True)
                    st.plotly_chart(fig, width="stretch", key=f"week_{metric}_{row.location_id}",
                                    config={"displayModeBar": False, "scrollZoom": False})
                    st.markdown('<div class="mc-foot">Hover or tap a day for details</div>', unsafe_allow_html=True)

        with r2c:
            if len(show) > 1:
                tm = show.iloc[1]; tc, tp = int(tm.rain_cat), float(tm.p_rain)
                st.markdown(f"""<div class="mc-tom">
  <div class="k">Tomorrow · {tm.date:%a %d %b}</div>
  <div class="p">{html.escape(short_name(row.label).split(',')[0])}</div>
  <div class="illo">{illo(tc, tp, 130)}</div>
  <div class="big">{tm.tmax:.0f}°C</div>
  <div class="c">{html.escape(fc.rain_label(tc, tp))}</div>
  <div class="lbl"><span>Rain chance</span><b>{tp*100:.0f}%</b></div>
  <div class="bar"><div style="width:{tp*100:.0f}%"></div></div>
</div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="mc-foot">Forecast from NASA POWER data up to {issue:%d %b %Y} '
                    f'({(today - issue).days} days ago) · {"island model" if region == "island" else "model trained on IMD rain-gauge data"} · '
                    f'chances are calibrated · research prototype, not an official forecast</div>', unsafe_allow_html=True)

    # ================================================================ MAP
    elif page == "map":
        st.markdown(f"#### Map · {short_name(row.label)}")
        with st.container(key="mapcard"):
            st.plotly_chart(map_figure(620, 6), width="stretch", key=f"map_page_{row.location_id}",
                            config={"displayModeBar": False, "scrollZoom": True})
        st.caption("Orange = selected place (today's max temperature). Blue dots = the 411 forecast locations; "
                   "hover to see names, scroll to zoom. Choose another place with the search box above.")

    # ================================================================ MODEL CHECK
    elif page == "check":
        st.markdown("#### Model check — forecasts vs what actually happened in 2025")
        st.caption("2025 was never seen during training. With NASA's ~3-day delay, 'today' is about lead 3 and 'tomorrow' lead 4.")
        VAL = load_validation()
        vid = row.get("near_id", row.location_id)
        if "near_id" in row:
            st.info(f"{short_name(row.label)} was not one of the 411 test locations, so this shows the nearest one: "
                    f"**{short_name(row.near_label)}** ({row.near_km:.0f} km away).")
        v = VAL[VAL.location_id == vid].copy()
        v["target_date"] = pd.to_datetime(v.target_date)
        c1, c2 = st.columns(2)
        lead = c1.select_slider("Forecast lead (days after the latest data)", options=list(range(1, F.H + 1)), value=4)
        month = c2.selectbox("Month", list(range(1, 13)), index=6, format_func=lambda m: dt.date(2025, m, 1).strftime("%B"))
        s = v[(v.lead == lead) & (v.target_date.dt.month == month)].sort_values("target_date")
        if s.empty:
            st.warning("No 2025 data for this selection.")
        else:
            hit = ((s.obs_rain >= 2.5) & (s.pred_rain >= 2.5)).sum()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Rainy days (observed)", int((s.obs_rain >= 2.5).sum()))
            k2.metric("…correctly forecast", int(hit))
            k3.metric("Max temp error", f"±{(s.obs_tmax - s.pred_tmax).abs().mean():.1f} °C")
            k4.metric("Humidity error", f"±{(s.obs_hum - s.pred_hum).abs().mean():.1f} %")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=s.target_date, y=s.obs_rain, name="Observed rain", marker_color=BLUE))
            fig.add_trace(go.Scatter(x=s.target_date, y=s.pred_rain, name="Forecast rain", mode="lines+markers",
                                     line=dict(color=ORANGE, width=2), marker=dict(size=8)))
            fig.add_hline(y=64.5, line=dict(color="#888", dash="dash", width=1),
                          annotation_text="heavy rain ≥ 64.5 mm", annotation_position="top left")
            fig.update_layout(height=320, yaxis_title="mm/day", hovermode="x unified", margin=dict(t=30, b=10),
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")
            fig = go.Figure()
            for col, nm, color in [("tmax", "Max temp", ORANGE), ("tmin", "Min temp", BLUE)]:
                fig.add_trace(go.Scatter(x=s.target_date, y=s[f"obs_{col}"], name=f"{nm} observed", line=dict(color=color, width=2)))
                fig.add_trace(go.Scatter(x=s.target_date, y=s[f"pred_{col}"], name=f"{nm} forecast",
                                         line=dict(color=color, width=2, dash="dash")))
            fig.update_layout(height=290, yaxis_title="°C", hovermode="x unified", margin=dict(t=30, b=10),
                              legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, width="stretch")
        st.markdown(f"##### Accuracy over all {region} locations, 2025")
        sk = pd.DataFrame(F.cfg["regions"][region]["skill"])
        st.dataframe(pd.DataFrame({
            "Lead (days)": sk.lead, "Rain CSI ≥2.5 mm": sk.CSI_rain, "Persistence CSI": sk.pers_CSI_rain,
            "Moderate CSI ≥15.6": sk.CSI_mod, "Heavy CSI ≥64.5": sk.CSI_heavy,
            "Max temp error °C": sk.MAE_tmax, "Min temp error °C": sk.MAE_tmin,
            "Humidity error %": sk.MAE_hum, "Wind error m/s": sk.MAE_wind}).round(3), width="stretch", hide_index=True)
        st.caption("CSI = hits ÷ (hits + misses + false alarms); 1 is perfect. Persistence = 'tomorrow is like today'.")

    # ================================================================ ABOUT
    else:
        st.markdown(f"""
#### About {APP_NAME}
1. **Input:** the last 30 days of daily weather for the place from the **NASA POWER** API
   (temperature, dewpoint, humidity, pressure, wind, rainfall, max/min temperature, wind components, specific humidity),
   plus derived features (day–night temperature range, dewpoint depression, pressure and moisture change,
   recent rainfall totals, rainy days in the last week, days since the last rain, season).
2. **Model:** a two-layer **LSTM** neural network that forecasts 7 days at once. The **rainfall model** predicts the
   chance of rain ≥ 2.5 mm, ≥ 15.6 mm and ≥ 64.5 mm (IMD categories) plus the amount (an *ordinal* rainfall head);
   a second model predicts **max/min temperature, humidity and wind speed**.
3. **Training data:** 411 locations, 2015–2022 (training), 2023–2024 (validation), 2025 (test). Mainland rainfall is
   learned from **IMD gridded rain-gauge data**; 13 island locations use NASA rainfall.

**Why the forecast starts a few days back:** NASA POWER publishes daily data about 3 days late, so today is about
3 days ahead of the model's newest input. The Check page shows the accuracy for each lead day.

**Limitations:** heavy rain is hard to forecast days ahead (shown as a probability, not a warning); very heavy amounts
are usually under-estimated. Any place in India can be searched: the model then runs on NASA data for that exact
point, but its accuracy was measured on the 411 training locations. Research prototype, **not an official
forecast** — for warnings, follow the India Meteorological Department.

**Air quality (AQI):** not predicted by our LSTM. It comes from the Copernicus (CAMS) air-quality forecast via the
free Open-Meteo API, converted to India's National AQI (CPCB breakpoints for PM2.5, PM10, NO₂, SO₂, O₃ and CO).

Map tiles © OpenStreetMap contributors. Weather icons: Meteocons by Bas Milius (MIT licence).
""")