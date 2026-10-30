
from pathlib import Path
from urllib.parse import urljoin
import html
import re

import joblib
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="Ocean Climate AI",
    page_icon="🌊",
    layout="wide"
)

BASE = "https://coralreefwatch.noaa.gov/product/vs/data.php"
ROOT = Path(__file__).resolve().parent

MODEL_FILE = ROOT / "ocean_climate_models.joblib"
CLASS_CSV = ROOT / "classification_results.csv"
REG_CSV = ROOT / "regression_results.csv"
LOCAL_CSV = ROOT / "noaa_hawaii_training_data.csv"

FEATURES = ["SST", "SST_Anomaly", "HotSpot", "DHW"]

ALERTS = {
    0: "No Stress",
    1: "Bleaching Watch",
    2: "Bleaching Warning",
    3: "Bleaching Alert Level 1",
    4: "Bleaching Alert Level 2"
}


# ======================================
# LOAD MODELS AND NOAA DATA
# ======================================

@st.cache_resource
def load_models():
    return joblib.load(MODEL_FILE)


@st.cache_data(ttl=86400, show_spinner=False)
def station_directory():
    response = requests.get(
        BASE,
        timeout=25,
        headers={"User-Agent": "OceanClimateStudentProject/1.0"}
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    stations = {}

    for row in soup.select("tr"):
        cells = row.find_all("td")

        if len(cells) < 4:
            continue

        links = [
            a for a in row.find_all("a", href=True)
            if a.get("href", "").lower().split("?")[0].endswith(".txt")
        ]

        if not links:
            continue

        name = cells[0].get_text(" ", strip=True)

        if name and name not in stations:
            stations[name] = urljoin(BASE, links[-1]["href"])

    return dict(sorted(stations.items()))


@st.cache_data(ttl=86400, show_spinner=False)
def station_history(url):
    response = requests.get(
        url,
        timeout=35,
        headers={"User-Agent": "OceanClimateStudentProject/1.0"}
    )
    response.raise_for_status()

    records = []

    # NOAA regional time-series columns:
    # Year, Month, Day, SST_MIN, SST_MAX,
    # SST_90th, SSTA_90th, HotSpot_90th,
    # DHW_90th, BAA_7day_max

    for line in response.text.splitlines():
        fields = re.split(r"[\s,]+", line.strip())

        if len(fields) < 10:
            continue

        if not re.fullmatch(r"(19|20)\d{2}", fields[0]):
            continue

        try:
            year, month, day = [int(x) for x in fields[:3]]

            date = pd.Timestamp(
                year=year,
                month=month,
                day=day
            )

            vals = [float(x) for x in fields[3:10]]

            if not all(pd.notna(v) for v in vals):
                continue

            sst, anomaly, hotspot, dhw = vals[2:6]

            if not (
                0 <= sst <= 45
                and -20 <= anomaly <= 20
                and 0 <= hotspot <= 30
                and 0 <= dhw <= 100
            ):
                continue

            records.append({
                "Date": date,
                "SST": sst,
                "SST_Anomaly": anomaly,
                "HotSpot": hotspot,
                "DHW": dhw
            })

        except (ValueError, TypeError, OverflowError):
            continue

    if not records:
        raise ValueError(
            "No usable dated measurements were found "
            "in this NOAA time-series file."
        )

    return (
        pd.DataFrame(records)
        .drop_duplicates("Date")
        .sort_values("Date")
    )


@st.cache_data(show_spinner=False)
def read_csv(path):
    return pd.read_csv(path)


# ======================================
# THEME
# ======================================

if "dark" not in st.session_state:
    st.session_state.dark = True

dark = st.session_state.dark

bg = "#09171d" if dark else "#f3f8f9"
panel = "#142832" if dark else "#ffffff"
fg = "#f1f7f8" if dark else "#183641"
muted = "#b6c8cc" if dark else "#516b75"
edge = "#31505c" if dark else "#d9e6ea"
side = "#10232d" if dark else "#e8f2f4"

st.markdown(
    f"""
<style>
.stApp {{
    background: {bg};
    color: {fg};
}}

[data-testid="stSidebar"] {{
    background: {side};
    border-right: 1px solid {edge};
}}

[data-testid="stSidebar"] * {{
    color: {fg} !important;
}}

.block-container {{
    max-width: 1320px;
    padding-top: 1.6rem;
}}

h1, h2, h3, p, label {{
    color: {fg};
}}

[data-testid="stCaptionContainer"] {{
    color: {muted} !important;
}}

[data-testid="stForm"],
[data-testid="stMetric"] {{
    background: {panel};
    border: 1px solid {edge};
    border-radius: 16px;
    padding: 16px;
}}

[data-testid="stNumberInput"] input {{
    background: {panel} !important;
    color: {fg} !important;
}}

[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {{
    color: {fg} !important;
}}

[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] * {{
    color: {fg} !important;
}}

div.stFormSubmitButton button {{
    background: #087f83 !important;
    color: white !important;
    border: 0;
    border-radius: 11px;
    font-weight: 700;
    min-height: 48px;
    transition: transform .2s;
}}

div.stFormSubmitButton button:hover {{
    transform: translateY(-2px);
}}

.hero {{
    background: linear-gradient(115deg, #06414f, #087d80);
    padding: 30px 36px;
    border-radius: 18px;
    color: white;
    margin-bottom: 20px;
    animation: appear .5s ease-out;
}}

.hero h1,
.hero p {{
    color: white !important;
}}

.result {{
    background: {panel};
    border: 1px solid {edge};
    border-left: 5px solid #0ba6a2;
    border-radius: 15px;
    padding: 23px;
    animation: appear .5s ease-out;
}}

.result h2 {{
    color: {fg} !important;
}}

@keyframes appear {{
    from {{
        opacity: 0;
        transform: translateY(12px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

@media (prefers-reduced-motion: reduce) {{
    * {{
        animation: none !important;
        transition: none !important;
    }}
}}
</style>
""",
    unsafe_allow_html=True
)


# ======================================
# SIDEBAR
# ======================================

with st.sidebar:
    st.markdown("### Ocean Climate AI")

    page = st.radio(
        "Navigation",
        [
            "Predict",
            "Model performance",
            "Data explorer",
            "About"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.toggle("Dark mode", key="dark")

    st.caption("NOAA Coral Reef Watch regional virtual stations")
    st.caption("Model trained on Main Hawaiian Islands data")


# ======================================
# HEADER
# ======================================

st.markdown(
    """
<div class="hero">
<div style="font-size:.75rem;letter-spacing:.16em;color:#b8f4ed">
NOAA CORAL REEF WATCH · OCEAN ANALYTICS
</div>

<h1>Ocean Climate AI</h1>

<p>
Explore real regional ocean measurements and
estimate next-day heat-stress risk.
</p>
</div>
""",
    unsafe_allow_html=True
)


# ======================================
# MODEL INITIALIZATION
# ======================================

if not MODEL_FILE.exists():
    st.error(
        "Missing ocean_climate_models.joblib. "
        "Place it beside app.py."
    )
    st.stop()

try:
    saved = load_models()

    classifier = saved["classifier"]
    regressor = saved["regressor"]
    model_features = list(saved["features"])

except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()


# ======================================
# PREDICTION PAGE
# ======================================

if page == "Predict":

    st.subheader("Station-wise next-day heat-stress prediction")

    try:
        directory = station_directory()

    except Exception as exc:
        directory = {}
        st.warning(
            f"Could not load NOAA station directory: {exc}"
        )

    if directory:

        names = list(directory)

        initial = (
            names.index("Main Hawaiian Islands")
            if "Main Hawaiian Islands" in names
            else 0
        )

        station = st.selectbox(
            "Select NOAA station",
            names,
            index=initial
        )

        try:
            history = station_history(directory[station])
            latest = history.iloc[-1]

            st.caption(
                f"NOAA regional station · "
                f"latest available data: {latest['Date']:%d %b %Y}"
            )

            st.link_button(
                "View original NOAA station data",
                directory[station]
            )

        except Exception as exc:
            history = None
            latest = None

            st.warning(
                f"Could not read station measurements: {exc}"
            )

    else:
        station = "Main Hawaiian Islands"
        history = None
        latest = None

    # Measurement source

    options = ["Manual measurements"]

    if latest is not None:
        options.insert(
            0,
            "Use station's latest recorded measurements"
        )

    source = st.radio(
        "Measurement source",
        options,
        horizontal=True
    )

    defaults = {
        "SST": 29.0,
        "SST_Anomaly": 1.2,
        "HotSpot": 1.5,
        "DHW": 5.0
    }

    if latest is not None and source.startswith("Use station"):
        defaults = {
            f: float(latest[f])
            for f in FEATURES
        }

    

    left, right = st.columns(
        [1.05, 1],
        gap="large"
    )

    # ==================================
    # INPUTS
    # ==================================

    with left:

        with st.form("forecast_form"):

            st.markdown("#### Ocean measurements")

            sst = st.number_input(
                "Sea surface temperature (°C)",
                min_value=0.0,
                max_value=45.0,
                value=float(
                    max(0, min(45, defaults["SST"]))
                ),
                step=0.1
            )

            anomaly = st.number_input(
                "SST anomaly (°C)",
                min_value=-20.0,
                max_value=20.0,
                value=float(
                    max(-20, min(20, defaults["SST_Anomaly"]))
                ),
                step=0.1
            )

            hotspot = st.number_input(
                "Coral bleaching HotSpot (°C)",
                min_value=0.0,
                max_value=30.0,
                value=float(
                    max(0, min(30, defaults["HotSpot"]))
                ),
                step=0.1
            )

            dhw = st.number_input(
                "Degree Heating Weeks (°C-weeks)",
                min_value=0.0,
                max_value=100.0,
                value=float(
                    max(0, min(100, defaults["DHW"]))
                ),
                step=0.1
            )

            submitted = st.form_submit_button(
                "Predict next-day risk",
                use_container_width=True
            )

        with st.expander("What do the measurements mean?"):
            st.markdown(
                """
**SST:** Sea surface temperature.

**Anomaly:** Difference from the usual temperature.

**HotSpot:** Excess thermal stress.

**DHW:** Accumulated thermal stress over recent weeks.
"""
            )

    # ==================================
    # MODEL PREDICTIONS
    # ==================================

    with right:

        st.markdown("#### Your forecast")

        if submitted:

            values = {
                "SST": sst,
                "SST_Anomaly": anomaly,
                "HotSpot": hotspot,
                "DHW": dhw
            }

            try:
                X = pd.DataFrame(
                    [[values[f] for f in model_features]],
                    columns=model_features
                )

                code = int(
                    classifier.predict(X)[0]
                )

                future_dhw = max(
                    0.0,
                    float(regressor.predict(X)[0])
                )

                label = ALERTS.get(
                    code,
                    f"Alert code {code}"
                )

                risk = (
                    "HIGH ALERT"
                    if code >= 3
                    else "LOWER ALERT"
                )

                st.markdown(
                    f"""
<div class="result">
<div style="opacity:.7">
PREDICTED NEXT-DAY ALERT
</div>

<h2>{html.escape(label)}</h2>
<strong>{risk}</strong>
</div>
""",
                    unsafe_allow_html=True
                )

                st.metric(
                    "Predicted next-day DHW",
                    f"{future_dhw:.2f} °C-weeks"
                )

                if hasattr(classifier, "predict_proba"):

                    probabilities = pd.DataFrame({
                        "Alert": [
                            ALERTS.get(int(c), str(c))
                            for c in classifier.classes_
                        ],
                        "Probability": classifier.predict_proba(X)[0]
                    }).set_index("Alert")

                    st.markdown("**Model probabilities**")

                    st.bar_chart(
                        probabilities,
                        horizontal=True,
                        color="#0a9396"
                    )

            except Exception as exc:
                st.error(
                    f"Prediction failed: {exc}"
                )

        else:
            st.info(
                "Choose a station, review its measurements, "
                "then click Predict next-day risk."
            )

    # Historical station measurements

    if history is not None:

        with st.expander(
            f"Historical measurements — {station}"
        ):

            st.line_chart(
                history.set_index("Date")[
                    ["SST", "DHW"]
                ].tail(365)
            )

            st.dataframe(
                history.tail(20),
                use_container_width=True,
                hide_index=True
            )

    st.caption(
        "The app uses the selected station's NOAA "
        "measurements, not its existing alert label. "
        "The saved prediction model was trained "
        "only on Main Hawaiian Islands history."
    )


# ======================================
# MODEL PERFORMANCE PAGE
# ======================================

elif page == "Model performance":

    st.subheader("Model comparison")

    for filename, title, metric in [
        (CLASS_CSV, "Classification", "Macro F1"),
        (REG_CSV, "Regression", "RMSE")
    ]:

        st.markdown(f"#### {title}")

        if filename.exists():

            frame = read_csv(str(filename))

            st.dataframe(
                frame,
                use_container_width=True,
                hide_index=True
            )

            if "Model" in frame and metric in frame:

                st.bar_chart(
                    frame.set_index("Model")[metric],
                    color="#0a9396"
                )

        else:
            st.warning(f"Missing {filename.name}")

    st.caption(
        "Metrics reflect the original training station, "
        "not performance on other stations."
    )


# ======================================
# DATA EXPLORER PAGE
# ======================================

elif page == "Data explorer":

    st.subheader("Historical training dataset")

    if LOCAL_CSV.exists():

        frame = read_csv(str(LOCAL_CSV))

        st.metric(
            "Training rows",
            f"{len(frame):,}"
        )

        st.dataframe(
            frame.tail(100),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download CSV",
            frame.to_csv(index=False),
            file_name=LOCAL_CSV.name,
            mime="text/csv"
        )

    else:
        st.info("Training CSV not found.")


# ======================================
# ABOUT PAGE
# ======================================

else:

    st.subheader("About Ocean Climate AI")

    st.markdown(
        """
The app discovers NOAA regional virtual stations
and downloads their historical measurements.

**Inputs:** SST, SST anomaly, HotSpot, and DHW.

**Classification:** Next-day NOAA-style alert category.

**Regression:** Next-day Degree Heating Weeks.

**Models:** Saved scikit-learn classifier and regressor.

**Important limitation:** The saved models were
trained on Main Hawaiian Islands observations.

Predictions for other stations are experimental
and have not been validated.

A properly validated multi-station model would
require retraining on multiple stations and
testing on held-out regions and later dates.

[NOAA Coral Reef Watch](https://coralreefwatch.noaa.gov/)
"""
    )
