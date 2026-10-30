import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Ocean Climate AI | Coral Stress Forecast", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")

MODEL_FILE = Path(__file__).with_name("ocean_climate_models.joblib")
CLASS_RESULTS = Path(__file__).with_name("classification_results.csv")
REG_RESULTS = Path(__file__).with_name("regression_results.csv")
TRAINING_DATA = Path(__file__).with_name("noaa_hawaii_training_data.csv")

ALERT_NAMES = {
    0: "No Stress",
    1: "Bleaching Watch",
    2: "Bleaching Warning",
    3: "Bleaching Alert Level 1",
    4: "Bleaching Alert Level 2",
}
FEATURES = ["SST", "SST_Anomaly", "HotSpot", "DHW"]

st.markdown("""
<style>
.stApp {background: #f5f8fa; color: #183441;}
[data-testid="stSidebar"] {background: #eaf2f5;}
.block-container {padding-top: 1.8rem; max-width: 1300px;}
.hero {background: linear-gradient(115deg, #063847, #08717a); color: #fff;
       padding: 2.2rem 2.5rem; border-radius: 18px; margin-bottom: 1.3rem;}
.hero h1 {color: #fff; margin: 0 0 .35rem; font-size: 2.45rem;}
.hero p {color: #e0f6f4; margin: 0;}
.kicker {color: #b9ece8; font-size: .76rem; letter-spacing: .16em; font-weight: 700;}
.result {background: #fff; border: 1px solid #d8e5ea; padding: 1.3rem;
         border-radius: 14px; margin-top: .8rem;}
.result h2 {margin: .3rem 0; color: #073e4c;}
.subtle {color: #5b7580; font-size: .88rem;}
[data-testid="stMetric"] {background: white; border: 1px solid #dce7eb;
                           padding: 14px; border-radius: 12px;}
div.stButton > button {background: #08717a; color: white; border: 0; width: 100%; font-weight: 700;}
div.stButton > button:hover {background: #075a63; color: white;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models(path):
    return joblib.load(path)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

st.sidebar.markdown("## Ocean Climate AI")
page = st.sidebar.radio("Explore", ["Predict", "Model performance", "Data explorer", "About"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("NOAA Coral Reef Watch · Historical Main Hawaiian Islands regional data")
st.sidebar.caption("Educational next-day heat-stress forecasting prototype")

st.markdown("""<div class="hero">
<div class="kicker">NOAA CORAL REEF WATCH · MACHINE LEARNING</div>
<h1>Ocean Climate AI</h1>
<p>Explore ocean heat-stress data and forecast the next day's NOAA-style alert using trained models.</p>
</div>""", unsafe_allow_html=True)

if not MODEL_FILE.exists():
    st.error("The trained model file is missing: `ocean_climate_models.joblib`.")
    st.info("Run the training notebook first. Download its generated `.joblib` and CSV files, and place them alongside `app.py`. Then restart Streamlit.")
    st.stop()

try:
    saved = load_models(str(MODEL_FILE))
    classifier = saved["classifier"]
    regressor = saved["regressor"]
    feature_names = saved["features"]
except Exception as exc:
    st.error(f"Could not load trained models: {exc}")
    st.stop()

if page == "Predict":
    st.subheader("Next-day ocean heat-stress prediction")
    st.caption("Enter today's environmental observations. The app sends these values to trained scikit-learn models; it does not look up a station's stored alert.")
    left, right = st.columns([1.03, 1], gap="large")
    with left:
        with st.form("forecast"):
            st.markdown("#### Today's ocean measurements")
            sst = st.number_input("Sea surface temperature (°C)", min_value=0.0, max_value=40.0, value=29.0, step=0.1)
            anomaly = st.number_input("SST anomaly (°C)", min_value=-15.0, max_value=15.0, value=1.2, step=0.1)
            hotspot = st.number_input("Coral bleaching HotSpot (°C)", min_value=0.0, max_value=20.0, value=1.5, step=0.1)
            dhw = st.number_input("Degree Heating Weeks (°C-weeks)", min_value=0.0, max_value=80.0, value=5.0, step=0.1)
            submitted = st.form_submit_button("Predict next-day risk", use_container_width=True)
        with st.expander("What do these measurements mean?"):
            st.markdown("**SST:** Current ocean surface temperature.  \n"
                        "**SST anomaly:** Difference from the expected climatological temperature.  \n"
                        "**HotSpot:** Temperature excess relevant to coral heat stress.  \n"
                        "**DHW:** Accumulated heat stress over recent weeks.")
    with right:
        st.markdown("#### Forecast results")
        if submitted:
            sample = pd.DataFrame([[sst, anomaly, hotspot, dhw]], columns=feature_names)
            try:
                predicted_class = int(classifier.predict(sample)[0])
                predicted_dhw = max(0.0, float(regressor.predict(sample)[0]))
                label = ALERT_NAMES.get(predicted_class, f"Alert code {predicted_class}")
                risk = "HIGH ALERT" if predicted_class >= 3 else "LOWER ALERT"
                st.markdown(f'<div class="result"><div class="subtle">PREDICTED NOAA-STYLE ALERT</div>'
                            f'<h2>{label}</h2><div class="subtle">Risk group: <b>{risk}</b></div></div>',
                            unsafe_allow_html=True)
                st.metric("Predicted next-day DHW", f"{predicted_dhw:.2f} °C-weeks")
                if hasattr(classifier, "predict_proba"):
                    probabilities = classifier.predict_proba(sample)[0]
                    classes = classifier.classes_
                    probability_df = pd.DataFrame({
                        "Alert category": [ALERT_NAMES.get(int(c), str(c)) for c in classes],
                        "Probability": probabilities
                    }).set_index("Alert category")
                    st.markdown("**Model class probabilities**")
                    st.bar_chart(probability_df)
                st.caption(f"Classifier: {saved.get('classification_model', 'trained model')} · "
                           f"Regressor: {saved.get('regression_model', 'trained model')}")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
        else:
            st.info("Enter measurements and click **Predict next-day risk** to run both trained models.")
    st.divider()
    st.warning("Educational prototype trained on historical Hawaiian regional observations. Predictions are next-day NOAA-style alert categories and DHW, not observed coral bleaching, coral mortality, or validated forecasts for other regions.")

elif page == "Model performance":
    st.subheader("Model comparison")
    st.caption("These metrics are generated by the training notebook on its chronological holdout set.")
    if CLASS_RESULTS.exists():
        cls = load_csv(str(CLASS_RESULTS))
        st.markdown("#### Classification models")
        st.dataframe(cls.style.format({c: "{:.3f}" for c in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"] if c in cls.columns}), hide_index=True, use_container_width=True)
        if "Macro F1" in cls:
            st.bar_chart(cls.set_index("Model")["Macro F1"])
    else:
        st.info("Run the notebook and copy `classification_results.csv` here to see classification results.")
    if REG_RESULTS.exists():
        reg = load_csv(str(REG_RESULTS))
        st.markdown("#### Regression models")
        st.dataframe(reg.style.format({c: "{:.3f}" for c in ["MAE", "RMSE", "R2"] if c in reg.columns}), hide_index=True, use_container_width=True)
        if "RMSE" in reg:
            st.bar_chart(reg.set_index("Model")["RMSE"])
    else:
        st.info("Run the notebook and copy `regression_results.csv` here to see regression results.")
    st.caption("Model selection in the notebook uses the test results for comparison. For rigorous model selection, use a separate validation period and reserve the test period for final evaluation.")

elif page == "Data explorer":
    st.subheader("Historical NOAA training observations")
    if TRAINING_DATA.exists():
        df = load_csv(str(TRAINING_DATA))
        st.metric("Valid next-day observation pairs", f"{len(df):,}")
        if "Date" in df:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        available = [c for c in FEATURES if c in df]
        if "Date" in df and available:
            feature = st.selectbox("Plot measurement", available)
            chart_df = df.dropna(subset=["Date", feature]).set_index("Date")[[feature]]
            st.line_chart(chart_df)
        st.dataframe(df.tail(100), use_container_width=True, hide_index=True)
        st.download_button("Download training data (CSV)", df.to_csv(index=False), file_name="noaa_hawaii_training_data.csv", mime="text/csv")
    else:
        st.info("Copy `noaa_hawaii_training_data.csv` from the notebook to enable the data explorer.")

else:
    st.subheader("About this project")
    st.markdown("""
**Objective:** Demonstrate data cleaning, statistical inference, regression, classification,
and an interactive machine-learning interface using NOAA Coral Reef Watch regional observations.

**Classification target:** Tomorrow's NOAA-style heat-stress alert category (0–4).

**Regression target:** Tomorrow's Degree Heating Weeks (DHW).

**Input features:** Today's SST, SST anomaly, HotSpot, and DHW.

**Methods compared:** Linear Regression, Ridge, Lasso, Decision Tree Regressor,
Random Forest Regressor; Logistic Regression, Decision Tree, Random Forest, KNN, and SVM.

**Limitations:** Adjacent daily observations are correlated; results can be optimistic.
This is not a prediction of measured coral damage, and no claim of geographic
generalization is made.
""")
    st.markdown("**Data source:** [NOAA Coral Reef Watch](https://coralreefwatch.noaa.gov/)")
