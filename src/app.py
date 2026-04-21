import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ── Page config ────────────────────────────────────────────
st.set_page_config(page_title="Motor Insurance Risk Predictor", 
                   page_icon="🚗", layout="wide")

# ── Load models and features ───────────────────────────────
@st.cache_resource
def load_models():
    lr  = joblib.load("models/lr_balanced.pkl")
    rf  = joblib.load("models/rf_balanced.pkl")
    xgb = joblib.load("models/xgb.pkl")
    features = joblib.load("../models/feature_names.pkl")
    return lr, rf, xgb, features

lr_model, rf_model, xgb_model, feature_names = load_models()

THRESHOLDS = {"Logistic Regression": 0.5, 
              "Random Forest": 0.3, 
              "XGBoost": 0.3}

# ── Header ─────────────────────────────────────────────────
st.title("🚗 Motor Insurance Risk Predictor")
st.markdown("This tool predicts whether a young UK driver (aged 17–25) "
            "is **high risk** based on road and vehicle conditions. "
            "It also explains *why* using SHAP values and audits "
            "predictions for demographic fairness.")

st.info("""
⚠️ **Important notice for users**

This app is a **research demonstration tool**, not a real insurance pricing system.

- The underlying models were trained and evaluated on **complete UK road accident records** (DfT 2024), using all 49 features. All performance and fairness metrics reported are based on that full dataset.
- In this demo, only 9 key features are collected from the user. The remaining 40 features are set to default values, which means **individual predictions here are illustrative only** and should not be used for real insurance decisions.
- Protected attributes (sex, age band) entered in the sidebar are **never used by the model** to make predictions — they are only used for the fairness audit display.
""")
st.divider()

# ── Sidebar: model selection ───────────────────────────────
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Choose a model", 
    ["Logistic Regression", "Random Forest", "XGBoost"]
)
st.sidebar.markdown(f"**Decision threshold:** {THRESHOLDS[model_choice]}")
st.sidebar.divider()
st.sidebar.markdown("**Protected attributes** (for fairness audit only — "
                    "not used in prediction)")
sex_input = st.sidebar.selectbox(
    "Sex of driver", [1, 2], 
    format_func=lambda x: "Male" if x == 1 else "Female"
)
age_band_input = st.sidebar.selectbox(
    "Age band", [4, 5],
    format_func=lambda x: "16–20" if x == 4 else "21–25"
)
st.sidebar.caption(
    "Age bands follow DfT STATS19 coding: "
    "Band 4 = ages 16–20 (newest drivers), "
    "Band 5 = ages 21–25 (slightly more experienced). "
    "Both groups are within the young driver population this model was trained on."
)

# ── Input form ─────────────────────────────────────────────
st.subheader("🛣️ Road & Vehicle Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    speed_limit = st.selectbox("Speed limit (mph)", [20, 30, 40, 50, 60, 70], index=1)
    road_type = st.selectbox("Road type", [1,2,3,6,7,12,9],
                              format_func=lambda x: {
                                  1:"Roundabout", 2:"One way street",
                                  3:"Dual carriageway", 6:"Single carriageway",
                                  7:"Slip road", 12:"One way/slip road", 9:"Unknown"
                              }.get(x, str(x)))
    light_conditions = st.selectbox("Light conditions", [1,4,5,6,7],
                                     format_func=lambda x: {
                                         1:"Daylight", 4:"Darkness - lights lit",
                                         5:"Darkness - lights unlit",
                                         6:"Darkness - no lighting",
                                         7:"Darkness - unknown"
                                     }.get(x, str(x)))

with col2:
    weather_conditions = st.selectbox("Weather conditions", [1,2,3,4,5,6,7,8,9],
                                       format_func=lambda x: {
                                           1:"Fine no winds", 2:"Raining no winds",
                                           3:"Snowing no winds", 4:"Fine + winds",
                                           5:"Raining + winds", 6:"Snowing + winds",
                                           7:"Fog or mist", 8:"Other", 9:"Unknown"
                                       }.get(x, str(x)))
    road_surface_conditions = st.selectbox("Road surface", [1,2,3,4,5],
                                            format_func=lambda x: {
                                                1:"Dry", 2:"Wet or damp",
                                                3:"Snow", 4:"Frost or ice",
                                                5:"Flood"
                                            }.get(x, str(x)))
    urban_or_rural_area = st.selectbox(
        "Area type", [1, 2],
        format_func=lambda x: "Urban" if x == 1 else "Rural"
    )

with col3:
    hour = st.slider("Hour of day", 0, 23, 8)
    age_of_driver = st.slider("Age of driver", 17, 25, 19)
    vehicle_type = st.selectbox("Vehicle type",
                                 [1,2,3,4,5,8,9,10,11,16,17,18,19,20,21,22,23,97,98],
                                 format_func=lambda x: {
                                     1:"Pedal cycle", 2:"Motorcycle 50cc",
                                     3:"Motorcycle 125cc", 4:"Motorcycle over 125cc",
                                     5:"Motorcycle over 500cc", 8:"Taxi",
                                     9:"Car", 10:"Minibus", 11:"Bus",
                                     16:"Ridden horse", 17:"Agricultural vehicle",
                                     18:"Tram", 19:"Van/goods ≤3.5t",
                                     20:"Goods 3.5-7.5t", 21:"Goods >7.5t",
                                     22:"Mobility scooter", 23:"Electric motorcycle",
                                     97:"Motorcycle unknown", 98:"Other"
                                 }.get(x, str(x)), index=6)

# ── Build full feature vector with defaults ────────────────
def build_input_vector():
    defaults = {f: 0 for f in feature_names}
    defaults.update({
        "speed_limit": speed_limit,
        "road_type": road_type,
        "light_conditions": light_conditions,
        "weather_conditions": weather_conditions,
        "road_surface_conditions": road_surface_conditions,
        "urban_or_rural_area": urban_or_rural_area,
        "hour": hour,
        "age_of_driver": age_of_driver,
        "vehicle_type": vehicle_type,
        "number_of_vehicles": 1,
        "day_of_week": 2,
        "month": 6,
        "collision_year_x": 2024,
    })
    return pd.DataFrame([defaults])[feature_names]

# ── Predict button ─────────────────────────────────────────
st.divider()
if st.button("🔍 Predict Risk", type="primary", use_container_width=True):

    input_df = build_input_vector()
    model_map = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }
    model = model_map[model_choice]
    threshold = THRESHOLDS[model_choice]

    proba = model.predict_proba(input_df)[0][1]
    prediction = int(proba >= threshold)

    st.divider()
    st.subheader("📊 Prediction Result")

    col_a, col_b = st.columns(2)
    with col_a:
        if prediction == 1:
            st.error(f"⚠️ **HIGH RISK** (probability: {proba:.1%})")
        else:
            st.success(f"✅ **LOW RISK** (probability: {proba:.1%})")
        st.caption(f"Model: {model_choice} | Threshold: {threshold}")

    with col_b:
        st.metric("Risk Probability", f"{proba:.1%}")
        st.progress(float(proba))

    # ── SHAP explanation ───────────────────────────────────
    st.divider()
    st.subheader("🔍 Why this prediction? (SHAP Explanation)")
    st.markdown("The chart below shows which features pushed the prediction "
                "towards high risk (red) or low risk (blue).")

    try:
        if model_choice in ["Random Forest", "XGBoost"]:
            if model_choice == "XGBoost":
                explainer = shap.TreeExplainer(xgb_model)
            else:
                explainer = shap.TreeExplainer(rf_model)

            shap_values = explainer(input_df)

            fig, ax = plt.subplots(figsize=(10, 4))

            # Handle both 2D and 3D SHAP value arrays
            if len(shap_values.shape) == 3:
                # 3D array — binary classification, take class 1 (high risk)
                sv = shap_values[:, :, 1]
            else:
                # 2D array — already correct
                sv = shap_values

            shap.plots.waterfall(sv[0], max_display=10, show=False)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("SHAP waterfall plot is available for Random Forest "
                    "and XGBoost only.")
    except Exception as e:
        st.warning(f"SHAP plot could not be generated: {e}")

    # ── Fairness note ──────────────────────────────────────
    st.divider()
    st.subheader("⚖️ Fairness Audit Note")
    sex_label = "Male" if sex_input == 1 else "Female"
    age_label = "16–20" if age_band_input == 4 else "21–25"

    fairness_data = {
        "Logistic Regression": {"DP(sex)": 0.054, "EO(sex)": 0.040,
                                  "DP(age)": 0.147, "EO(age)": 0.175},
        "Random Forest":       {"DP(sex)": 0.125, "EO(sex)": 0.118,
                                  "DP(age)": 0.133, "EO(age)": 0.166},
        "XGBoost":             {"DP(sex)": 0.136, "EO(sex)": 0.124,
                                  "DP(age)": 0.089, "EO(age)": 0.149},
    }

    fd = fairness_data[model_choice]
    st.markdown(f"Driver profile entered: **{sex_label}, Age band {age_label}**")
    st.markdown(f"The **{model_choice}** model has the following fairness gaps:")

    def traffic_light(value, label_high, label_low):
        if value < 0.05:
            return f"🟢 Low disparity ({value:.3f}) — {label_low}"
        elif value < 0.15:
            return f"🟡 Moderate disparity ({value:.3f}) — some difference in treatment"
        else:
            return f"🔴 High disparity ({value:.3f}) — {label_high}"

    st.markdown("**Sex-based fairness:**")

    st.markdown("*Who gets flagged as high risk?*")
    st.info(traffic_light(
        fd['DP(sex)'],
        "model flags male drivers as high risk at a significantly higher rate than female drivers",
        "model flags male and female drivers at roughly equal rates"
    ))

    st.markdown("*Of truly high risk drivers, who gets caught?*")
    st.info(traffic_light(
        fd['EO(sex)'],
        "model catches truly high risk male drivers much better than female drivers",
        "model catches truly high risk drivers equally across both sexes"
    ))

    st.markdown("**Age-based fairness (16–20 vs 21–25):**")
    st.caption(
        "Age band 4 = drivers aged 16–20 (newest drivers), "
        "Age band 5 = drivers aged 21–25 (slightly more experienced). "
        "Both groups are within the young driver population this model was trained on."
    )

    st.markdown("*Who gets flagged as high risk?*")
    st.info(traffic_light(
        fd['DP(age)'],
        "model flags 16–20 year old drivers as high risk at a significantly higher rate than 21–25 year olds",
        "model flags 16–20 and 21–25 year old drivers at roughly equal rates"
    ))

    st.markdown("*Of truly high risk drivers, who gets caught?*")
    st.info(traffic_light(
        fd['EO(age)'],
        "model catches truly high risk 16–20 year olds much better than 21–25 year olds",
        "model catches truly high risk drivers equally across both age groups"
    ))

    st.caption(
        "🟢 Below 0.05 = fair — both groups treated equally  |  "
        "🟡 0.05–0.15 = moderate disparity  |  "
        "🔴 Above 0.15 = high disparity — one group treated less fairly. "
        "For both metrics, closer to 0 is better. "
        "Computed using Fairlearn on the held-out test set."
    )