import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Motor Insurance Risk Predictor", 
    page_icon="🚗", layout="wide"
)

@st.cache_resource
def load_models():
    lr  = joblib.load("../models/lr_balanced.pkl")
    rf  = joblib.load("../models/rf_balanced.pkl")
    xgb = joblib.load("../models/xgb.pkl")
    features = joblib.load("../models/feature_names.pkl")
    return lr, rf, xgb, features

lr_model, rf_model, xgb_model, feature_names = load_models()

THRESHOLDS = {
    "Logistic Regression": 0.5, 
    "Random Forest": 0.3, 
    "XGBoost": 0.3
}

FEATURE_LABELS = {
    "speed_limit": "Speed limit (mph)",
    "road_type": "Road type",
    "light_conditions": "Light conditions",
    "weather_conditions": "Weather conditions",
    "road_surface_conditions": "Road surface",
    "urban_or_rural_area": "Area type",
    "hour": "Time of day",
    "age_of_driver": "Age of driver",
    "vehicle_type": "Vehicle type",
    "number_of_vehicles": "Number of vehicles involved",
    "police_force": "Police force area (region)",
    "vehicle_manoeuvre": "Vehicle manoeuvre",
    "vehicle_manoeuvre_historic": "Vehicle manoeuvre (historic)",
    "first_point_of_impact": "First point of impact",
    "vehicle_leaving_carriageway": "Vehicle leaving carriageway",
    "driver_imd_decile": "Driver deprivation index",
    "trunk_road_flag": "Trunk road flag",
    "engine_capacity_cc": "Engine size (cc)",
    "day_of_week": "Day of week",
    "month": "Month of year",
}

# ── Header ──────────────────────────────────────────────────
st.title("🚗 Motor Insurance Risk Predictor")
st.markdown(
    "This research tool explores how machine learning can predict "
    "collision risk for young UK drivers (aged 17–25) — and whether "
    "it does so **fairly** across different groups."
)

with st.expander("ℹ️ How does this work?", expanded=False):
    st.markdown("""
    **What this app does:**
    - Takes road and vehicle conditions as inputs
    - Uses a machine learning model trained on 13,375 real UK road accident records (DfT 2024)
    - Predicts whether those conditions are associated with **high or low collision risk**
    - Explains **why** using SHAP (a technique that shows which factors mattered most)
    - Shows whether the model treats male/female and younger/older drivers **fairly**
    
    **What this app does NOT do:**
    - It does not predict whether YOU personally will have an accident
    - It is not a real insurance pricing tool
    - Individual predictions are illustrative only — 40 of 49 features use default values
    
    **Why does this matter for insurance?**
    Young drivers in the UK pay significantly higher premiums than older drivers, 
    largely justified by age alone. This tool investigates whether road conditions 
    and vehicle type actually explain risk better than age — and whether ML models 
    used for pricing decisions treat all drivers fairly.
    """)

st.divider()

# ── Sidebar ─────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Selection")
st.sidebar.markdown("Choose which ML model to use for the prediction.")
model_choice = st.sidebar.selectbox(
    "Model",
    ["Logistic Regression", "Random Forest", "XGBoost"],
    help="Each model has different accuracy and fairness characteristics."
)

model_info = {
    "Logistic Regression": {
        "desc": "A simple, transparent model. Best choice if fairness by sex matters most.",
        "accuracy": "57.8%", "recall": "59%", "roc": "0.629"
    },
    "Random Forest": {
        "desc": "An ensemble of decision trees. Best at catching high-risk drivers.",
        "accuracy": "65.0%", "recall": "61.4%", "roc": "0.689"
    },
    "XGBoost": {
        "desc": "Gradient boosted trees. Best overall accuracy and precision.",
        "accuracy": "69.1%", "recall": "51.8%", "roc": "0.689"
    }
}

info = model_info[model_choice]
st.sidebar.info(info["desc"])
st.sidebar.markdown(f"""
| Metric | Value |
|--------|-------|
| Accuracy | {info['accuracy']} |
| Recall | {info['recall']} |
| ROC-AUC | {info['roc']} |
| Threshold | {THRESHOLDS[model_choice]} |
""")

st.sidebar.divider()
st.sidebar.markdown("### 👤 Your demographic profile")
st.sidebar.markdown(
    "Your sex and age group are **never used by the model** to predict risk. "
    "They are only used in the **fairness audit** section below the prediction, "
    "to show whether the model treats your demographic group fairly."
)
sex_input = st.sidebar.selectbox(
    "Your sex", [1, 2],
    format_func=lambda x: "Male" if x == 1 else "Female"
)
age_band_input = st.sidebar.selectbox(
    "Your age group", [4, 5],
    format_func=lambda x: "16–20 (newer driver)" if x == 4 else "21–25 (more experienced)"
)

# ── Input form ───────────────────────────────────────────────
st.subheader("Step 1: Enter the driving conditions")
st.markdown(
    "Think of a recent journey you made — or imagine a typical driving scenario. "
    "Fill in the conditions below and click **Predict Risk** to see the result."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Road conditions**")
    speed_limit = st.selectbox(
        "Speed limit on the road (mph)", 
        [20, 30, 40, 50, 60, 70], index=1
    )
    road_type = st.selectbox(
        "Type of road", [6, 1, 3, 2, 7, 12, 9],
        format_func=lambda x: {
            6:"Single carriageway (typical road)",
            1:"Roundabout",
            3:"Dual carriageway (A-road/motorway style)",
            2:"One way street",
            7:"Slip road",
            12:"One way/slip road",
            9:"Unknown"
        }.get(x, str(x))
    )
    urban_or_rural_area = st.selectbox(
        "Where are you driving?", [1, 2],
        format_func=lambda x: "Urban (town or city) 🏙️" if x == 1 else "Rural (countryside) 🌳"
    )

with col2:
    st.markdown("**Conditions at the time**")
    light_conditions = st.selectbox(
        "Lighting", [1, 4, 5, 6, 7],
        format_func=lambda x: {
            1:"Daytime ☀️",
            4:"Night — street lights on 🌙",
            5:"Night — street lights off 🌑",
            6:"Night — no street lighting 🌑",
            7:"Night — lighting unknown 🌑"
        }.get(x, str(x))
    )
    weather_conditions = st.selectbox(
        "Weather", [1, 2, 3, 4, 5, 6, 7, 8, 9],
        format_func=lambda x: {
            1:"Clear, no wind ☀️",
            2:"Raining 🌧️",
            3:"Snowing ❄️",
            4:"Clear + strong winds 💨",
            5:"Raining + strong winds ⛈️",
            6:"Snowing + strong winds 🌨️",
            7:"Fog or mist 🌫️",
            8:"Other",
            9:"Unknown"
        }.get(x, str(x))
    )
    road_surface_conditions = st.selectbox(
        "Road surface", [1, 2, 3, 4, 5],
        format_func=lambda x: {
            1:"Dry 🟫",
            2:"Wet or damp 💧",
            3:"Snow ❄️",
            4:"Frost or ice 🧊",
            5:"Flood 🌊"
        }.get(x, str(x))
    )

with col3:
    st.markdown("**About the driver and vehicle**")
    
    # Link age_of_driver to age_band selection
    if age_band_input == 4:
        age_default = 18
        age_min, age_max = 16, 20
    else:
        age_default = 23
        age_min, age_max = 21, 25
    
    age_of_driver = st.slider(
        "Age of driver", 
        age_min, age_max, age_default,
        help="Age is used by the model but ranks 13th out of 49 features — road conditions matter more!"
    )
    hour = st.select_slider(
        "Approximate time of day",
        options=list(range(24)),
        value=8,
        format_func=lambda x: {
            0:"Midnight 🌙", 1:"1am 🌙", 2:"2am 🌙", 3:"3am 🌙",
            4:"4am 🌅", 5:"5am 🌅", 6:"6am 🌅", 7:"7am ☀️",
            8:"8am ☀️", 9:"9am ☀️", 10:"10am ☀️", 11:"11am ☀️",
            12:"Noon ☀️", 13:"1pm ☀️", 14:"2pm ☀️", 15:"3pm ☀️",
            16:"4pm 🌇", 17:"5pm 🌇", 18:"6pm 🌇", 19:"7pm 🌆",
            20:"8pm 🌃", 21:"9pm 🌃", 22:"10pm 🌃", 23:"11pm 🌙"
        }.get(x, str(x)),
        help="Time of day matters — late night driving is associated with higher risk."
    )
    vehicle_type = st.selectbox(
        "Vehicle type", [9,2,3,4,5,8,10,11,19,22,23,97,98],
        format_func=lambda x: {
            9:"Car 🚗",
            2:"Motorcycle (small, up to 50cc) 🏍️",
            3:"Motorcycle (up to 125cc) 🏍️",
            4:"Motorcycle (over 125cc) 🏍️",
            5:"Motorcycle (over 500cc) 🏍️",
            8:"Taxi 🚕",
            10:"Minibus 🚐",
            11:"Bus or coach 🚌",
            19:"Van or small goods vehicle 🚚",
            22:"Mobility scooter 🛵",
            23:"Electric motorcycle ⚡",
            97:"Motorcycle (size unknown) 🏍️",
            98:"Other"
        }.get(x, str(x))
    )

# ── Build feature vector ────────────────────────────────────
def build_input_vector():
    defaults = {
        "collision_year_x": 2024,
        "vehicle_type": vehicle_type,
        "towing_and_articulation": 0,
        "vehicle_manoeuvre_historic": -1,
        "vehicle_manoeuvre": 19,
        "vehicle_direction_from": 5,
        "vehicle_direction_to": 1,
        "vehicle_location_restricted_lane": 0,
        "junction_location": 0,
        "skidding_and_overturning": 0,
        "hit_object_in_carriageway": 0,
        "vehicle_leaving_carriageway": 0,
        "hit_object_off_carriageway": 0,
        "first_point_of_impact": 1,
        "vehicle_left_hand_drive": 1,
        "journey_purpose_of_driver_historic": -1,
        "journey_purpose_of_driver": 6,
        "age_of_driver": age_of_driver,
        "engine_capacity_cc": -1,
        "propulsion_code": 1,
        "age_of_vehicle": -1,
        "driver_imd_decile": 1,
        "driver_distance_banding": 1,
        "police_force": 20,
        "number_of_vehicles": 2,
        "day_of_week": 2,
        "local_authority_district": -1,
        "first_road_class": 3,
        "first_road_number": 0,
        "road_type": road_type,
        "speed_limit": speed_limit,
        "junction_detail_historic": -1,
        "junction_detail": 0,
        "junction_control": 4,
        "second_road_class": 6,
        "second_road_number": 0,
        "pedestrian_crossing_human_control_historic": -1,
        "pedestrian_crossing_physical_facilities_historic": -1,
        "pedestrian_crossing": 0,
        "light_conditions": light_conditions,
        "weather_conditions": weather_conditions,
        "road_surface_conditions": road_surface_conditions,
        "special_conditions_at_site": -1,
        "carriageway_hazards_historic": -1,
        "carriageway_hazards": 0,
        "urban_or_rural_area": urban_or_rural_area,
        "trunk_road_flag": 2,
        "hour": hour,
        "month": 6,
    }
    return pd.DataFrame([defaults])[feature_names]

# ── Predict ─────────────────────────────────────────────────
st.divider()
if st.button("Predict Risk", type="primary", use_container_width=True):

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

    # ── Result ──────────────────────────────────────────────
    st.divider()
    st.subheader("Step 2: Prediction Result")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        if prediction == 1:
            st.error(f"⚠️ **HIGH RISK** — {proba:.1%} probability")
            st.markdown(f"""
            The **{model_choice}** model predicts that driving under these 
            conditions is associated with **higher collision risk**.
            
            This means these specific road and vehicle conditions have 
            historically been linked to more serious or fatal accidents 
            in UK road safety data.
            
            > 💡 This does **not** mean you will have an accident — 
            it reflects the risk level of these particular conditions.
            """)
        else:
            st.success(f"✅ **LOW RISK** — {proba:.1%} probability")
            st.markdown(f"""
            The **{model_choice}** model predicts that driving under these 
            conditions is associated with **lower collision risk**.
            
            These road and vehicle conditions have historically been 
            linked to fewer serious or fatal accidents in UK road safety data.
            
            > 💡 This does **not** guarantee safety — always drive carefully 
            regardless of the prediction.
            """)

    with col_b:
        st.metric(
            label="Risk Score", 
            value=f"{proba:.1%}",
            help=f"Scores above {threshold:.0%} = HIGH RISK for {model_choice}"
        )
        st.progress(float(proba))
        
        # Visual risk gauge
        if proba < 0.3:
            st.success("🟢 Low")
        elif proba < 0.5:
            st.warning("🟡 Moderate")
        else:
            st.error("🔴 High")
        
        st.caption(
            f"The model flags drivers as HIGH RISK when their "
            f"probability score exceeds **{threshold:.0%}**. "
            f"This threshold was chosen to balance catching "
            f"genuinely high-risk drivers without too many false alarms."
        )

    # ── SHAP ────────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 Step 3: Why did the model make this prediction?")
    st.markdown("""
    The chart below shows the **top 10 factors** that influenced 
    this specific prediction — and by how much.
    
    | Colour | Meaning |
    |--------|---------|
    | 🔴 Red bar (pointing right) | This factor **increased** the risk score |
    | 🔵 Blue bar (pointing left) | This factor **decreased** the risk score |
    | Longer bar | Stronger influence on the prediction |
    """)

    try:
        if model_choice in ["Random Forest", "XGBoost"]:
            explainer = shap.TreeExplainer(
                xgb_model if model_choice == "XGBoost" else rf_model
            )
            shap_values = explainer(input_df)

            if len(shap_values.shape) == 3:
                sv = shap_values[:, :, 1]
            else:
                sv = shap_values

            sv.feature_names = [
                FEATURE_LABELS.get(f, f.replace("_", " ").title())
                for f in feature_names
            ]

            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(sv[0], max_display=10, show=False)
            plt.title(
                "Which factors pushed this prediction towards HIGH or LOW risk?",
                fontsize=11, pad=15
            )
            st.pyplot(fig)
            plt.close()

            st.caption(
                "Only the top 10 most influential factors are shown. "
                "The remaining features had smaller effects and are "
                "grouped as 'other features' at the bottom of the chart."
            )
        else:
            st.info(
                "The explanation chart is available for Random Forest "
                "and XGBoost. Switch the model in the sidebar to see it."
            )
    except Exception as e:
        st.warning(f"Explanation chart could not be generated: {e}")

    # ── Fairness Audit ───────────────────────────────────────
    st.divider()
    st.subheader("Step 4: Is this model fair?")

    sex_label = "Male" if sex_input == 1 else "Female"
    age_label = "16–20" if age_band_input == 4 else "21–25"

    st.markdown(f"""
    You identified as **{sex_label}, aged {age_label}**.
    
    Even if the model is accurate overall, it may still treat some 
    groups of drivers unfairly — for example, flagging one group as 
    high risk more often than another, even when their actual risk 
    is similar.
    
    This matters for insurance because **unfair models can lead to 
    unfair premiums** — where some groups pay more not because they 
    are genuinely riskier, but because the model is biased.
    
    The two fairness checks below measure this for the **{model_choice}** model:
    """)

    fairness_data = {
        "Logistic Regression": {
            "DP(sex)": 0.054, "EO(sex)": 0.040,
            "DP(age)": 0.147, "EO(age)": 0.175
        },
        "Random Forest": {
            "DP(sex)": 0.125, "EO(sex)": 0.118,
            "DP(age)": 0.133, "EO(age)": 0.166
        },
        "XGBoost": {
            "DP(sex)": 0.136, "EO(sex)": 0.124,
            "DP(age)": 0.089, "EO(age)": 0.149
        },
    }

    fd = fairness_data[model_choice]

    def fairness_card(value, question, green_msg, amber_msg, red_msg):
        if value < 0.05:
            icon = "🟢"
            level = "Fair"
            msg = green_msg
            colour = "success"
        elif value < 0.15:
            icon = "🟡"
            level = "Moderate disparity"
            msg = amber_msg
            colour = "warning"
        else:
            icon = "🔴"
            level = "Concerns about fairness"
            msg = red_msg
            colour = "error"

        st.markdown(f"**{question}**")
        if colour == "success":
            st.success(f"{icon} **{level}** : {msg}")
        elif colour == "warning":
            st.warning(f"{icon} **{level}** : {msg}")
        else:
            st.error(f"{icon} **{level}** : {msg}")
        st.caption(f"Fairness score: {value:.3f} (closer to 0 is fairer)")

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.markdown("### Male vs Female drivers")
        st.caption(
            "These checks compare how the model treats male and female drivers."
        )

        fairness_card(
            fd['DP(sex)'],
            "Are male and female drivers flagged as high risk at similar rates?",
            green_msg="Yes — the model flags male and female drivers at roughly equal rates. This is fair.",
            amber_msg=(
                f"Not quite — male drivers are flagged as high risk more often than "
                f"female drivers. In insurance terms, this could mean female drivers "
                f"are being over-charged relative to their actual risk."
            ),
            red_msg=(
                f"No — there is a significant difference in how often male vs female "
                f"drivers are flagged. This could contribute to unfair premium differences."
            )
        )

        fairness_card(
            fd['EO(sex)'],
            "Of drivers who are genuinely high risk, are they identified equally across sexes?",
            green_msg="Yes — the model catches truly high-risk male and female drivers at equal rates.",
            amber_msg=(
                "Not quite — the model is better at identifying high-risk drivers "
                "of one sex than the other. This means some genuinely risky drivers "
                "may be missed depending on their sex."
            ),
            red_msg=(
                "No — there is a big gap. The model misses genuinely high-risk "
                "drivers of one sex far more than the other. This is a serious "
                "fairness concern for insurance decisions."
            )
        )

    with col_f2:
        st.markdown("### Younger (16–20) vs older (21–25) drivers")
        st.caption(
            "These checks compare how the model treats the two young driver age groups."
        )

        fairness_card(
            fd['DP(age)'],
            "Are 16–20 and 21–25 year old drivers flagged as high risk at similar rates?",
            green_msg="Yes — both age groups are flagged at roughly equal rates.",
            amber_msg=(
                "Not quite — 16–20 year olds are flagged as high risk more often "
                "than 21–25 year olds. In insurance terms, this could mean newer "
                "drivers face higher premiums partly due to model bias rather than "
                "actual risk differences."
            ),
            red_msg=(
                "No — there is a significant gap between age groups. "
                "This could be contributing to the much higher insurance premiums "
                "that younger drivers face."
            )
        )

        fairness_card(
            fd['EO(age)'],
            "Of drivers who are genuinely high risk, are they identified equally across age groups?",
            green_msg="Yes — the model catches truly high-risk drivers equally across both age groups.",
            amber_msg=(
                "Not quite — the model identifies high-risk drivers in one age "
                "group better than the other. Some genuinely risky drivers in one "
                "age group may be systematically missed."
            ),
            red_msg=(
                "No — there is a large gap. The model is much better at catching "
                "high-risk drivers in one age group than the other. This is a "
                "serious concern if this model were used for real insurance pricing."
            )
        )

    st.divider()
    st.caption(
        "🟢 Score below 0.05 = Fair | "
        "🟡 0.05–0.15 = Moderate disparity | "
        "🔴 Above 0.15 = Concerns about fairness. "
        "Scores computed using Fairlearn on 2,675 held-out test drivers."
    )

    # ── Key Finding ─────────────────────────────────────────
    st.divider()
    st.subheader("Key research finding")
    st.markdown("""
    > **Age of driver ranked only 13th out of 49 features** in terms of 
    influence on predictions. Road environment features — the region, 
    number of vehicles involved, vehicle type, and speed limit — 
    matter far more than age alone.
    
    This directly challenges the common justification for charging 
    young drivers significantly higher insurance premiums based 
    primarily on age. If road conditions explain risk better than age, 
    then age-based pricing may be both **unfair and inaccurate**.
    """)

    st.info(
        "💡 Try changing the road conditions above and clicking Predict again "
        "to see how much the risk score changes — and compare this to "
        "changing just the age of the driver."
    )