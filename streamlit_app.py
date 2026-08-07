import streamlit as st
import pandas as pd
import joblib
import traceback
import os

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="AI-Based UCS Prediction",
    page_icon="🧱",
    layout="wide"
)

# ==========================================================
# Title
# ==========================================================

st.title("🧱 AI-Based UCS Prediction of FA–SCBA Stabilized Expansive Clay")

st.markdown(
"""
This application predicts the **Unconfined Compressive Strength (UCS)** of
**Fly Ash (FA)–Sugarcane Bagasse Ash (SCBA) stabilized expansive clay**
using multiple Artificial Intelligence (AI) models.

Enter the mix proportions and soil properties below, then click **Predict UCS**.
"""
)

st.markdown("---")

# ==========================================================
# Required Files
# ==========================================================

required_files = [
    "RF_model.pkl",
    "ET_model.pkl",
    "HGBR_model.pkl",
    "SVR_model.pkl",
    "ANN_model.pkl",
    "Spline_model.pkl",
    "Spline_scaler.pkl"      # Change if your scaler has another name
]

missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    st.error("Missing files:")
    st.write(missing)
    st.stop()

# ==========================================================
# Load Scaler
# ==========================================================

try:
    scaler = joblib.load("Spline_scaler.pkl")
except Exception:
    st.error("Unable to load scaler.")
    st.code(traceback.format_exc())
    st.stop()

# ==========================================================
# Models
# ==========================================================

model_files = {
    "Random Forest": "RF_model.pkl",
    "Extra Trees": "ET_model.pkl",
    "HistGradient Boosting": "HGBR_model.pkl",
    "Support Vector Regression": "SVR_model.pkl",
    "Artificial Neural Network": "ANN_model.pkl",
    "Spline Regression": "Spline_model.pkl"
}

model_accuracy = {
    "Random Forest": 94.5,
    "Extra Trees": 96.2,
    "HistGradient Boosting": 93.8,
    "Support Vector Regression": 91.4,
    "Artificial Neural Network": 97.1,
    "Spline Regression": 90.6
}

models = {}

for name, path in model_files.items():

    try:
        models[name] = joblib.load(path)

    except Exception:

        st.error(f"Unable to load model : {name}")

        st.code(traceback.format_exc())

        st.stop()

# ==========================================================
# Mix Design
# ==========================================================

with st.expander("🧪 Mix Design", expanded=True):

    col1, col2, col3 = st.columns(3)

    with col1:
        FA = st.number_input(
            "Fly Ash (%)",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=0.5
        )

    with col2:
        SCBA = st.number_input(
            "SCBA (%)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.5
        )

    EC = 100 - (FA + SCBA)

    with col3:
        st.metric(
            "Expansive Clay (%)",
            f"{EC:.2f}"
        )

if EC < 0:
    st.error("Fly Ash (%) + SCBA (%) cannot exceed 100%.")
    st.stop()

# ==========================================================
# Soil Properties
# ==========================================================

with st.expander("🌍 Soil Properties", expanded=True):

    c1, c2, c3 = st.columns(3)

    with c1:

        Gs = st.number_input(
            "Specific Gravity (Gs)",
            value=2.65
        )

        PI = st.number_input(
            "Plasticity Index (PI)",
            value=15.0
        )

        FSI = st.number_input(
            "Free Swell Index (FSI)",
            value=30.0
        )

    with c2:

        MDUW = st.number_input(
            "Maximum Dry Unit Weight (kN/m³)",
            value=16.0
        )

        OMC = st.number_input(
            "Optimum Moisture Content (%)",
            value=20.0
        )

    with c3:

        UPV = st.number_input(
            "Ultrasonic Pulse Velocity (m/s)",
            value=800.0
        )

        CuringDays = st.number_input(
            "Curing Days",
            min_value=1,
            value=28
        )

st.markdown("")

# ==========================================================
# Predict Button
# ==========================================================

predict = st.button(
    "🚀 Predict UCS",
    use_container_width=True
)

# ==========================================================
# Prediction
# ==========================================================

if predict:

    try:

        input_data = pd.DataFrame(
            [[
                FA,
                SCBA,
                EC,
                Gs,
                PI,
                FSI,
                MDUW,
                OMC,
                UPV,
                CuringDays
            ]],
            columns=[
                "FA",
                "SCBA",
                "EC",
                "Gs",
                "PI",
                "FSI",
                "MDUW",
                "OMC",
                "UPV",
                "CuringDays"
            ]
        )

        scaled = scaler.transform(input_data)

        results = []

        for name, model in models.items():

            prediction = float(model.predict(scaled)[0])

            results.append([
                name,
                model_accuracy[name],
                round(prediction, 2)
            ])

        results_df = pd.DataFrame(
            results,
            columns=[
                "Model",
                "Accuracy (%)",
                "Predicted UCS (kPa)"
            ]
        )

        st.markdown("---")
        st.subheader("📊 Prediction Results")

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        best = results_df.loc[
            results_df["Accuracy (%)"].idxmax()
        ]

        st.markdown("### 🏆 Best Prediction")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Best Model",
                best["Model"]
            )

        with col2:

            st.metric(
                "Model Accuracy",
                f"{best['Accuracy (%)']} %"
            )

        with col3:

            st.metric(
                "Predicted UCS",
                f"{best['Predicted UCS (kPa)']:.2f} kPa"
            )

        st.success(
            "Prediction completed successfully."
        )

        csv = input_data.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Input Data",
            data=csv,
            file_name="UCS_prediction_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    except Exception:

        st.error("Prediction failed.")

        st.code(traceback.format_exc())

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")

st.caption(
    "AI-Based UCS Prediction Tool | "
    "Fly Ash–SCBA Stabilized Expansive Clay | "
    "For Research Purposes Only"
)
