import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import traceback
import os

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="UCS Prediction FA-SCBA-EC",
    page_icon="🧱",
    layout="wide"
)

st.title("AI-Based UCS Prediction of FA–SCBA Stabilized Expansive Clay")

# ---------------------------------------------------
# Check Required Files
# ---------------------------------------------------
required_files = [
    "RF_model.pkl",
    "ET_model.pkl",
    "HGBR_model.pkl",
    "SVR_model.pkl",
    "ANN_model.pkl",
    "Spline_model.pkl",
    "Spline_scaler.pkl"      # Change to scaler.pkl if applicable
]

missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error("The following required files are missing:")
    st.write(missing_files)
    st.stop()

# ---------------------------------------------------
# Load Scaler
# ---------------------------------------------------
try:
    scaler = joblib.load("Spline_scaler.pkl")      # Replace with scaler.pkl if needed
    st.success("Scaler loaded successfully.")
except Exception:
    st.error("Unable to load scaler.")
    st.code(traceback.format_exc())
    st.stop()

# ---------------------------------------------------
# Model Dictionary
# ---------------------------------------------------
model_files = {
    "RF": "RF_model.pkl",
    "ET": "ET_model.pkl",
    "HGBR": "HGBR_model.pkl",
    "SVR": "SVR_model.pkl",
    "ANN": "ANN_model.pkl",
    "Spline": "Spline_model.pkl"
}

model_accuracy = {
    "RF": 94.5,
    "ET": 96.2,
    "HGBR": 93.8,
    "SVR": 91.4,
    "ANN": 97.1,
    "Spline": 90.6
}

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------
models = {}

st.subheader("Loading AI Models")

for name, path in model_files.items():

    try:
        models[name] = joblib.load(path)
        st.success(f"✓ {name} model loaded.")

    except Exception:

        st.error(f"Failed to load {name}")

        st.code(traceback.format_exc())

        st.stop()

# ---------------------------------------------------
# User Inputs
# ---------------------------------------------------
st.subheader("Mix Design")

col1, col2 = st.columns(2)

with col1:
    FA = st.number_input("Fly Ash (%)", value=15.0)

with col2:
    SCBA = st.number_input("SCBA (%)", value=5.0)

EC = 100 - (FA + SCBA)

st.write(f"### Expansive Clay (%) = {EC:.2f}")

if EC < 0:
    st.error("Fly Ash + SCBA cannot exceed 100%.")
    st.stop()

# ---------------------------------------------------
# Soil Properties
# ---------------------------------------------------
st.subheader("Soil Properties")

col1, col2, col3 = st.columns(3)

with col1:
    Gs = st.number_input("Specific Gravity (Gs)", value=2.65)
    PI = st.number_input("Plasticity Index (PI)", value=15.0)

with col2:
    FSI = st.number_input("Free Swell Index (FSI)", value=30.0)
    MDUW = st.number_input("Maximum Dry Unit Weight", value=16.0)

with col3:
    OMC = st.number_input("Optimum Moisture Content (%)", value=20.0)
    UPV = st.number_input("Ultrasonic Pulse Velocity", value=800.0)

CuringDays = st.number_input(
    "Curing Days",
    min_value=1,
    value=28
)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button("Predict UCS"):

    try:

        input_df = pd.DataFrame(
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

        scaled_input = scaler.transform(input_df)

        results = []

        for name, model in models.items():

            prediction = model.predict(scaled_input)[0]

            results.append([
                name,
                model_accuracy[name],
                round(float(prediction), 2)
            ])

        results_df = pd.DataFrame(
            results,
            columns=[
                "Model",
                "Accuracy (%)",
                "Predicted UCS (kPa)"
            ]
        )

        st.subheader("Prediction Results")

        st.dataframe(results_df, use_container_width=True)

        best = results_df.loc[
            results_df["Accuracy (%)"].idxmax()
        ]

        st.success(
            f"""
            Best Model : {best['Model']}

            Reported Accuracy : {best['Accuracy (%)']} %

            Predicted UCS : {best['Predicted UCS (kPa)']} kPa
            """
        )

        # -------------------------
        # Plot
        # -------------------------
        fig, ax = plt.subplots(figsize=(8,5))

        ax.bar(
            results_df["Model"],
            results_df["Predicted UCS (kPa)"]
        )

        ax.set_xlabel("Model")
        ax.set_ylabel("Predicted UCS (kPa)")
        ax.set_title("Comparison of AI Models")

        st.pyplot(fig)

        # -------------------------
        # Download CSV
        # -------------------------
        report = input_df.copy()

        csv = report.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Input Data",
            data=csv,
            file_name="UCS_prediction_report.csv",
            mime="text/csv"
        )

    except Exception:

        st.error("Prediction failed.")

        st.code(traceback.format_exc())
