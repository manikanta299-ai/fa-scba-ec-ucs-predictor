import streamlit as st
import pandas as pd
import joblib
import traceback
import os

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(page_title="AI-Based UCS Prediction", page_icon="🧱", layout="wide")

st.title("🧱 AI-Based UCS Prediction of FA–SCBA Stabilized Expansive Clay")
st.markdown(
"""
This application estimates the **Unconfined Compressive Strength (UCS)** of
**Fly Ash (FA)–Sugarcane Bagasse Ash (SCBA) stabilized expansive clay** from
measured specimen properties, using an **Artificial Neural Network (ANN)** model.

The ANN was the best-generalizing model in the study
(nested leave-one-mixture-out R² = 0.875). Enter the mix proportions and
measured properties below, then click **Predict UCS**.
"""
)
st.markdown("---")

# ==========================================================
# Load ANN model (pipeline with embedded scaler)
# ==========================================================
if not os.path.exists("ANN_model.pkl"):
    st.error("Missing file: ANN_model.pkl"); st.stop()
try:
    ann = joblib.load("ANN_model.pkl")
except Exception:
    st.error("Unable to load ANN model."); st.code(traceback.format_exc()); st.stop()

ANN_R2 = 0.875
UCS_MIN, UCS_MAX = 34.0, 457.0
COLS = ["FA","SCBA","EC","Gs","PI","FSI","MDUW","OMC","UPV","CuringDays"]

# ==========================================================
# Mix Design
# ==========================================================
with st.expander("🧪 Mix Design", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        FA = st.number_input("Fly Ash (%)", 0.0, 100.0, 15.0, 0.5)
    with col2:
        SCBA = st.number_input("SCBA (%)", 0.0, 100.0, 5.0, 0.5)
    EC = 100 - (FA + SCBA)
    with col3:
        st.metric("Expansive Clay (%)", f"{EC:.2f}")
if EC < 0:
    st.error("Fly Ash (%) + SCBA (%) cannot exceed 100%."); st.stop()

# ==========================================================
# Measured Properties
# ==========================================================
with st.expander("🌍 Measured Specimen Properties", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        Gs = st.number_input("Specific Gravity (Gs)", value=2.65)
        PI = st.number_input("Plasticity Index (PI)", value=15.0)
        FSI = st.number_input("Free Swell Index (FSI)", value=30.0)
    with c2:
        MDUW = st.number_input("Maximum Dry Unit Weight (kN/m³)", value=16.0)
        OMC = st.number_input("Optimum Moisture Content (%)", value=20.0)
    with c3:
        UPV = st.number_input("Ultrasonic Pulse Velocity (m/s)", value=800.0)
        CuringDays = st.number_input("Curing Days", min_value=1, value=28)

st.caption("Note: PI, FSI, MDUW, OMC, and UPV are measured specimen properties; "
           "this tool is a post-characterization estimator, not a mix-design predictor.")
st.markdown("")

predict = st.button("🚀 Predict UCS", use_container_width=True)

# ==========================================================
# Prediction
# ==========================================================
if predict:
    try:
        input_data = pd.DataFrame(
            [[FA, SCBA, EC, Gs, PI, FSI, MDUW, OMC, UPV, CuringDays]], columns=COLS)

        pred = float(ann.predict(input_data)[0])   # pipeline scales internally
        pred = max(0.0, pred)                        # UCS cannot be negative

        st.markdown("---"); st.subheader("📊 Prediction Result")
        col1, col2 = st.columns(2)
        col1.metric("Predicted UCS (ANN)", f"{pred:.2f} kPa")
        col2.metric("Model Test R² (nested)", f"{ANN_R2:.3f}")

        if pred < UCS_MIN or pred > UCS_MAX:
            st.warning(f"The predicted UCS lies outside the tested range "
                       f"({UCS_MIN:.0f}–{UCS_MAX:.0f} kPa); treat this as extrapolation.")
        else:
            st.success("Prediction completed successfully.")

        csv = input_data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Input Data", csv,
                           "UCS_prediction_report.csv", "text/csv",
                           use_container_width=True)
    except Exception:
        st.error("Prediction failed."); st.code(traceback.format_exc())

st.markdown("---")
st.caption("AI-Based UCS Prediction Tool | Fly Ash–SCBA Stabilized Expansive Clay | "
           "ANN post-characterization estimator | For Research Purposes Only")
