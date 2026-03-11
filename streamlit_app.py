import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="UCS Prediction FA-SCBA-EC", page_icon="🧱", layout="wide")

st.title("AI-Based UCS Prediction of FA–SCBA Stabilized Expansive Clay")

scaler = joblib.load("scaler.pkl")

model_files = {
"RF":"RF_model.pkl",
"ET":"ET_model.pkl",
"HGBR":"HGBR_model.pkl",
"SVR":"SVR_model.pkl",
"ANN":"ANN_model.pkl",
"Spline":"Spline_model.pkl"
}

models = {name: joblib.load(path) for name, path in model_files.items()}

model_accuracy = {
"RF":94.5,
"ET":96.2,
"HGBR":93.8,
"SVR":91.4,
"ANN":97.1,
"Spline":90.6
}

st.subheader("Mix Design")

col1, col2 = st.columns(2)

with col1:
FA = st.number_input("Fly Ash (%)", value=15.0)

with col2:
SCBA = st.number_input("SCBA (%)", value=5.0)

EC = 100 - (FA + SCBA)

st.write(f"Expansive Clay (EC %) = {EC}")

if EC < 0:
st.error("FA + SCBA cannot exceed 100%")

st.subheader("Soil Properties")

col1, col2, col3 = st.columns(3)

with col1:
Gs = st.number_input("Specific Gravity (Gs)", value=2.65)
PI = st.number_input("Plasticity Index (PI)", value=15.0)

with col2:
FSI = st.number_input("Free Swell Index (FSI)", value=40.0)
MDUW = st.number_input("Maximum Dry Unit Weight", value=16.0)

with col3:
OMC = st.number_input("Optimum Moisture Content (%)", value=20.0)
UPV = st.number_input("Ultrasonic Pulse Velocity", value=2000.0)

CuringDays = st.number_input("Curing Days", value=28)

if st.button("Predict UCS"):

```
if EC >= 0:

    input_data = pd.DataFrame([[FA,SCBA,EC,Gs,PI,FSI,MDUW,OMC,UPV,CuringDays]],
    columns=["FA","SCBA","EC","Gs","PI","FSI","MDUW","OMC","UPV","CuringDays"])

    scaled = scaler.transform(input_data)

    predictions = []

    for name, model in models.items():

        pred = model.predict(scaled)[0]

        predictions.append([
            name,
            model_accuracy[name],
            round(pred,2)
        ])

    results_df = pd.DataFrame(
        predictions,
        columns=["Model","Accuracy (%)","Predicted UCS (kPa)"]
    )

    results_df = results_df.sort_values(by="Predicted UCS (kPa)", ascending=False)

    st.subheader("Model Prediction Results")

    st.dataframe(results_df)

    best_model = results_df.iloc[0]["Model"]
    best_value = results_df.iloc[0]["Predicted UCS (kPa)"]

    st.success(f"Best Model: {best_model} | Predicted UCS = {best_value} kPa")

    st.subheader("Predicted UCS Comparison")

    fig, ax = plt.subplots()

    ax.bar(results_df["Model"], results_df["Predicted UCS (kPa)"])

    ax.set_xlabel("Model")
    ax.set_ylabel("Predicted UCS (kPa)")
    ax.set_title("Model Prediction Comparison")

    st.pyplot(fig)

    report = pd.DataFrame({
        "Parameter":["FA","SCBA","EC","Gs","PI","FSI","MDUW","OMC","UPV","CuringDays"],
        "Value":[FA,SCBA,EC,Gs,PI,FSI,MDUW,OMC,UPV,CuringDays]
    })

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Input Data",
        csv,
        "UCS_prediction_report.csv",
        "text/csv"
    )
```
