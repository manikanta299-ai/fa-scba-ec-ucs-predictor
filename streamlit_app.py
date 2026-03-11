import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="UCS Prediction FA-SCBA-EC", page_icon="🧱", layout="wide")

st.title("AI-Based UCS Prediction of FA-SCBA Mixtures Blended with EC")

scaler = joblib.load("scaler.pkl")

model_name = st.selectbox(
    "Select Machine Learning Model",
    ["RF","ET","HGBR","SVR","ANN","Spline"]
)

model_files = {
    "RF":"RF_model.pkl",
    "ET":"ET_model.pkl",
    "HGBR":"HGBR_model.pkl",
    "SVR":"SVR_model.pkl",
    "ANN":"ANN_model.pkl",
    "Spline":"Spline_model.pkl"
}

model = joblib.load(model_files[model_name])

st.subheader("Input Parameters")

col1,col2,col3 = st.columns(3)

with col1:
    FA = st.number_input("Fly Ash (%)",value=20.0)
    Gs = st.number_input("Specific Gravity",value=2.65)
    MDUW = st.number_input("Maximum Dry Unit Weight",value=16.0)

with col2:
    SCBA = st.number_input("SCBA (%)",value=10.0)
    PI = st.number_input("Plasticity Index",value=15.0)
    OMC = st.number_input("Optimum Moisture Content",value=20.0)

with col3:
    EC = st.number_input("Eggshell Powder (%)",value=5.0)
    FSI = st.number_input("Free Swell Index",value=40.0)
    UPV = st.number_input("UPV",value=2000.0)

CuringDays = st.number_input("Curing Days",value=28)

if st.button("Predict UCS"):

    input_data = pd.DataFrame([[FA,SCBA,EC,Gs,PI,FSI,MDUW,OMC,UPV,CuringDays]],
    columns=["FA","SCBA","EC","Gs","PI","FSI","MDUW","OMC","UPV","CuringDays"])

    scaled = scaler.transform(input_data)

    prediction = model.predict(scaled)[0]

    st.success(f"Predicted UCS = {prediction:.2f} kPa")

    report = pd.DataFrame({
        "Parameter":["FA","SCBA","EC","Gs","PI","FSI","MDUW","OMC","UPV","CuringDays","Predicted UCS"],
        "Value":[FA,SCBA,EC,Gs,PI,FSI,MDUW,OMC,UPV,CuringDays,prediction]
    })

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Prediction Report",
        csv,
        "UCS_prediction_report.csv",
        "text/csv"
    )

st.subheader("Model Prediction Accuracy (%)")

model_accuracy = {
    "RF":94.5,
    "ET":96.2,
    "HGBR":93.8,
    "SVR":91.4,
    "ANN":97.1,
    "Spline":90.6
}

accuracy_df = pd.DataFrame({
    "Model":list(model_accuracy.keys()),
    "Accuracy (%)":list(model_accuracy.values())
})

st.dataframe(accuracy_df)

best_model = max(model_accuracy, key=model_accuracy.get)

st.success(f"Best Performing Model: {best_model} ({model_accuracy[best_model]}%)")

st.subheader("3D UCS Strength Surface")

fa_range = np.linspace(0,40,15)
ec_range = np.linspace(0,15,15)

X_mesh,Y_mesh = np.meshgrid(fa_range,ec_range)

Z = []

for i in range(len(ec_range)):
    row = []
    for j in range(len(fa_range)):

        temp = pd.DataFrame([[fa_range[j],SCBA,ec_range[i],Gs,PI,FSI,MDUW,OMC,UPV,CuringDays]],
        columns=["FA","SCBA","EC","Gs","PI","FSI","MDUW","OMC","UPV","CuringDays"])

        scaled = scaler.transform(temp)

        pred = model.predict(scaled)[0]

        row.append(pred)

    Z.append(row)

Z = np.array(Z)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(X_mesh,Y_mesh,Z,cmap="viridis")

ax.set_xlabel("Fly Ash (%)")
ax.set_ylabel("Eggshell Powder (%)")
ax.set_zlabel("Predicted UCS")

st.pyplot(fig)