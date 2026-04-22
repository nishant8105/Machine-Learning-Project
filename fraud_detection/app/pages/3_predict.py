import streamlit as st
import pandas as pd

from src.predict import predict_single, predict_batch
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

st.set_page_config(page_title="Predict", page_icon="🔮", layout="wide")
st.title("🔮 Fraud Prediction")


if "model_name" not in st.session_state:
    st.warning("⚠️ Please train a model on the Model page first")
    st.stop()

model_name = st.session_state.model_name


threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)
st.sidebar.caption("Lower = catch more fraud, more false alarms")


tab1, tab2 = st.tabs(["🔍 Single", "📂 Batch"])

with tab1:
    st.subheader("Single Transaction Prediction")

    # Layout in columns for better UI
    col1, col2, col3 = st.columns(3)

    with col1:
        amount = st.number_input("Amount ($)", min_value=0.0, value=100.0)
        time = st.number_input("Time (seconds)", min_value=0.0, value=50000.0)

        v1 = st.number_input("V1", value=0.0, format="%.4f")
        v2 = st.number_input("V2", value=0.0, format="%.4f")
        v3 = st.number_input("V3", value=0.0, format="%.4f")
        v4 = st.number_input("V4", value=0.0, format="%.4f")
        v5 = st.number_input("V5", value=0.0, format="%.4f")
        v6 = st.number_input("V6", value=0.0, format="%.4f")
        v7 = st.number_input("V7", value=0.0, format="%.4f")
        v8 = st.number_input("V8", value=0.0, format="%.4f")
        v9 = st.number_input("V9", value=0.0, format="%.4f")
        v10 = st.number_input("V10", value=0.0, format="%.4f")

    with col2:
        v11 = st.number_input("V11", value=0.0, format="%.4f")
        v12 = st.number_input("V12", value=0.0, format="%.4f")
        v13 = st.number_input("V13", value=0.0, format="%.4f")
        v14 = st.number_input("V14", value=0.0, format="%.4f")
        v15 = st.number_input("V15", value=0.0, format="%.4f")
        v16 = st.number_input("V16", value=0.0, format="%.4f")
        v17 = st.number_input("V17", value=0.0, format="%.4f")
        v18 = st.number_input("V18", value=0.0, format="%.4f")
        v19 = st.number_input("V19", value=0.0, format="%.4f")
        v20 = st.number_input("V20", value=0.0, format="%.4f")

    with col3:
        v21 = st.number_input("V21", value=0.0, format="%.4f")
        v22 = st.number_input("V22", value=0.0, format="%.4f")
        v23 = st.number_input("V23", value=0.0, format="%.4f")
        v24 = st.number_input("V24", value=0.0, format="%.4f")
        v25 = st.number_input("V25", value=0.0, format="%.4f")
        v26 = st.number_input("V26", value=0.0, format="%.4f")
        v27 = st.number_input("V27", value=0.0, format="%.4f")
        v28 = st.number_input("V28", value=0.0, format="%.4f")

    # Build transaction dict
    transaction = {
        "Time" : time,
        "Amount": amount,
        "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5,
        "V6": v6, "V7": v7, "V8": v8, "V9": v9, "V10": v10,
        "V11": v11, "V12": v12, "V13": v13, "V14": v14, "V15": v15,
        "V16": v16, "V17": v17, "V18": v18, "V19": v19, "V20": v20,
        "V21": v21, "V22": v22, "V23": v23, "V24": v24, "V25": v25,
        "V26": v26, "V27": v27, "V28": v28,
    }

    if st.button("🔍 Predict Transaction"):
        result = predict_single(transaction, model_name, threshold)

        label = result["label"]
        prob  = result["probability"]

        st.write(f"### Probability of Fraud: `{prob:.4f}`")

        # Risk level
        if prob > 0.8:
            st.error("🚨 HIGH RISK FRAUD")
        elif prob > 0.5:
            st.warning("⚠️ MEDIUM RISK")
        else:
            st.info("✅ LOW RISK")

        # Final label
        if label == "Fraud":
            st.error("💳 FRAUD DETECTED")
        else:
            st.success("✔️ LEGIT TRANSACTION")

with tab2:
    st.subheader("Batch Prediction (CSV Upload)")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        st.write("Preview:")
        st.dataframe(df.head())

        if st.button("📊 Predict Batch"):
            results = predict_batch(df, model_name)

            st.write("Results:")
            st.dataframe(results)

            # Download
            csv = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download Results",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )