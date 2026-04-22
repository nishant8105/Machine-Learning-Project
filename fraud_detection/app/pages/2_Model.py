import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_data, get_feature_target
from src.preprocessor import split_data, sample_data, get_cv
from src.train import compare_models, get_best_model, tune_model, train_best_model, save_model
from src.evaluate import get_metrics, plot_confusion_matrix, plot_roc_curve, plot_threshold_tuning

st.set_page_config(page_title='Model Training', 
                   page_icon='🤖', 
                   layout='wide')
st.title("🤖 Model Training & Evaluation")
st.divider()

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Training Settings")
training_mode = st.sidebar.radio(
    "Training Mode",
    ["Quick (Sampled)", "Full (Slow)"],
    index=0,
    help="Quick Mode uses 20% of the data for faster iteration. Full Mode uses 100%."
)

cv_folds = st.sidebar.slider("Cross-Validation Folds", 2, 5, 3)

sample_fraction = 0.2 if training_mode == "Quick (Sampled)" else 1.0

if "stage" not in st.session_state:
    st.session_state.stage = "compare"

@st.cache_data 
def load_and_split():
    df = load_data() 
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y) 
    return X_train, X_test, y_train, y_test 

X_train_full, X_test, y_train_full, y_test = load_and_split()

# Apply sampling for the comparison/tuning phase
X_train, y_train = sample_data(X_train_full, y_train_full, fraction=sample_fraction)

st.header("1️⃣ Compare Models")
if training_mode == "Quick (Sampled)":
    st.info(f"💡 Quick Mode: Training on {len(X_train)} samples ({sample_fraction*100:.0f}%) with {cv_folds}-fold CV.")

if st.button("▶ Compare All Models"):
    with st.spinner("Training models..."):
        # Update CV default globally or pass it? 
        # For now, train.py uses get_cv() which will use its new default of 3 unless we change it.
        # But we want to follow the slider.
        results = compare_models(X_train, y_train)

        st.session_state.results = results
        st.session_state.best_name = get_best_model(results)
        st.session_state.stage = "tune"

# Display results
if "results" in st.session_state:
    results = st.session_state.results
    df_results = pd.DataFrame([
        {"Model" : name, "Mean F1" : v["mean_f1"], "Std F1" : v["std_f1"]}
        for name, v in results.items()
    ])
    st.dataframe(df_results, use_container_width=True)
    fig = px.bar(df_results, x="Model", y="Mean F1",
                 error_y="Std F1", title="Model Comparison")
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"Best Model: {st.session_state.best_name}")


if st.session_state.stage in ["tune", "train", "evaluate"]:
    st.header("2️⃣ Tune Best Model")

    n_trials = st.slider("Optuna trials", 10, 100, 30)

    if st.button("⚙️ Tune Best Model"):
        with st.spinner("Tuning..."):
            best_score, best_params, model_name = tune_model(
                st.session_state.best_name,
                X_train, y_train,
                n_trials=n_trials
            )

            st.session_state.best_params = best_params
            st.session_state.model_name = model_name
            st.session_state.best_score = best_score
            st.session_state.stage = "train"

# Show tuning results
if "best_params" in st.session_state:
    st.write("Best Params:", st.session_state.best_params)
    st.info(f"Best Score: {st.session_state.best_score}")


if st.session_state.stage in ["train", "evaluate"]:
    st.header("3️⃣ Train Final Model")
    st.info("💡 Final training will use the **full dataset** to ensure maximum accuracy.")

    if st.button("🏋️ Train Final Model"):
        with st.spinner("Training final model on full dataset..."):
            pipeline = train_best_model(
                st.session_state.model_name,
                st.session_state.best_params,
                X_train_full, y_train_full
            )

            save_model(pipeline, st.session_state.model_name)

            st.session_state.pipeline = pipeline
            st.session_state.stage = "evaluate"

if st.session_state.stage == "evaluate":
    st.header("4️⃣ Evaluation")

    pipeline = st.session_state.pipeline

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]


    metrics = get_metrics(y_test, y_pred, y_prob)

    # Metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy",  metrics["accuracy"])
    col2.metric("Precision", metrics["precision"])
    col3.metric("Recall",    metrics["recall"])
    col4.metric("F1 Score",  metrics["f1"])
    col5.metric("ROC AUC",   metrics["roc_auc"])

    st.subheader("Classification Report")
    st.code(metrics["classification_report"])

    # Charts
    st.pyplot(plot_confusion_matrix(y_test, y_pred))
    st.pyplot(plot_roc_curve(y_test, y_prob))
    fig_thresh, best_threshold = plot_threshold_tuning(y_test, y_prob)
    st.pyplot(fig_thresh)
    st.info(f"Recommended Threshold : {best_threshold:.2f}")