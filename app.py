import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load model from pickle
def load_model(model_filename):
    model_path = os.path.join("models", model_filename)
    with open(model_path, "rb") as file:
        return pickle.load(file)

# Load the standard scaler
def load_scaler():
    scaler_path = os.path.join("models", "standard_scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as file:
            return pickle.load(file)
    return None

# UI
def main():
    st.set_page_config(page_title="ML Models Web App", layout="centered")
    st.title("🔍 Life Expectancy Prediction Model")

    tabs = st.tabs(["🔮 Predict", "📊 Regression Report"])

    with tabs[0]:
        st.subheader("Select a Model")
        model_files = {
            "Best Random Forest": "best_random_forest_model.pkl",
            "Random Forest": "random_forest_model.pkl",
            "Decision Tree": "decision_tree_model.pkl",
            "Linear Regression": "linear_regression_model.pkl"
        }

        model_name = st.selectbox("Choose a Model", list(model_files.keys()))
        model = load_model(model_files[model_name])
        scaler = load_scaler()

        if hasattr(model, "feature_names_in_"):
            st.subheader("Enter Input Features")
            input_data = {}
            for feature in model.feature_names_in_:
                input_data[feature] = st.number_input(f"{feature}", value=0.0)

            input_df = pd.DataFrame([input_data])

            if scaler:
                try:
                    input_df_scaled = scaler.transform(input_df)
                except Exception as e:
                    st.warning("Scaler could not transform inputs. Proceeding without scaling.")
                    input_df_scaled = input_df
            else:
                input_df_scaled = input_df

            if st.button("Predict"):
                prediction = model.predict(input_df_scaled)
                st.success(f"Prediction: {prediction[0]:.4f}")

        else:
            st.warning("Model does not have identifiable input features.")

    with tabs[1]:
        st.subheader("📄 Regression Report")
        report_path = os.path.join("models", "regression_report.txt")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report = f.read()
            st.text_area("Regression Report:", value=report, height=400)
        else:
            st.error("regression_report.txt not found in models folder.")

if __name__ == "__main__":
    main()
