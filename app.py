
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="7-Day Mortality Risk Prediction for Acute Myocardial Infarction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_model():
    """Load the trained CatBoost model and selected features."""
    try:
        model_data = joblib.load('NB_CatBoost_final.pkl')
        
        # The saved object is a tuple: (pipeline, selected_features)
        if isinstance(model_data, tuple) and len(model_data) == 2:
            pipeline, selected_features = model_data
            return pipeline, selected_features
        elif hasattr(model_data, 'predict_proba'):
            # If directly a model (backward compatibility)
            return model_data, None
        else:
            st.error(f"Unexpected model format: {type(model_data)}")
            return None, None
    except FileNotFoundError:
        st.error("Model file 'NB_CatBoost_final.pkl' not found. Please ensure it is in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

# Page title and description
st.title("Risk Prediction Model for 7-Day Mortality in Acute Myocardial Infarction")
st.markdown("---")

# Load model
model, selected_features = load_model()
if model is None:
    st.stop()

# Use provided feature list if not extracted from file (fallback)
if selected_features is None:
    selected_features = [
        'Urinary_output_24h', 'Age', 'Bun', 'Bicarbonate', 'PT', 'RR', 
        'Glucose', 'cTNT', 'Norepinephrine', 'WBC', 'HR', 'SBP', 
        'MCHC', 'Hemoglobin', 'Potassium', 'Atorvastatin'
    ]

# Define clinical ranges and units for each feature
feature_ranges = {
    'Urinary_output_24h': (0, 10000, 'mL/24h'),
    'Age': (18, 120, 'years'),
    'Bun': (0, 300, 'mg/dL'),
    'Bicarbonate': (0, 100, 'mmol/L'),
    'PT': (0, 200, 'seconds'),
    'RR': (0, 100, 'breaths/min'),
    'Glucose': (0, 800, 'mg/dL'),
    'cTNT': (0, 1000, 'ng/mL'),
    'Norepinephrine': (0, 100, 'mg/24h'),
    'WBC': (0, 500, '×10⁹/L'),
    'HR': (0, 250, 'bpm'),
    'SBP': (0, 300, 'mmHg'),
    'MCHC': (0, 100, 'g/dL'),
    'Hemoglobin': (0, 250, 'g/dL'),
    'Potassium': (0, 10, 'mmol/L'),
    'Atorvastatin': (0, 320, 'mg')   # Updated range for continuous variable
}

# Layout: input form (left) and results (right)
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Patient Information Input")
    
    with st.form("patient_form"):
        input_col1, input_col2 = st.columns(2)
        input_values = {}
        
        # Split features into two columns
        half = len(selected_features) // 2
        with input_col1:
            for feature in selected_features[:half]:
                min_val, max_val, unit = feature_ranges.get(feature, (0, 100, ''))
                input_values[feature] = st.number_input(
                    f"{feature} ({unit})" if unit else feature,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),
                    step=0.1,
                    key=f"{feature}_1"
                )
        
        with input_col2:
            for feature in selected_features[half:]:
                min_val, max_val, unit = feature_ranges.get(feature, (0, 100, ''))
                input_values[feature] = st.number_input(
                    f"{feature} ({unit})" if unit else feature,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),
                    step=0.1,
                    key=f"{feature}_2"
                )
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Verify all features are provided
            missing_features = set(selected_features) - set(input_values.keys())
            if missing_features:
                st.error(f"Missing features: {missing_features}")
            else:
                # Create DataFrame with correct feature order
                input_data = [input_values[feature] for feature in selected_features]
                input_df = pd.DataFrame([input_data], columns=selected_features)
                
                try:
                    # Get probability of mortality (class 1)
                    probability = model.predict_proba(input_df)[0][1] * 100
                    st.success("Prediction completed!")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

with col2:
    st.header("Prediction Results")
    
    if submitted and 'probability' in locals():
        st.metric(
            label="7-Day Mortality Risk Probability",
            value=f"{probability:.1f}%"
        )
    else:
        st.info("Please fill in the patient information and click 'Predict' to see results.")

st.markdown("---")
st.header("About This Calculator")

about_text = f"""
This prediction model estimates the **7-day mortality risk** for patients with **Acute Myocardial Infarction (AMI)**.  
It is based on a **CatBoost machine learning model** trained on clinical data with recursive feature elimination and hyperparameter optimization.

**Selected Clinical Features ({len(selected_features)} total)**:

- **Vital Signs & Demographics**: Age, Heart Rate (HR), Respiratory Rate (RR), Systolic Blood Pressure (SBP)
- **Laboratory Values**: Blood Urea Nitrogen (Bun), Bicarbonate, Prothrombin Time (PT), Glucose, Cardiac Troponin (cTNT), White Blood Cells (WBC), MCHC, Hemoglobin, Potassium
- **Renal Function & Output**: 24-hour Urinary Output
- **Medications & Vasopressors**: Norepinephrine dose, Atorvastatin dose (mg)

**Disclaimer**: This tool is intended for clinical decision support only and should be used in conjunction with professional medical judgment. Always verify predictions against full clinical assessment.
"""

st.markdown(about_text)
st.caption("© 2026 - AMI Mortality Risk Calculator | Model: Calibrated CatBoost")
