import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1) Load the Trained Model & Preprocessor
# ---------------------------------------------------------
try:
    model = joblib.load("xgb_model.pkl")         # entire pipeline
    preprocessor = joblib.load("preprocessor.pkl")
except Exception as e:
    st.error(f"❌ Error loading model or preprocessor: {e}")
    st.stop()

if not hasattr(model, "predict"):
    st.error("❌ Model was not loaded correctly. Ensure it's an XGBoost model.")
    st.stop()

# ---------------------------------------------------------
# 2) Define Your EXACT Features (same as training)
# ---------------------------------------------------------
numeric_feature = "Assessed Improved Value (parcel)"

categorical_features = {
    "Structure Category": [
        "Single Residence", "Other Minor Structure", "Multiple Residence",
        "Nonresidential Commercial", "Mixed Commercial/Residential",
        "Infrastructure", "Agriculture"
    ],
    "Roof Construction": [
        "Asphalt", "Tile", "Unknown", "Metal", "Concrete", "Other", "Wood",
        "Combustible", "Fire Resistant", "No Deck/Porch", "Non Combustible"
    ],
    "Eaves": [
        "Unenclosed", "Enclosed", "Unknown", "No Eaves", "Not Applicable", "Combustible"
    ],
    "Vent Screen": [
        "Mesh Screen <= 1/8\"\"", "Mesh Screen > 1/8\"\"", "Unscreened", "Unknown",
        "No Vents", "Screened", ">30", "21-30", "Deck Elevated", "Attached Fence"
    ],
    "Exterior Siding": [
        "Wood", "Stucco Brick Cement", "Unknown", "Metal", "Other", "Vinyl",
        "Ignition Resistant", "Combustible", "Fire Resistant", "Stucco/Brick/Cement"
    ],
    "Window Pane": [
        "Single Pane", "Multi Pane", "Unknown", "No Windows", "No Deck/Porch",
        "Radiant Heat", "Asphalt"
    ],
    "Fence Attached to Structure": [
        "No Fence", "Combustible", "Unknown", "Non Combustible"
    ]
}

# ---------------------------------------------------------
# 3) Streamlit UI
# ---------------------------------------------------------
st.title("Wildfire Damage Prediction App")
st.write("Enter the required features to predict the level of damage.")

# Numeric input
numerical_input = st.number_input(
    numeric_feature, min_value=0, max_value=10_000_000, value=100000, step=1000
)

# Categorical inputs
cat_inputs = {}
for cat_col, valid_options in categorical_features.items():
    cat_inputs[cat_col] = st.selectbox(cat_col, options=valid_options)

# ---------------------------------------------------------
# 4) Prediction Button
# ---------------------------------------------------------
if st.button("Predict"):
    try:
        # 4.1 Create a single-row DataFrame from user input
        data_row = [numerical_input] + list(cat_inputs.values())
        input_df = pd.DataFrame([data_row],
            columns=[numeric_feature] + list(categorical_features.keys())
        )

        # 4.2 Transform with the pipeline's preprocessor
        #     (since 'model' includes the pipeline, we can also do 'model.predict')
        #     But if you want to transform manually, do:
        # input_processed = preprocessor.transform(input_df)
        # prediction = model['classifier'].predict(input_processed)

        # However, if 'model' is the entire pipeline, we can do:
        prediction = model.predict(input_df)

        # 4.3 Map numeric prediction back to original damage labels
        damage_labels = {
            4: "No Damage",
            0: "Affected (1-9%)",
            3: "Minor (10-25%)",
            1: "Destroyed (>50%)",
            2: "Major (26-50%)"
        }
        predicted_label = damage_labels.get(prediction[0], "Unknown")

        st.success(f"🔥 **Predicted Damage Level: {predicted_label} (Label={prediction[0]})**")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

#streamlit run app.py
