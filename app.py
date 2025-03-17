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
st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f4; /* Light gray background */
        }
        .stApp {
            background-color: #f4f4f4;
        }
        /* Make headings and main text black */
        h1, h2, h3, h4, h5, h6, p {
            color: black !important;
        }
        /* Ensure labels (like "Assessed Improved Value") are black */
        label {
            color: black !important;
            font-weight: bold;
        }

        /* Style the number input field to match dropdowns */
        input[type="number"] {
            background-color: #e0e0e0 !important; /* Light gray */
            color: black !important; /* Black text */
            border-radius: 5px;
            padding: 8px;
            border: 1px solid #ccc;
        }

        /* Button inside number input */

        .stButton>button {
            background-color: #FFDEAD; /* Light brown */
            color: white !important; /* White text */
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
        }

        input[type="number"]::-webkit-inner-spin-button, 
        input[type="number"]::-webkit-outer-spin-button {
            opacity: 1;
            background: #e0e0e0;
        }

        /* Dropdown input box */
        .stSelectbox {
            background-color: #e0e0e0 !important; /* Light gray */
            color: black !important; /* Black text */
            border-radius: 5px;
            padding: 5px;
        }
        
        /* Dropdown options */
        div[data-baseweb="select"] > div {
            background-color: #f5f5f5 !important; /* Lighter gray dropdown */
            color: black !important;
        }
        
        /* Highlighted option */
        div[data-baseweb="select"] div[aria-selected="true"] {
            background-color: #FFDEAD !important; /* Light brown highlight */
            color: black !important;
        }

        /* Change dropdown symbol (arrow) color */
        .stSelectbox svg {
            fill: black !important; /* Change to black */
        }
    </style>
    """,
    unsafe_allow_html=True
)


numeric_feature = "Assessed Improved Value (parcel)"

assessed_value_options = [
    "0 - 50,000", "50,001 - 100,000", "100,001 - 200,000", "200,001 - 500,000", "500,001 - 1,000,000", "1,000,001+"
]

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
        "Mesh Screen <= 1/8\"", "Mesh Screen > 1/8\"", "Unscreened", "Unknown",
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
st.title("Wildfire Damage Prediction App 🔥")
st.write("Enter the required features to predict the level of damage.")

# Numeric input as a manual entry field
numerical_input = st.number_input(
    "Assessed Improved Value (parcel)", 
    min_value=0, 
    max_value=5000000,  # Adjust max based on your data
    value=100000,  # Default value
    step=10000  # Increment step
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
        prediction = model.predict(input_df)

        # 4.3 Map numeric prediction back to original damage labels
        damage_labels = {
            4: ("No Damage", "#2E8B57"),   # Green
            0: ("Affected (1-9%)", "#FFD700"),  # Yellow
            3: ("Minor (10-25%)", "#FFA500"),  # Orange
            1: ("Destroyed (>50%)", "#B22222"),  # Red
            2: ("Major (26-50%)", "#FF6347")   # Light Red
        }
        predicted_label, color = damage_labels.get(prediction[0], ("Unknown", "#808080"))

        st.markdown(
            f'<div style="padding: 10px; border-radius: 5px; background-color: {color}; color: white;">'
            f'🔥 <b>Predicted Damage Level: {predicted_label} </b>'
            f'</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
