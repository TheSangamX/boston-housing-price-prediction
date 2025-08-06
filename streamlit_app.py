import streamlit as st
import numpy as np
import joblib

# --- Page Styling ---
# This section injects custom CSS to create a clean, light gradient background and style other components.
# The styling is applied using st.markdown with unsafe_allow_html=True.
# This approach isolates the styling from the app's core logic and features.
page_bg = """
<style>
/* Sets the background for the main app container */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e0f7fa 0%, #ffffff 100%);
    background-attachment: fixed;
}

/* Ensures the header is transparent to show the gradient underneath */
[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
}

/* Adds a semi-transparent white background to the sidebar for readability */
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.9);
    border-radius: 10px; /* Optional: adds rounded corners to the sidebar */
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --- App Logic ---

# Load the trained model - ensure "Boston.joblib" is in the same directory
try:
    model = joblib.load("Boston.joblib")
except FileNotFoundError:
    st.error("Error: The model file 'Boston.joblib' was not found. Please make sure it's in the correct directory.")
    st.stop() # Halts the app if the model can't be loaded

# Title of the app
st.title("Boston Housing Price Predictor")
st.write("Enter the details below to get a prediction of the house price.")

# Input fields for all 13 features with default values
# Using columns for a more compact layout
col1, col2 = st.columns(2)

with col1:
    CRIM = st.number_input("CRIM - Per capita crime rate by town", value=0.1, format="%.4f")
    ZN = st.number_input("ZN - Proportion of residential land zoned for lots over 25,000 sq.ft.", value=0.0)
    INDUS = st.number_input("INDUS - Proportion of non-retail business acres per town", value=7.0)
    CHAS = st.selectbox("CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)", [0, 1])
    NOX = st.number_input("NOX - Nitric oxides concentration (parts per 10 million)", value=0.5, format="%.4f")
    RM = st.number_input("RM - Average number of rooms per dwelling", value=6.0)
    AGE = st.number_input("AGE - Proportion of owner-occupied units built prior to 1940", value=60.0)

with col2:
    DIS = st.number_input("DIS - Weighted distances to five Boston employment centres", value=4.0)
    RAD = st.number_input("RAD - Index of accessibility to radial highways", value=1)
    TAX = st.number_input("TAX - Full-value property-tax rate per $10,000", value=296)
    PTRATIO = st.number_input("PTRATIO - Pupil-teacher ratio by town", value=15.0)
    B = st.number_input("B - 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town", value=396.9)
    LSTAT = st.number_input("LSTAT - % lower status of the population", value=12.5)


# Prediction button
if st.button("Predict Price", type="primary"):
    # Create a numpy array from the user inputs
    input_data = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE,
                            DIS, RAD, TAX, PTRATIO, B, LSTAT]])
    
    # Make a prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Display the prediction in a success box
    # The model predicts in $1000s, so we multiply by 1000 for the actual price.
    st.success(f"🏡 Predicted House Price: ${prediction[0]*1000:,.2f}")

