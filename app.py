import streamlit as st
import pandas as pd
import joblib

st.title("House Rent Prediction")

model = joblib.load("house_rent_prediction.pkl")

BHK = st.number_input("Enter the number of BHK:", 1, 10, 2)

Area_Type = st.selectbox(
    "Select Area Type",
    ["Super Area", "Carpet Area", "Built Area"]
)

City = st.selectbox(
    "Select City",
    ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"]
)

Furnishing_Status = st.selectbox(
    "Select Furnishing Status",
    ["Furnished", "Semi-Furnished", "Unfurnished"]
)

Tenant_Preferred = st.selectbox(
    "Select Tenant Preferred",
    ["Family", "Bachelors", "Any"]
)

Bathroom = st.number_input(
    "Enter Bathrooms",
    1,
    10,
    2
)

Point_of_Contact = st.selectbox(
    "Point of Contact",
    ["Contact Owner", "Contact Agent"]
)

input_df = pd.DataFrame({
    "BHK":[BHK],
    "Area Type":[Area_Type],
    "City":[City],
    "Furnishing Status":[Furnishing_Status],
    "Tenant Preferred":[Tenant_Preferred],
    "Bathroom":[Bathroom],
    "Point of Contact":[Point_of_Contact]
})

if st.button("Predict Rent"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Rent = ₹{prediction[0]:,.0f}")