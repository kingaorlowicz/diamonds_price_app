import streamlit as st
import pandas as pd
import joblib

# 1. Load the model and the columns
model = joblib.load('xgboost_diamonds_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎")
st.title("💎 Diamond Price Estimator")
st.write("Enter the characteristics of the diamond to get a price prediction from the XGBoost model.")

st.divider()

# 2. Input Fields
col1, col2 = st.columns(2)

with col1:
    carat = st.number_input("Weight (Carat)", min_value=0.2, max_value=5.0, value=0.7, step=0.01)
    depth = st.number_input("Depth Percentage", min_value=40.0, max_value=80.0, value=61.0, step=0.1)
    table = st.number_input("Table Width", min_value=40.0, max_value=80.0, value=57.0, step=0.1)

with col2:
    cut = st.selectbox("Cut Quality", ["Ideal", "Premium", "Very Good", "Good", "Fair"])
    color = st.selectbox("Color Grade", ["D", "E", "F", "G", "H", "I", "J"])
    clarity = st.selectbox("Clarity Grade", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])

st.divider()

# 3. Prediction Logic
if st.button("Calculate Predicted Price"):
    input_dict = {
        'carat': carat,
        'depth': depth,
        'table': table,
        'cut': cut,
        'color': color,
        'clarity': clarity
    }
    
    input_df = pd.DataFrame([input_dict])
    
    input_df_encoded = pd.get_dummies(input_df)
    
    for col in model_columns:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0
    input_df_encoded = input_df_encoded[model_columns]
    
    # Predict
    prediction = model.predict(input_df_encoded)
    
    # Result
    st.balloons()
    st.success(f"### Predicted Price: ${prediction[0]:,.2f}")
    st.caption("The prediction is based on the features provided using a trained regression model.")