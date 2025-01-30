import streamlit as st
import pandas as pd
import joblib

st.write("""
# HDB Resale Price Prediction App
This app predicts the **resale price** of an HDB flat based on various features.
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    floor_area_sqm = st.sidebar.number_input('Floor Area (sqm)', min_value=30.0, max_value=200.0, value=80.0)

    town = st.sidebar.selectbox('Town', [
        'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH',
        'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
        'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG',
        'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN'
    ])

    flat_type = st.sidebar.selectbox('Flat Type', [
        '1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'
    ])

    storey_range = st.sidebar.selectbox('Storey Range', [
        '01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21',
        '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42',
        '43 TO 45', '46 TO 48', '49 TO 51'
    ])

    flat_model = st.sidebar.selectbox('Flat Model', [
        '2-room', '3Gen', 'Adjoined flat', 'Apartment', 'DBSS', 'Improved', 'Improved-Maisonette',
        'Maisonette', 'Model A', 'Model A-Maisonette', 'Model A2', 'Multi Generation', 'New Generation',
        'Premium Apartment', 'Premium Apartment Loft', 'Premium Maisonette', 'Simplified', 'Standard',
        'Terrace', 'Type S1', 'Type S2'
    ])

    year = st.sidebar.selectbox('Year of Resale', [
        2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025
    ])

    month_name = st.sidebar.selectbox('Month of Resale', [
        'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
        'September', 'October', 'November', 'December'
    ])

    remaining_lease_binned = st.sidebar.selectbox('Remaining Lease (Binned)', [
        '40-50', '51-60', '61-70', '71-80', '81-90', '91-100'
    ])

    # Create input dictionary matching model feature names
    data = {'floor_area_sqm': floor_area_sqm}

    for t in [
        'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH',
        'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
        'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG',
        'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN'
    ]:
        data[f'town_{t}'] = 1 if town == t else 0

    for f in ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']:
        data[f'flat_type_{f}'] = 1 if flat_type == f else 0

    for s in [
        '01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21',
        '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42',
        '43 TO 45', '46 TO 48', '49 TO 51'
    ]:
        data[f'storey_range_{s}'] = 1 if storey_range == s else 0

    for fm in [
        '2-room', '3Gen', 'Adjoined flat', 'Apartment', 'DBSS', 'Improved', 'Improved-Maisonette',
        'Maisonette', 'Model A', 'Model A-Maisonette', 'Model A2', 'Multi Generation', 'New Generation',
        'Premium Apartment', 'Premium Apartment Loft', 'Premium Maisonette', 'Simplified', 'Standard',
        'Terrace', 'Type S1', 'Type S2'
    ]:
        data[f'flat_model_{fm}'] = 1 if flat_model == fm else 0

    for y in range(2015, 2026):
        data[f'year_{y}'] = 1 if year == y else 0

    for m in [
        'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
        'September', 'October', 'November', 'December'
    ]:
        data[f'month_name_{m}'] = 1 if month_name == m else 0

    for rl in ['40-50', '51-60', '61-70', '71-80', '81-90', '91-100']:
        data[f'remaining_lease_binned_{rl}'] = 1 if remaining_lease_binned == rl else 0

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

# Load the trained regression model
model = joblib.load('decision_tree_regression.pkl')

# Predict resale price
prediction = model.predict(df)

st.subheader('Predicted Resale Price')
st.write(f"${prediction[0]:,.2f}")
