import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
import gdown

# ===== 1️⃣ Download & load data =====
url = "https://drive.google.com/file/d/1UJE2gOZT8grQVXvmuZRZnGwpxeHSC-X-/view?usp=sharing"
file_id = url.split('/')[-2]
download_url = f"https://drive.google.com/uc?id={file_id}"
output = 'hotel_booking_data.csv'
gdown.download(download_url, output, quiet=False)

df = pd.read_csv(output, engine='python', on_bad_lines='skip', encoding='utf-8')

# ===== 2️⃣ Buat target revenue =====
df['total_revenue'] = df['adr'] * (df['stays_in_weekend_nights'] + df['stays_in_week_nights'])

# ===== 3️⃣ Sidebar Filter =====
st.sidebar.header("Filter Data Booking")

def selectbox_with_all(label, options):
    return st.sidebar.multiselect(label, options=['All'] + sorted(options), default=['All'])

country_filter = selectbox_with_all("Country", df['country'].astype(str).unique())
hotel_filter = selectbox_with_all("Hotel", df['hotel'].astype(str).unique())
market_filter = selectbox_with_all("Market Segment", df['market_segment'].astype(str).unique())
dist_filter = selectbox_with_all("Distribution Channel", df['distribution_channel'].astype(str).unique())
customer_filter = selectbox_with_all("Customer Type", df['customer_type'].astype(str).unique())

# ===== 4️⃣ Terapkan filter =====
filtered_df = df.copy()
def apply_filter(df, col, selected_values):
    if 'All' in selected_values:
        return df
    else:
        return df[df[col].astype(str).isin(selected_values)]

filtered_df = apply_filter(filtered_df, 'country', country_filter)
filtered_df = apply_filter(filtered_df, 'hotel', hotel_filter)
filtered_df = apply_filter(filtered_df, 'market_segment', market_filter)
filtered_df = apply_filter(filtered_df, 'distribution_channel', dist_filter)
filtered_df = apply_filter(filtered_df, 'customer_type', customer_filter)

st.header("Data Booking Setelah Filter")
st.write(f"Menampilkan {len(filtered_df)} baris dari total {len(df)} baris")
st.dataframe(filtered_df)

# ===== 5️⃣ Prediksi =====
st.header("Prediksi Potential Revenue & Fasilitas")
st.subheader("Isi Data Booking")

with st.form("booking_form"):
    stays_weekend = st.number_input("Weekend Nights", 0, 10, 2)
    stays_week = st.number_input("Week Nights", 0, 20, 3)
    adults = st.number_input("Adults", 1, 10, 2)
    children = st.number_input("Children", 0, 5, 0)
    babies = st.number_input("Babies", 0, 2, 0)
    
    # Pilih kategori dari data yang sudah difilter
    hotel_options = filtered_df['hotel'].astype(str).unique().tolist()
    country_options = filtered_df['country'].astype(str).unique().tolist()
    market_options = filtered_df['market_segment'].astype(str).unique().tolist()
    dist_options = filtered_df['distribution_channel'].astype(str).unique().tolist()
    customer_options = filtered_df['customer_type'].astype(str).unique().tolist()
    meal_options = filtered_df['meal'].astype(str).unique().tolist()
    
    input_data = {}
    input_data['hotel'] = st.selectbox("Hotel", hotel_options)
    input_data['country'] = st.selectbox("Country", country_options)
    input_data['market_segment'] = st.selectbox("Market Segment", market_options)
    input_data['distribution_channel'] = st.selectbox("Distribution Channel", dist_options)
    input_data['customer_type'] = st.selectbox("Customer Type", customer_options)
    input_data['meal'] = st.selectbox("Meal", meal_options)
    
    input_data['stays_in_weekend_nights'] = stays_weekend
    input_data['stays_in_week_nights'] = stays_week
    input_data['adults'] = adults
    input_data['children'] = children
    input_data['babies'] = babies
    input_data['booking_changes'] = int(st.checkbox("Booking Changes?", value=False))
    input_data['required_car_parking_spaces'] = int(st.checkbox("Parking Spaces?", value=False))
    input_data['total_of_special_requests'] = int(st.checkbox("Special Requests?", value=False))
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # ===== 6️⃣ Encode kategori =====
    features = [
        'hotel', 'country', 'market_segment', 'distribution_channel', 'customer_type',
        'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
        'meal', 'booking_changes', 'required_car_parking_spaces', 'total_of_special_requests'
    ]
    categorical_cols = ['hotel','country','market_segment','distribution_channel','customer_type','meal']
    target_facilities = ['meal','booking_changes','required_car_parking_spaces','total_of_special_requests']
    
    # Encode kategori dari filtered data
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        filtered_df[col] = filtered_df[col].astype(str)
        filtered_df[col] = le.fit_transform(filtered_df[col])
        label_encoders[col] = le
    
    target_encoders = {}
    for col in target_facilities:
        filtered_df[col] = filtered_df[col].fillna(0)
        filtered_df[col] = filtered_df[col].astype(float).astype(int).astype(str)
        le = LabelEncoder()
        filtered_df[col] = le.fit_transform(filtered_df[col])
        target_encoders[col] = le
    
    # ===== 7️⃣ Train model =====
    X_rev = filtered_df[features]
    y_rev = filtered_df['total_revenue']
    rf_rev = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_rev.fit(X_rev, y_rev)

    y_fac = filtered_df[target_facilities]
    clf_fac = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
    clf_fac.fit(X_rev, y_fac)
    
    # Encode input user
    input_df = pd.DataFrame(input_data, index=[0])
    for col in categorical_cols:
        le = label_encoders[col]
        input_df[col] = input_df[col].astype(str)
        if input_df[col][0] not in le.classes_:
            input_df[col][0] = le.classes_[0]
        input_df[col] = le.transform(input_df[col])
    
    input_df = input_df[features]
    
    # Prediksi Revenue
    predicted_revenue = rf_rev.predict(input_df)[0]
    
    # Prediksi Fasilitas
    y_pred_fac = clf_fac.predict(input_df)
    predicted_facilities = {}
    for i, col in enumerate(target_facilities):
        le = target_encoders[col]
        predicted_facilities[col] = le.inverse_transform([y_pred_fac[0][i]])[0]
    
    def to_int(x):
        return int(float(x))
    
    # Simulasi biaya fasilitas
    cost_meal = 20 if predicted_facilities['meal'] != '0' else 0
    cost_booking_changes = 10 * to_int(predicted_facilities['booking_changes'])
    cost_parking = 5 * to_int(predicted_facilities['required_car_parking_spaces'])
    cost_special_requests = 15 * to_int(predicted_facilities['total_of_special_requests'])
    
    total_simulated_revenue = predicted_revenue + cost_meal + cost_booking_changes + cost_parking + cost_special_requests
    
    # Tampilkan hasil
    st.subheader("Prediksi Potential Revenue (EUR)")
    st.write(f"€ {predicted_revenue:,.2f}")
    
    st.subheader("Prediksi Fasilitas Booking")
    st.write(f"Meal: {predicted_facilities['meal']}")
    st.write(f"Booking Changes: {predicted_facilities['booking_changes']}")
    st.write(f"Parking Spaces: {predicted_facilities['required_car_parking_spaces']}")
    st.write(f"Special Requests: {predicted_facilities['total_of_special_requests']}")
    
    st.subheader("Total Simulasi Revenue dengan Fasilitas Tambahan")
    st.write(f"€ {total_simulated_revenue:,.2f}")
