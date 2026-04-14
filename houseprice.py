import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Housing Price Predictor")

st.title("Housing Price Prediction")

df = pd.read_csv("Housing.csv")

df = pd.get_dummies(df, drop_first=True)

X = df.drop("price", axis=1)
y = df["price"]

@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

if "ready" not in st.session_state:
    st.session_state.ready = False

if st.button("Train Model"):
    model = train_model()
    st.session_state.model = model
    st.session_state.ready = True
    st.success("Model trained")

if st.session_state.ready:

    st.subheader("Enter House Details")

    area = st.number_input("Area", value=5000)
    bedrooms = st.number_input("Bedrooms", value=3)
    bathrooms = st.number_input("Bathrooms", value=2)
    stories = st.number_input("Stories", value=2)
    parking = st.number_input("Parking", value=1)

    mainroad = st.selectbox("Main Road", ["yes", "no"])
    guestroom = st.selectbox("Guest Room", ["yes", "no"])
    basement = st.selectbox("Basement", ["yes", "no"])
    hotwater = st.selectbox("Hot Water Heating", ["yes", "no"])
    aircon = st.selectbox("Air Conditioning", ["yes", "no"])
    prefarea = st.selectbox("Preferred Area", ["yes", "no"])
    furnishing = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

    mainroad_yes = 1 if mainroad == "yes" else 0
    guestroom_yes = 1 if guestroom == "yes" else 0
    basement_yes = 1 if basement == "yes" else 0
    hotwater_yes = 1 if hotwater == "yes" else 0
    aircon_yes = 1 if aircon == "yes" else 0
    prefarea_yes = 1 if prefarea == "yes" else 0

    furnishing_semi = 1 if furnishing == "semi-furnished" else 0
    furnishing_unfurnished = 1 if furnishing == "unfurnished" else 0

    if st.button("Predict"):

        input_data = np.array([[area, bedrooms, bathrooms, stories, parking,
                                mainroad_yes, guestroom_yes, basement_yes,
                                hotwater_yes, aircon_yes, prefarea_yes,
                                furnishing_semi, furnishing_unfurnished]])

        prediction = st.session_state.model.predict(input_data)[0]

        st.subheader(f"Estimated Price: {int(prediction)}")

else:
    st.info("Train the model first")