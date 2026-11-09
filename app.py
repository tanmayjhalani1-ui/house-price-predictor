import json

import pandas as pd
import streamlit as st

from src.inference import load_model, predict_price
from src.train import DATA_PATH, METADATA_PATH

st.set_page_config(page_title="Housing Price Intelligence", page_icon="🏠", layout="wide")
st.title("Housing Price Intelligence")
st.caption("Estimate house prices using property attributes and compare model performance.")

try:
    model = load_model()
except FileNotFoundError:
    st.warning("No trained model found. Train the model first.")
    st.code("python -m src.train")
    st.stop()

metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
metrics = metadata.get("holdout_metrics", {})
top = st.columns(4)
top[0].metric("Dataset records", f"{metadata.get('dataset_rows', 0):,}")
top[1].metric("Selected model", metadata.get("selected_model", "N/A").replace("_", " ").title())
top[2].metric("Holdout RMSE", f"{metrics.get('rmse', 0):,.0f}")
top[3].metric("Holdout R²", f"{metrics.get('r2', 0):.3f}")

predict_tab, explore_tab, model_tab = st.tabs(["Price estimator", "Housing market overview", "Model evaluation"])

with predict_tab:
    st.subheader("Property details")
    with st.form("property_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            area = st.number_input("Area (square feet)", min_value=100, max_value=50000, value=5000, step=100)
            bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
            bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
            stories = st.selectbox("Stories", [1, 2, 3, 4], index=1)
        with c2:
            parking = st.selectbox("Parking spaces", [0, 1, 2, 3], index=1)
            mainroad = st.selectbox("Main road access", ["yes", "no"])
            guestroom = st.selectbox("Guest room", ["yes", "no"])
            basement = st.selectbox("Basement", ["yes", "no"])
        with c3:
            hotwater = st.selectbox("Hot water heating", ["yes", "no"])
            aircon = st.selectbox("Air conditioning", ["yes", "no"])
            prefarea = st.selectbox("Preferred area", ["yes", "no"])
            furnishing = st.selectbox("Furnishing", ["furnished", "semi-furnished", "unfurnished"])
        submitted = st.form_submit_button("Estimate property price", type="primary", use_container_width=True)
    if submitted:
        house = {
            "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
            "stories": stories, "parking": parking, "mainroad": mainroad,
            "guestroom": guestroom, "basement": basement,
            "hotwaterheating": hotwater, "airconditioning": aircon,
            "prefarea": prefarea, "furnishingstatus": furnishing,
        }
        result = predict_price(model, house)
        st.success(f"Estimated price: {result['predicted_price']:,.0f} {result['currency']}")

with explore_tab:
    st.subheader("Dataset overview")
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
        a, b, c = st.columns(3)
        a.metric("Properties", f"{len(data):,}")
        b.metric("Median listed price", f"{data['price'].median():,.0f}")
        c.metric("Median area", f"{data['area'].median():,.0f}")
        st.write("Price distribution")
        st.histogram if False else None
        st.bar_chart(data.groupby("bedrooms")["price"].median().rename("Median price by bedrooms"))
        st.write("Median price by furnishing status")
        st.bar_chart(data.groupby("furnishingstatus")["price"].median())
        st.write("Area and price")
        st.scatter_chart(data[["area", "price"]])
    else:
        st.info("Dataset not found.")

with model_tab:
    st.subheader("Held-out evaluation")
    if metrics:
        cols = st.columns(3)
        cols[0].metric("MAE", f"{metrics['mae']:,.0f}")
        cols[1].metric("RMSE", f"{metrics['rmse']:,.0f}")
        cols[2].metric("R²", f"{metrics['r2']:.3f}")
        cv = metadata.get("cross_validation", {})
        if cv:
            st.write("Cross-validation RMSE by model (lower is better)")
            st.bar_chart(pd.Series({name: value["mean_cv_rmse"] for name, value in cv.items()}))
        st.caption("Price units follow the source dataset. Metrics are estimates on one held-out split, not a guarantee of market accuracy.")
