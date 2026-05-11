import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Prediction App")
st.write("This app predicts whether a Titanic passenger would survive using a Logistic Regression model.")

@st.cache_resource
def load_model():
    return joblib.load("titanic_logistic_regression_model.pkl")

try:
    model = load_model()

    st.sidebar.header("Passenger Details")

    pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
    sex = st.sidebar.selectbox("Sex", ["male", "female"])
    age = st.sidebar.slider("Age", 0, 100, 25)
    sibsp = st.sidebar.number_input("Siblings / Spouses Aboard", min_value=0, max_value=10, value=0)
    parch = st.sidebar.number_input("Parents / Children Aboard", min_value=0, max_value=10, value=0)
    fare = st.sidebar.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
    embarked = st.sidebar.selectbox("Embarked", ["S", "C", "Q"])

    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    st.subheader("Input Data")
    st.dataframe(input_data)

    if st.button("Predict Survival"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"The passenger is likely to survive. Survival probability: {probability:.2f}")
        else:
            st.error(f"The passenger is unlikely to survive. Survival probability: {probability:.2f}")

except FileNotFoundError:
    st.error("Model file not found. Please run the notebook first to create titanic_logistic_regression_model.pkl.")