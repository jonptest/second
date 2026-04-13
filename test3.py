import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('titanic_model.pkl')

st.title("Titanic Survival Predictor")
st.write("Enter passenger details to see if they would have survived the disaster.")

# Sidebar for inputs
st.sidebar.header("Passenger Details")
pclass = st.sidebar.selectbox("Ticket Class (1st, 2nd, 3rd)", [1, 2, 3])
sex = st.sidebar.selectbox("Gender", ["male", "female"])
age = st.sidebar.slider("Age", 0, 100, 25)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 10, 0)

# Preprocess inputs
sex_encoded = 1 if sex == "male" else 0
input_data = pd.DataFrame([[pclass, age, sibsp, parch, sex]], 
                          columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Sex_male'])

# Predict Button
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # Probability of Class 1 (Survived)

    if prediction == 1:
        st.success(f"Result: **Survived**")
    else:
        st.error(f"Result: **Did Not Survive**")
    
    st.metric(label="Survival Probability", value=f"{probability:.2%}")
