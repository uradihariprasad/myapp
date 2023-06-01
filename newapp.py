import streamlit as st
import pandas as pd
import numpy as np
import pickle
from decisiontree import DecisionTree


# Define a function to make predictions
def predict_churn(data):
    # Load the saved model
    with open('chaid_model.pkl', 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict(data)
    print(prediction)
    if prediction[0] == 0:
        return "No"
    else:
        return "Yes"

# Define the Streamlit app
def main():
    # Set the app title
    st.set_page_config(page_title='Bank Marketing Predictions', page_icon=':money_with_wings:')

    # Set the app heading
    st.title('Bank Marketing Predictions')

    # Define the input fields for the app
    age = st.slider("Age", 18, 100)
    job = st.selectbox("Job", ["Admin.", "Blue-collar", "Entrepreneur", "Housemaid", "Management", "Retired", "Self-employed", "Services", "Student", "Technician", "Unemployed", "Unknown"])
    marital = st.selectbox("Marital Status", ["Divorced", "Married", "Single", "Unknown"])
    education = st.selectbox("Education Level", ["Basic 4y", "Basic 6y", "Basic 9y", "High School", "Professional Course", "University Degree", "Unknown", "Illiterate"])
    default = st.radio("Has Credit in Default?", ("No", "Yes", "Unknown"))
    housing = st.radio("Has Housing Loan?", ("No", "Yes", "Unknown"))
    loan = st.radio("Has Personal Loan?", ("No", "Yes", "Unknown"))
    contact = st.selectbox("Contact Communication Type", ["Cellular", "Telephone"])
    month = st.selectbox("Last Contact Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    day_of_week = st.selectbox("Last Contact Day of the Week", ["Mon", "Tue", "Wed", "Thu", "Fri"])
    duration = st.slider("Last Contact Duration (Seconds)", 0, 5000)
    campaign = st.slider("Number of Contacts During Campaign", 1, 50)
    pdays = st.slider("Days Since Last Contact", 0, 1000)
    previous = st.slider("Number of Contacts Before This Campaign", 0, 10)
    poutcome = st.selectbox("Outcome of Previous Marketing Campaign", ["Failure", "Nonexistent", "Success"])
    emp_var_rate = st.slider("Employment Variation Rate", -4.0, 1.0, -2.0)
    cons_price_idx = st.slider("Consumer Price Index", 92.0, 95.0, 93.0)
    cons_conf_idx = st.slider("Consumer Confidence Index", -60.0, -20.0, -40.0)
    euribor3m = st.slider("Euribor 3 Month Rate", 0.0, 5.0, 2.0)
    nr_employed = st.slider("Number of Employees", 5000, 5500, 5200)

    # Create a dictionary with the input data
    data = {"age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "housing": housing,
            'loan': loan,
            'contact': contact,
            'month': month,
            'day_of_week': day_of_week,
            'duration': duration,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'poutcome': poutcome,
            'emp.var.rate': emp_var_rate,
            'cons.price.idx': cons_price_idx,
            'cons.conf.idx': cons_conf_idx,
            'euribor3m': euribor3m,
            'nr.employed': nr_employed}
# Convert the dictionary into a dataframe
    data_df = pd.DataFrame(data, index=[0])

# Apply one-hot encoding to the categorical columns
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'poutcome']

    data_encoded = pd.get_dummies(data_df, columns=cat_cols)

    prediction=""
    if st.button("Predict"):
        prediction=predict_churn(data_encoded)
        st.success(f"The predicted status is: {prediction}")
    

if __name__=='__main__':
    main()

            