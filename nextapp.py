import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings


# Load the dataset
cancer_data = pd.read_csv('datasetcancer.CSV', header=None, names=['Age', 'Gender', 'Alcohol use', 'Dust Allergy', 'Occupational Hazards', 'Genetic Risk', 'Chronic Lung Disease','Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss','Shortness of Breath', 'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring', 'Level'])

# Remove rows with 'Level' as the header
cancer_data = cancer_data[cancer_data['Level'] != 'Level']
# Convert 'Age' column to numeric
cancer_data['Age'] = pd.to_numeric(cancer_data['Age'], errors='coerce')
# Split features and target variable
X = cancer_data.drop(columns='Level', axis=1)
Y = cancer_data['Level']

# Split the data into Training Data & Test Data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)

# Streamlit UI
st.set_page_config(page_title='Lung Cancer Prediction', page_icon=':lungs:')
st.title('Lung Cancer Prediction')

# Input form for user
age = st.number_input('Age', min_value=1, max_value=100, step=1)
gender = st.radio('Gender', ['Male', 'Female'], index=0)

st.subheader('Lifestyle Factors:')
alcohol_use = st.number_input('Alcohol Use', min_value=1, max_value=9, step=1)
dust_allergy = st.number_input('Dust Allergy', min_value=1, max_value=9, step=1)
occupational_hazards = st.number_input('Occupational Hazards', min_value=1, max_value=9, step=1)
genetic_risk = st.number_input('Genetic Risk', min_value=1, max_value=9, step=1)
chronic_lung_disease = st.number_input('Chronic Lung Disease', min_value=1, max_value=9, step=1)
balanced_diet = st.number_input('Balanced Diet', min_value=1, max_value=9, step=1)
obesity = st.number_input('Obesity', min_value=1, max_value=9, step=1)
smoking = st.number_input('Smoking', min_value=1, max_value=9, step=1)
passive_smoker = st.number_input('Passive Smoker', min_value=1, max_value=9, step=1)

st.subheader('Symptoms:')
chest_pain = st.number_input('Chest Pain', min_value=1, max_value=9, step=1)
coughing_of_blood = st.number_input('Coughing of Blood', min_value=1, max_value=9, step=1)
fatigue = st.number_input('Fatigue', min_value=1, max_value=9, step=1)
weight_loss = st.number_input('Weight Loss', min_value=1, max_value=9, step=1)
shortness_of_breath = st.number_input('Shortness of Breath', min_value=1, max_value=9, step=1)
wheezing = st.number_input('Wheezing', min_value=1, max_value=9, step=1)
swallowing_difficulty = st.number_input('Swallowing Difficulty', min_value=1, max_value=9, step=1)
clubbing_of_finger_nails = st.number_input('Clubbing of Finger Nails', min_value=1, max_value=9, step=1)
frequent_cold = st.number_input('Frequent Cold', min_value=1, max_value=9, step=1)
dry_cough = st.number_input('Dry Cough', min_value=1, max_value=9, step=1)
snoring = st.number_input('Snoring', min_value=1, max_value=9, step=1)

# Make prediction on user input
if st.button('Predict'):
    input_data = np.array([[age, 1 if gender == 'Male' else 2, alcohol_use, dust_allergy, occupational_hazards, genetic_risk, chronic_lung_disease, balanced_diet, obesity, smoking, passive_smoker, chest_pain, coughing_of_blood, fatigue, weight_loss, shortness_of_breath, wheezing, swallowing_difficulty, clubbing_of_finger_nails, frequent_cold, dry_cough, snoring]])
    prediction = model.predict(input_data)

    # Display the prediction result creatively
    st.subheader('Prediction Result:')
    if prediction[0] == 'High':
        st.error('The person has high chances of lung cancer :disappointed_relieved:')
    elif prediction[0] == 'Low':
        st.success('The person has low chances of having lung cancer :smile:')
    else:
        st.warning('The person has medium chances of having lung cancer :neutral_face:')

    # Switch to another page
    st.markdown('<hr>', unsafe_allow_html=True)
    st.title('Thank You for Using Lung Cancer Prediction App!')
    st.write('If you have any concerns, please consult with a healthcare professional.')

# Display additional information about the dataset
st.sidebar.title('Dataset Information')
st.sidebar.subheader('Number of rows and columns:')
st.sidebar.write(cancer_data.shape)

st.sidebar.subheader('Level distribution:')
st.sidebar.write(cancer_data['Level'].value_counts())

st.sidebar.subheader('Statistical measures about the dataset:')
st.sidebar.write(cancer_data.describe())
