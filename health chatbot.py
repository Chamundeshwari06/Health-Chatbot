import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import csv
import pickle
from deep_translator import GoogleTranslator
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load dataset
training = pd.read_csv(r"C:\Users\bhuva\Downloads\Training.csv")
testing = pd.read_csv(r"C:\Users\bhuva\Downloads\Testing.csv")
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Encoding labels
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

# Load model
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Translate text
def translate_text(text, target_lang="hi"):
    translator = GoogleTranslator(source="auto", target=target_lang)
    return translator.translate(text)

# Streamlit UI
st.title("Health Chatbot")
st.write("Enter your symptoms and get a preliminary diagnosis.")

# User input
symptoms = st.text_input("Enter symptoms (comma-separated, e.g., fever, headache)")
language = st.selectbox("Select Language for Response", ["English", "Hindi"])

if st.button("Get Diagnosis"):
    if symptoms:
        model = load_model()
        input_features = np.zeros((1, len(model.feature_names_in_)))
        for symptom in symptoms.split(","):
            symptom = symptom.strip()
            if symptom in model.feature_names_in_:
                index = np.where(model.feature_names_in_ == symptom)[0][0]
                input_features[0, index] = 1
        
        prediction = model.predict(input_features)[0]
        response = f"Possible Condition: {le.inverse_transform([prediction])[0]}"
        
        if language == "Hindi":
            response = translate_text(response, "hi")
        
        st.success(response)
    else:
        st.error("Please enter at least one symptom.")

# Run the app using: streamlit run app.py
