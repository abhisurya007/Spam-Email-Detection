import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load('svm_spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit App
st.title("Email Spam Classifier")
st.write("Enter an email below to check if it's Spam or Not Spam.")

# Input Text Box
input_text = st.text_area("Email Text:", "")

# Prediction Button
if st.button("Predict"):
    if input_text:  # Ensure input isn't empty
        # Transform the input text
        input_vectorized = vectorizer.transform([input_text])
        
        # Make a prediction
        prediction = model.predict(input_vectorized)
        
        # Show result
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        st.success(f"Prediction: **{result}**")
    else:
        st.warning("Please enter some text to predict.")
