import streamlit as st
import joblib
import os
import re

# Load the trained model and vectorizer
model_path = 'C:\\Users\\abhis\\Downloads\\machine\\Spam Email Detection\\Spam Email Detection\\model.pkl'
vectorizer_path = 'C:\\Users\\abhis\\Downloads\\machine\\Spam Email Detection\\Spam Email Detection\\vectorizer.pkl'

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"Model file not found at {model_path}")
    st.stop()

if os.path.exists(vectorizer_path):
    vectorizer = joblib.load(vectorizer_path)
else:
    st.error(f"Vectorizer file not found at {vectorizer_path}")
    st.stop()

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove all non-word characters
    text = text.strip()  # Remove leading and trailing whitespace
    return text

# Streamlit app
def main():
    st.title("Spam Email Detection")
    st.write("### Enter text below to check if it's Spam or Not Spam.")

    # User input
    user_input = st.text_area("Enter your email text here:", "")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text for prediction.")
        else:
            try:
                # Preprocess the input
                preprocessed_text = preprocess_text(user_input)

                # Transform input using vectorizer
                data_vectorized = vectorizer.transform([preprocessed_text])

                # Make prediction
                prediction = model.predict(data_vectorized)
                prediction_label = "Spam" if prediction[0] == 1 else "Not Spam"

                # Display result
                st.success(f"Prediction: **{prediction_label}**")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
