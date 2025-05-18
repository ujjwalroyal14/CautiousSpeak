import streamlit as st
import joblib
import pandas as pd

# Load model and vectorizer
@st.cache_resource
def load_components():
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    model = joblib.load('lda_model.joblib')
    return vectorizer, model

vectorizer, model = load_components()

# Streamlit UI
st.title('🚨 Hate Speech Detection')
st.write("Enter text to check for hate speech:")

user_input = st.text_area("Input Text", "", height=150)

if st.button('Analyze'):
    if user_input:
        # Transform and predict
        X = vectorizer.transform([user_input])
        prediction = model.predict(X.toarray())
        proba = model.predict_proba(X.toarray())[0]
        
        # Display results
        st.subheader("Results")
        if prediction[0] == 1:
            st.error("Hate speech detected!")
        else:
            st.success("No hate speech detected.")
        
        st.write(f"Confidence: {max(proba)*100:.1f}%")
        
        # Show probability breakdown
        with st.expander("Detailed Analysis"):
            st.write(f"Non-hate probability: {proba[0]*100:.1f}%")
            st.write(f"Hate speech probability: {proba[1]*100:.1f}%")
    else:
        st.warning("Please enter some text to analyze")