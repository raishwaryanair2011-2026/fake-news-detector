import streamlit as st
import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer

st.set_page_config(page_title="Fake News Detector")

# load model
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("AI Fake News & Misinformation Detector")

st.write("Enter a news article to check if it is REAL or FAKE.")

text = st.text_area("Enter news text", height=220)

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter text")
        st.stop()

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    label = "REAL" if pred == 1 else "FAKE"
    confidence = float(np.max(prob))

    st.subheader("Prediction Result")

    if label == "REAL":
        st.success(f"Prediction: {label}")
    else:
        st.error(f"Prediction: {label}")

    st.write(f"Confidence Score: {confidence:.2f}")
    st.progress(confidence)

    # LIME explanation
    st.subheader("Important Words")

    explainer = LimeTextExplainer(class_names=["Fake","Real"])

    exp = explainer.explain_instance(
        text,
        lambda x: model.predict_proba(vectorizer.transform(x)),
        num_features=8
    )

    for word, weight in exp.as_list():
        if weight > 0:
            st.markdown(f"<span style='color:green'>{word}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:red'>{word}</span>", unsafe_allow_html=True)

    st.subheader("Explanation")

    if label == "FAKE":
        st.write("The article contains sensational or misleading patterns.")
    else:
        st.write("The article follows factual reporting style.")

st.markdown("---")
st.caption("Model: TF-IDF + Logistic Regression | Explainable AI: LIME")
