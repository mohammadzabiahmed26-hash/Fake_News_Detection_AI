import streamlit as st
import pickle

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# =========================================================
# LOAD TRAINED MODEL
# =========================================================
with open("fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)
# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.main {
    padding: 1rem 2rem;
}

/* Main Header */
.hero {
    text-align: center;
    padding: 25px 10px 15px 10px;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 18px;
    opacity: 0.75;
}

/* Cards */
.info-card {
    padding: 18px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Result */
.result-card {
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    margin-top: 20px;
    border: 1px solid rgba(128,128,128,0.25);
}

.result-title {
    font-size: 32px;
    font-weight: 800;
}

.confidence {
    font-size: 20px;
    font-weight: 600;
}

/* Footer */
.footer {
    text-align: center;
    opacity: 0.65;
    font-size: 13px;
    padding: 20px;
}

/* Mobile */
@media (max-width: 600px) {
    .hero-title {
        font-size: 34px;
    }

    .hero-subtitle {
        font-size: 15px;
    }
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">📰 Fake News Detector</div>
    <div class="hero-subtitle">
        AI & Machine Learning Based News Classification System
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================================================
# PROJECT INFORMATION
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Technology", "AI / ML")

with col2:
    st.metric("NLP", "TF-IDF")

with col3:
    st.metric("Task", "Classification")

st.divider()

# =========================================================
# NEWS INPUT
# =========================================================
st.subheader("🔎 Analyze News Article")

st.write(
    "Enter or paste a news article below. "
    "The trained Machine Learning model will classify it as Fake or Real."
)

news_text = st.text_area(
    "News Article",
    height=240,
    placeholder="Paste the complete news article here...",
    label_visibility="collapsed"
)

# Character counter
character_count = len(news_text)

st.caption(f"📝 Characters entered: {character_count}")

# =========================================================
# BUTTONS
# =========================================================
col1, col2 = st.columns(2)

with col1:
    analyze = st.button(
        "🔍 Analyze News",
        use_container_width=True,
        type="primary"
    )

with col2:
    reset = st.button(
        "🔄 Clear",
        use_container_width=True
    )

if reset:
    st.rerun()

# =========================================================
# PREDICTION
# =========================================================
if analyze:

    if news_text.strip() == "":
        st.warning("⚠️ Please enter a news article before analyzing.")

    else:

        with st.spinner("🤖 AI is analyzing the news..."):

            # Convert text into TF-IDF features
            news_vector = vectorizer.transform([news_text])

            # Prediction
            prediction = model.predict(news_vector)[0]

            # Probability
            probabilities = model.predict_proba(news_vector)[0]

            confidence = max(probabilities) * 100

        st.divider()

        # =================================================
        # FAKE RESULT
        # =================================================
        if prediction == "FAKE":

            st.markdown("""
            <div class="result-card">
                <div class="result-title">🚨 FAKE NEWS</div>
                <br>
                <div class="confidence">
                    Model Confidence
                </div>
            </div>
            """, unsafe_allow_html=True)

        # =================================================
        # REAL RESULT
        # =================================================
        else:

            st.markdown("""
            <div class="result-card">
                <div class="result-title">✅ REAL NEWS</div>
                <br>
                <div class="confidence">
                    Model Confidence
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Confidence percentage
        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        # Confidence progress bar
        st.progress(min(confidence / 100, 1.0))

        # =================================================
        # MODEL PROBABILITIES
        # =================================================
        st.subheader("📊 Classification Details")

        fake_probability = 0
        real_probability = 0

        classes = list(model.classes_)

        for i, class_name in enumerate(classes):

            if class_name == "FAKE":
                fake_probability = probabilities[i] * 100

            elif class_name == "REAL":
                real_probability = probabilities[i] * 100

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🚨 Fake Probability",
                f"{fake_probability:.2f}%"
            )

        with col2:
            st.metric(
                "✅ Real Probability",
                f"{real_probability:.2f}%"
            )

        # =================================================
        # DISCLAIMER
        # =================================================
        st.info(
            "ℹ️ This prediction is generated by a Machine Learning "
            "model and should be used as an automated classification "
            "aid. Always verify important news using reliable sources."
        )

# =========================================================
# ABOUT PROJECT
# =========================================================
st.divider()

with st.expander("📚 About This Project"):

    st.write("""
    **Fake News Detection using Artificial Intelligence**

    This project uses Natural Language Processing (NLP) and
    Machine Learning to classify news articles as FAKE or REAL.

    **Technologies Used:**
    - Python
    - Machine Learning
    - Natural Language Processing
    - TF-IDF Vectorization
    - Streamlit
    - Scikit-learn

    The news text is converted into numerical TF-IDF features
    and then passed to the trained Machine Learning model for
    classification.
    """)

# =========================================================
# FOOTER
# =========================================================
st.divider()

st.markdown("""
<div class="footer">
    Developed using Python • Machine Learning • NLP • TF-IDF • Streamlit
    <br><br>
    🎓 AI/ML Academic Project
</div>
""", unsafe_allow_html=True)
