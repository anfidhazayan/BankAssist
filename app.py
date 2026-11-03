import streamlit as st
import joblib
st.set_page_config(
    page_title="BankAssist",
    page_icon="🏦",
    layout="centered"
)
model = joblib.load("models/bankassist_model.pkl")
st.markdown(
    '<div class="main-title">🏦 BankAssist</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Intelligent Banking Query Classification</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "Your intelligent banking assistant. "
    "Enter a banking-related question to identify its intent."
)
st.divider()

st.header("Ask BankAssist")
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

def clear_query():
    st.session_state.user_query = ""

user_query = st.text_area(
    "Enter your banking question:",
    placeholder="Example: My card hasn't arrived yet...",
    key="user_query"
)

st.button("Clear", on_click=clear_query)
if st.button("Predict Intent"):

    if user_query.strip():

        # Predict the most likely intent
        prediction = model.predict([user_query])

        # Get probabilities for all 77 categories
        probabilities = model.predict_proba([user_query])[0]

        # Get category names
        categories = model.classes_

        # Find indices of the top 3 probabilities
        top_indices = probabilities.argsort()[-3:][::-1]

        # Display main prediction
        confidence = probabilities[top_indices[0]] * 100

        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">PREDICTED BANKING INTENT</div>
            <div class="prediction-value">{prediction[0]}</div>
        </div>
        """, unsafe_allow_html=True)

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        # Display top 3 predictions
        st.caption("Here are the three most likely categories predicted by the model.")

        for index in top_indices:

            category = categories[index]
            probability = probabilities[index] * 100

            st.write(f"**{category}** — {probability:.2f}%")

            st.progress(float(probabilities[index]))

        # Confidence warning
        if confidence < 50:
            st.warning("Low confidence! Please verify this prediction.")

        elif confidence < 75:
            st.info("Moderate confidence. Verification is recommended.")

        else:
            st.success("High model confidence!")

    else:
        st.warning("Please enter a banking query.")
st.markdown("""
<style>

.stApp {
    background-color: #F5F8FC;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #17365D;
    text-align: center;
}

.subtitle {
    font-size: 17px;
    color: #64748B;
    text-align: center;
}

.result-card {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    margin-top: 20px;
}
.prediction-card {
    background-color: white;
    padding: 22px;
    border-radius: 12px;
    border-left: 5px solid #2563EB;
    margin: 15px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.prediction-label {
    color: #64748B;
    font-size: 14px;
}

.prediction-value {
    color: #17365D;
    font-size: 25px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)