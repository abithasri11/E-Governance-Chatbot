import streamlit as st

from src.ml_classifier import MLClassifier
from src.rag_system import RAGSystem
from src.chatbot import EGovernanceChatbot


DATA_PATH = "data"


@st.cache_resource
def load_chatbot():

    ml = MLClassifier(DATA_PATH)
    ml.train()

    rag = RAGSystem(DATA_PATH)

    return EGovernanceChatbot(ml, rag)


chatbot = load_chatbot()


st.sidebar.title("E-Governance Chatbot")

page = st.sidebar.radio(
    "Select Page",
    ["🤖 Chatbot", "⚙️ Hyperparameter Tuning"]
)


# ---------------- CHATBOT ----------------

if page == "🤖 Chatbot":

    st.title("E-Governance Chatbot")

    st.write(
        "Ask questions about Tamil Nadu e-Sevai services."
    )

    question = st.text_input(
        "Enter your question:"
    )

    if st.button("Ask"):

        if question.strip():

            result = chatbot.answer(question)

            st.subheader("Chatbot Response")

            st.write(
                result["document"]["text"]
            )

            st.subheader("Technical Details")

            st.write(
                "Predicted Service:",
                result["intent"]
            )

            st.write(
                "Confidence:",
                round(result["confidence"], 3)
            )

        else:

            st.warning(
                "Please enter a question."
            )


# -------- HYPERPARAMETER TUNING --------

else:

    st.title("Logistic Regression Hyperparameter Tuning")

    st.write(
        "Comparison of different Logistic Regression "
        "hyperparameter configurations."
    )

    st.subheader("Tuning Results")

    st.dataframe(
        chatbot.ml.tuning_results.round(3),
        use_container_width=True
    )

    st.subheader("Best Hyperparameters")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "C",
            chatbot.ml.best_params["C"]
        )

    with col2:
        st.metric(
            "Class Weight",
            chatbot.ml.best_params["Class Weight"]
        )

    st.subheader("Final Model Evaluation")

    metrics = chatbot.ml.evaluation_metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.3f}")

    with col2:
        st.metric("Precision", f"{metrics['Precision']:.3f}")

    with col3:
        st.metric("Recall", f"{metrics['Recall']:.3f}")

    with col4:
        st.metric("F1 Score", f"{metrics['F1 Score']:.3f}")