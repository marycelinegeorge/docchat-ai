import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

# Section 1: Toxic Comment Detection
st.title("DocChat AI")

# CASE 1: Session Activation for history backup

if "tox_history" not in  st.session_state:
    st.session_state.tox_history = []

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# CASE 2: Activation Confidence Threshold Slider

confidence_threshold = st.slider(
    "Confidence Threshold",
    
    min_value= 0.0,
    max_value= 1.0,
    value = 0.5,
    step =0.05

    )
             
# SECTION 1: Toxic Comment Detection   

st.header("Toxic Comment Detection")
comment = st.text_area(
    "Enter Comment"
)

if st.button("Predict Toxicity"):
    payload = {
        "comment": comment
    }
    try:
        with st.spinner("Analyzing comment..."):
            response = requests.post(
                f"{API_URL}/predict",
                json = payload
            )

            result = response.json()
        confidence = result.get("confidence", 0)


        if confidence >= confidence_threshold:
            st.session_state.tox_history.append(result)
            st.success("Prediction Completed")
            st.write(result)

        else:
            st.warning("Prediction confidence below threshold.")
    except requests.exceptions.ConnectionError:

        st.error(
            "FastAPI server is offline."
        )

# SCENARIO -- Toxic Chat History
st.subheader("Toxic Prediction History")


for item in st.session_state.tox_history:

    st.write(item)

# SECTION 2: Document Question Answering  

st.header("Document Question Answering")

document = st.text_area(
    "Enter Document"
)
question = st.text_input(
    "Ask a Question"
)


if st.button("Get Answer"):
    payload = {
        "document": document,
        "question": question
    }

    try:
        with st.spinner("Generating answer..."):
            response = requests.post(
                f"{API_URL}/ask",
                json = payload)
            
            result = response.json()

        st.session_state.qa_history.append(result)
        st.success("Answer Generated")
        st.write(result)

    except requests.exceptions.ConnectionError:
        st.error(
            "FastAPI server is offline."
        )


# SCENARIO -- Q&A Chat History
st.subheader("Q&A History")

for item in st.session_state.qa_history:
    st.write(item)

