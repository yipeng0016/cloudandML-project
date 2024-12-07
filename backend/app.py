import os
import streamlit as st
import requests

# Streamlit app title
st.title("T5 Translation & Fill-Mask API with Streamlit")

# Define global variables for Ingress
INGRESS_HOST = os.getenv("INGRESS_HOST", "localhost")  # Default to localhost
INGRESS_PORT = os.getenv("INGRESS_PORT", "80")        # Default to port 80

# Define URLs and hostnames for services
translate_url = f"http://{INGRESS_HOST}:{INGRESS_PORT}/openai/v1/completions"
fillmask_url = f"http://{INGRESS_HOST}:{INGRESS_PORT}/v1/models/albert:predict"

# Hostnames specific to each service
translate_service_hostname = os.getenv("TRANSLATE_SERVICE_HOSTNAME", "huggingface-t5.kserve-test.example.com")
fillmask_service_hostname = os.getenv("FILLMASK_SERVICE_HOSTNAME", "huggingface-albert.kserve-test.example.com")

# Tabbed layout for multiple features
tab1, tab2 = st.tabs(["Translation", "Fill-Mask"])

# **Tab 1: Translation**
with tab1:
    st.subheader("Translation Feature")
    
    # Supported languages
    languages = ["English", "French", "German", "Romanian"]

    # Dropdowns for source and target languages
    source_language = st.selectbox("Select the source language:", languages, key="src")
    target_language = st.selectbox("Select the target language:", languages, key="tgt")

    # Ensure the source and target languages are not the same
    if source_language == target_language:
        st.warning("Source and target languages must be different.")

    # Text area for user input
    prompt_text = st.text_area("Enter the text to translate:", key="translate")

    # Streamlit button for translation
    if st.button("Translate", key="translate_btn"):
        if prompt_text.strip() and source_language != target_language:
            # Prepare the payload
            payload = {
                "model": "t5",
                "prompt": f"translate {source_language} to {target_language}: {prompt_text}",
                "stream": False,
                "max_tokens": 30,
            }

            # Make the API call
            try:
                response = requests.post(
                    translate_url,
                    headers={
                        "Content-Type": "application/json",
                        "Host": translate_service_hostname
                    },
                    json=payload
                )
                if response.status_code == 200:
                    # Parse the response
                    translation = (
                        response.json()
                        .get("choices", [{}])[0]
                        .get("text", "No translation provided")
                    )
                    st.success(f"Translation: {translation}")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error while calling the API: {e}")
        else:
            if not prompt_text.strip():
                st.warning("Please enter some text to translate.")

# **Tab 2: Fill-Mask**
with tab2:
    st.subheader("Fill-Mask Feature")

    # Text input for Fill-Mask functionality
    fill_mask_text = st.text_area(
        "Enter text with a [MASK] token to predict:",
        value="The capital of France is [MASK].",
    )

    # Streamlit button for Fill-Mask
    if st.button("Predict Mask"):
        if "[MASK]" in fill_mask_text:
            # Prepare the payload in the correct format
            payload = {
                "instances": [fill_mask_text]  # List[str] as required by the API
            }

            # Make the API call
            try:
                response = requests.post(
                    fillmask_url,  # fillmask_url should point to /v1/models/albert:predict
                    headers={
                        "Content-Type": "application/json",
                        "Host": fillmask_service_hostname  # Must match the service hostname
                    },
                    json=payload
                )

                # Process response
                if response.status_code == 200:
                    # Parse predictions from the API response
                    predictions = response.json().get("predictions", [])
                    if predictions:
                        st.success(f"Predicted token(s): {', '.join(predictions)}")
                    else:
                        st.warning("No predictions returned from the API.")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error while calling the API: {e}")
        else:
            st.warning("Please include a [MASK] token in your text.")

