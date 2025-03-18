import cv2
import easyocr
import streamlit as st
from PIL import Image
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# Image preprocessing
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = cv2.bitwise_not(img_bin)
    return img_bin

# Extract text using EasyOCR
def extract_text_from_image(image):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image)
    return " ".join([res[1] for res in results])

# Set up Ollama with Langchain to extract relevant questions
def send_to_ollama_with_langchain(extracted_text, user_question):
    llm = Ollama(model='llama2')

    # Define the prompt
    chat_prompt = ChatPromptTemplate.from_template(
        "You are an assistant helping a user. Extract only the questions from the following text related to the topic: {topic}. \n\nText: {extracted_text}"
    )

    output_parser = StrOutputParser()
    chain = chat_prompt | llm | output_parser

    # Invoke the chain with extracted text and user query
    response = chain.invoke({"topic": user_question, "extracted_text": extracted_text})
    return response

# Streamlit UI
st.title("📄 Question Extractor from Image (Topic-based)")
st.write("Upload your question paper image and ask a topic-related question!")

# Upload image
uploaded_file = st.file_uploader("Upload a question paper image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Question Paper", use_column_width=True)

    # Process the image
    st.write("Processing the image... 🔍")
    preprocessed_image = preprocess_image(image)
    extracted_text = extract_text_from_image(preprocessed_image)

    st.success("✅ Text extracted successfully!")
    st.text_area("Extracted Text:", extracted_text, height=200)

    # Take user query for a specific topic
    user_question = st.text_input("Ask about a specific topic:")

    if user_question:
        # Send to Ollama with Langchain for topic-based extraction
        st.write("🤖 Finding relevant questions for your topic...")
        llm_response = send_to_ollama_with_langchain(extracted_text, user_question)

        # Display results
        st.subheader("🎯 Relevant Questions Found:")
        st.write(llm_response)

