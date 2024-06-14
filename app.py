import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('fine_tuned_model')
model = GPT2LMHeadModel.from_pretrained('fine_tuned_model')

def generate_response(query):
    inputs = tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(inputs['input_ids'], max_length=150, num_beams=2, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Streamlit app
st.title("RAG-Powered Chatbot")

query = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if query:
        answer = generate_response(query)
        st.text_area("Answer:", value=answer, height=150)
