import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Function to load the MarianMT model and tokenizer
@st.cache_resource
def load_model(source_lang, target_lang):
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Function to translate text using the model and tokenizer
def translate_text(text, model, tokenizer):
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

# Streamlit UI
st.title("French-English Language Translator")

direction = st.selectbox("Translation Direction", ["French to English", "English to French"])

text_to_translate = st.text_area("Enter text to translate")

if st.button("Translate"):
    if direction == "French to English":
        model, tokenizer = load_model("fr", "en")
    else:
        model, tokenizer = load_model("en", "fr")
    
    translation = translate_text(text_to_translate, model, tokenizer)
    st.success(translation[0])
