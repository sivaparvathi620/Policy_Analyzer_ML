import streamlit as st
from langdetect import detect
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    pipeline
)
# --- Using the standard 'requests' library for stable translation ---
import requests
import warnings
import time
import json 

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SUMMARIZATION_MODEL_NAME = "facebook/bart-large-cnn"
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

LANGUAGE_MAP = {
    "Auto-Detect": "auto",
    "English (en)": "en",
    "Hindi (hi)": "hi",
    "Telugu (te)": "te",
    "Spanish (es)": "es",
    "French (fr)": "fr",
}

# --- MODEL INITIALIZATION (Cached for Performance) ---

@st.cache_resource
def load_common_models():
    """Loads summarization and sentiment models."""
    st.info("Loading shared models (Summarization/Sentiment). This may take a moment on first run...")
    
    summary_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME)
    summary_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_NAME)
    
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model=SENTIMENT_MODEL_NAME, 
        tokenizer=SUMMARIZATION_MODEL_NAME # Use a stable tokenizer
    )
    return summary_tokenizer, summary_model, sentiment_pipeline

# --- CORE FUNCTIONS: STABLE TRANSLATION IMPLEMENTATION ---

# This function uses a reliable unofficial Google Translate endpoint via requests.
# It solves all ModuleNotFoundError, 404 errors, and gcloud authentication issues.
def run_translation(text, src_lang, target_lang):
    """Handles translation using a stable HTTP request to Google Translate."""
    
    if src_lang == target_lang or not text.strip():
        return text 
        
    try:
        # Google Translate endpoint structure
        url = "https://translate.googleapis.com/translate_a/single"
        
        # 'auto' for source detection is passed to the API
        src = src_lang if src_lang != 'auto' else 'auto'

        params = {
            'client': 'gtx',
            'sl': src, # Source language
            'tl': target_lang, # Target language
            'dt': 't',
            'q': text, # Text to translate
        }
        
        # Use a timeout for stability
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
        
        data = response.json()
        
        # The translated text is often fragmented, we join the pieces
        translated_text = "".join([sentence[0] for sentence in data[0]])
        
        return translated_text
        
    except requests.exceptions.Timeout:
        st.error("Translation failed: Request timed out. Try again or check network.")
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Translation failed due to connection error: {e}")
        return text
    except Exception as e:
        st.error(f"Translation failed: Could not parse response. {e}")
        return text 

# --- OTHER CORE FUNCTIONS ---

def detect_language(text):
    """Detects the language of the input text."""
    try:
        lang_code = detect(text[:500])
        return lang_code
    except:
        return "undetermined"

def summarize_text(text, summary_tokenizer, summary_model):
    """Generates a concise summary of the (English) text."""
    inputs = summary_tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    
    summary_ids = summary_model.generate(
        inputs['input_ids'], 
        num_beams=4, 
        min_length=50, 
        max_length=150, 
        do_sample=False, 
        early_stopping=True
    )
    summary_text = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

def analyze_policy(policy_text, selected_src_lang_code, selected_target_lang_code):
    """Executes the full analysis workflow."""
    
    summary_tokenizer, summary_model, sentiment_pipeline = load_common_models()
    
    # 1. Language Determination
    actual_src_lang = selected_src_lang_code
    if actual_src_lang == 'auto':
        actual_src_lang = detect_language(policy_text)
        st.subheader("1. Language Detection")
        st.write(f"**Source Language (Auto-Detected):** `{actual_src_lang.upper()}`")
    else:
        st.subheader("1. Language Detection")
        st.write(f"**Source Language (User Selected):** `{actual_src_lang.upper()}`")
    
    translated_text_for_analysis = policy_text
    
    # 2. Source -> English Translation (Required for Summarization/Sentiment)
    st.subheader("2. Source -> English Translation (For Analysis)")
    if actual_src_lang != 'en':
        translated_text_for_analysis = run_translation(policy_text, actual_src_lang, 'en')
        with st.expander("Show Translated English Text"):
            st.code(translated_text_for_analysis)
    else:
        st.info("Document is already in English, proceeding to analysis.")
        
    # 3. Policy Summarization (Run on English text)
    st.subheader("3. Policy Summarization (in English)")
    english_summary = summarize_text(translated_text_for_analysis, summary_tokenizer, summary_model)
    st.success(english_summary)
    
    # 4. Sentiment Analysis (Run on English summary)
    st.subheader("4. Sentiment Analysis")
    sentiment_result = sentiment_pipeline(english_summary)[0]
    st.metric(
        label="Overall Policy Tone",
        value=sentiment_result['label'],
        delta=f"Confidence: {sentiment_result['score']:.2f}"
    )

    # 5. Final Summary Translation (User's desired language)
    st.subheader(f"5. Final Summary Translation (to {selected_target_lang_code.upper()})")
    if selected_target_lang_code != 'en':
        final_summary = run_translation(english_summary, 'en', selected_target_lang_code)
        st.markdown(f"**Translated Summary:**")
        st.success(final_summary) 
    else:
        st.info("Target language is English. Final summary is above.")


# --- STREAMLIT UI ---

st.set_page_config(page_title="Policy Analyzer", layout="wide")
st.title("🌍 Multilingual Government Policy Analyzer")
st.markdown("Analyze policy from any source language (via text or file) and get the final summary translated to your preferred target language.")

# --- SIDEBAR INPUTS ---

# 1. Source Language
st.sidebar.header("1. Source Language")
src_lang_display_name = st.sidebar.selectbox(
    "Select Source Language",
    options=list(LANGUAGE_MAP.keys()),
    index=list(LANGUAGE_MAP.keys()).index("Telugu (te)") 
)
selected_src_lang_code = LANGUAGE_MAP[src_lang_display_name]

# 2. Target Language 
st.sidebar.header("2. Target Language")
target_lang_display_name = st.sidebar.selectbox(
    "Select Target Language for Final Summary",
    options=["English (en)", "Hindi (hi)", "Telugu (te)", "Spanish (es)"],
    index=0 
)
selected_target_lang_code = target_lang_display_name.split('(')[1].replace(')', '').strip()

# 3. Policy Input
st.sidebar.header("3. Policy Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload Policy Document (.txt only)", 
    type=['txt'] 
)

input_text = "" 
pasted_text_placeholder = "ప్రధానమంత్రి ఆవాస్ యోజన యొక్క లక్ష్యం ప్రతి పౌరుడికి ఇల్లు కల్పించడం." 

if uploaded_file is not None:
    # --- FILE UPLOAD PRIORITY ---
    try:
        input_text = uploaded_file.getvalue().decode("utf-8")
        st.sidebar.success("File Upload Successful!") 
        st.sidebar.text_area(
            "File Content Snippet:", 
            input_text[:300] + ("..." if len(input_text) > 300 else ""), 
            height=150, 
            disabled=True
        )
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        input_text = ""
        
else:
    # --- TEXT PASTE FALLBACK ---
    input_text = st.sidebar.text_area(
        "OR Paste Policy Text Here:",
        value=pasted_text_placeholder, 
        height=200
    )


# --- ANALYSIS EXECUTION ---
st.sidebar.markdown("---")
if st.sidebar.button("Analyze Policy", type="primary"):
    if input_text and len(input_text.strip()) > 10:
        # Run analysis
        with st.spinner('Analyzing document... this may take a moment for analysis.'):
            analyze_policy(input_text, selected_src_lang_code, selected_target_lang_code)
    else:
        st.warning("Please provide enough input (file upload or pasted text) to start the analysis.")