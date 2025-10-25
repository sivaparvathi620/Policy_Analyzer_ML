import torch
from langdetect import detect
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    pipeline
)

# --- CONFIGURATION ---
TRANSLATION_MODEL_NAME = 'Helsinki-NLP/opus-mt-hi-en'
SUMMARIZATION_MODEL_NAME = "facebook/bart-large-cnn"
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

# --- INITIALIZATION ---

print("Initializing Models...")
try:
    translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME)
except Exception as e:
    print(f"Error loading translation model: {e}")
    print("Please check model name and internet connection.")
    exit()

summary_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME)
summary_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_NAME)
sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME)
print("Initialization Complete.")


# --- CORE FUNCTIONS ---

def detect_language(text):
    """Detects the language of the input text."""
    try:
        lang_code = detect(text)
        print(f"-> Language Detected: {lang_code}")
        return lang_code
    except:
        return "undetermined"

def translate_text(text, source_lang='hi', target_lang='en'):
    """Translates text from source_lang to target_lang."""
    print(f"-> Translating from {source_lang} to {target_lang}...")
    try:
        input_ids = translation_tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
        output_ids = translation_model.generate(input_ids, max_length=512, num_beams=4)
        translated_text = translation_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        print(f"Translation Error: {e}")
        return text 

def summarize_text(text):
    """Generates a concise summary of the (English) text."""
    print("-> Generating Summary...")
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

def analyze_sentiment(text):
    """Analyzes the sentiment of the (English) text."""
    print("-> Analyzing Sentiment...")
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

# --- MAIN WORKFLOW ---

def analyze_policy(policy_text):
    """Executes the full policy analysis workflow."""
    
    source_lang = detect_language(policy_text)
    
    if source_lang != 'en':
        translated_text = translate_text(policy_text, source_lang)
    else:
        print("-> Document is already in English, skipping translation.")
        translated_text = policy_text
        
    summary = summarize_text(translated_text)
    
    sentiment_label, sentiment_score = analyze_sentiment(summary)
    
    # 5. Output Report
    print("\n" + "="*50)
    print("      *** POLICY ANALYSIS REPORT ***")
    print("="*50)
    print(f"Source Language: {source_lang.upper()}")
    print("\n--- Translated Text (First 500 chars) ---")
    print(translated_text[:500] + "...")
    print("\n--- Summary ---")
    print(summary)
    print("\n--- Sentiment Analysis ---")
    print(f"Tone: {sentiment_label} (Confidence: {sentiment_score:.4f})")
    print("="*50)
    
    return {
        "source_lang": source_lang,
        "translated_text": translated_text,
        "summary": summary,
        "sentiment": {"label": sentiment_label, "score": sentiment_score}
    }


if __name__ == "__main__":
    
    HINDI_POLICY = (
        "प्रधानमंत्री आवास योजना का उद्देश्य हर नागरिक को घर प्रदान करना है। "
        "सरकार ने इस योजना के तहत 2022 तक एक करोड़ से अधिक घरों के निर्माण का लक्ष्य रखा है। "
        "यह योजना गरीबों और मध्यम वर्ग के लिए बहुत फायदेमंद साबित हुई है, जिससे सामाजिक असमानता कम हुई है।"
    )
    
    analyze_policy(HINDI_POLICY)