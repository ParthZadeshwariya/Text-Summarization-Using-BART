import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# --- CONFIGURATION ---
st.set_page_config(page_title="Text Summarizer", layout="wide")

# This is the path to your folder containing model files
MODEL_DIRECTORY = "./model_files" 

# --- MODEL LOADING ---

# Cache the model loading to prevent reloading on every interaction
@st.cache_resource
def load_model():
    """Loads the fine-tuned model and tokenizer from the local directory."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIRECTORY)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIRECTORY)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure your model files are in a folder named 'model_files' in the same directory as app.py.")
        return None, None

# --- WEB SCRAPING FUNCTION ---

def scrape_text(url):
    """Scrapes text content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all paragraph tags
        paragraphs = soup.find_all('p')
        
        # Concatenate the text from all paragraphs
        article_text = ' '.join([p.get_text() for p in paragraphs])
        
        if not article_text:
            st.warning("Could not find any paragraph (<p>) text on this page. Trying to get all text.")
            # Fallback: get all text, strip whitespace
            article_text = ' '.join(soup.stripped_strings)

        return article_text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        return None
    except Exception as e:
        st.error(f"Error scraping text: {e}")
        return None

# --- SUMMARIZATION FUNCTION ---

def summarize(tokenizer, model, text, min_len, max_len):
    """Generates a summary for the given text using the loaded model."""
    try:
        # Check for GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        # Prepare the text for the model
        inputs = tokenizer(text, 
                           max_length=1024,  # Truncate long inputs
                           return_tensors="pt", 
                           padding=True, 
                           truncation=True)
        
        input_ids = inputs.input_ids.to(device)
        
        # Generate the summary
        summary_ids = model.generate(input_ids, 
                                     num_beams=4, 
                                     min_length=min_len, 
                                     max_length=max_len, 
                                     early_stopping=True)
        
        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return None

# --- STREAMLIT UI ---

st.title("📄 Abstractive Text Summarizer")
st.markdown("Summarize text or blog posts using your fine-tuned model.")

# Load the model and tokenizer
tokenizer, model = load_model()

if model is not None:
    # --- Input Selection ---
    input_type = st.radio(
        "Choose your input type:",
        ('Direct Text Input', 'Blog Post URL'),
        horizontal=True
    )

    # --- UI Columns ---
    col1, col2 = st.columns([2, 1])

    with col1:
        if input_type == 'Direct Text Input':
            input_text = st.text_area("Enter your text here:", height=300, key="text_input")
        else:
            url_input = st.text_input("Enter the URL:", key="url_input")
            input_text = None # Will be filled by scraping

    with col2:
        st.subheader("Summary Configuration")
        min_len = st.slider("Minimum Summary Length:", min_value=10, max_value=200, value=30)
        max_len = st.slider("Maximum Summary Length:", min_value=50, max_value=500, value=150)
        
        if max_len < min_len:
            st.warning("Max length should be greater than min length.")
            summarize_button = False
        else:
            summarize_button = st.button("Summarize ✨", type="primary", use_container_width=True)

    st.divider()

    # --- Processing and Output ---
    if summarize_button:
        final_text_to_summarize = ""

        if input_type == 'Direct Text Input':
            if input_text:
                final_text_to_summarize = input_text
            else:
                st.error("Please enter some text to summarize.")
        
        else: # URL Input
            if url_input:
                with st.spinner("Scraping website... 🕸️"):
                    scraped_text = scrape_text(url_input)
                    if scraped_text:
                        final_text_to_summarize = scraped_text
                        # Show a snippet of the scraped text
                        st.text_area("Scraped Text (Snippet):", 
                                     scraped_text[:1000] + "...", 
                                     height=150, 
                                     disabled=True)
            else:
                st.error("Please enter a URL to scrape.")

        # --- Generate Summary ---
        if final_text_to_summarize:
            with st.spinner("Generating summary... 🧠"):
                summary = summarize(tokenizer, model, final_text_to_summarize, min_len, max_len)
                
                if summary:
                    st.subheader("Generated Summary")
                    st.success(summary)

else:
    st.error("Model could not be loaded. The application cannot start.")