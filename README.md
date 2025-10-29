# 🧠 Abstractive Text Summarizer (BART)

This project demonstrates how to fine-tune a **BART (Bidirectional and Auto-Regressive Transformers)** model for **abstractive text summarization** and deploy it as an interactive web application using **Streamlit**.

---

## 📘 Overview

The project is divided into two main parts:

1. **Text_Summarization.ipynb**  
   A Jupyter Notebook that details the complete process of loading the [`knkarthick/dialogsum`](https://huggingface.co/datasets/knkarthick/dialogsum) dataset, fine-tuning a `facebook/bart-base` model, and saving the resulting model files.

2. **app.py**  
   A Streamlit web application that loads the fine-tuned model to provide a user-friendly interface for summarizing text, either from direct input or a scraped URL.

---

## 🚀 App Features

The **Streamlit application (`app.py`)** provides a simple and efficient UI with several features:

- 📝 **Dual Input Modes:** Choose between _Direct Text Input_ or _Blog Post URL_ for summarization.  
- 🌐 **Web Scraping:** Automatically scrapes all paragraph text from a given URL using `requests` and `BeautifulSoup`.  
- ⚙️ **Configurable Summary Length:** Set minimum and maximum token length for the generated summary.  
- ⚡ **Cached Model Loading:** Uses `@st.cache_resource` to load the model once for faster performance.  

---

## 🛠️ Technology Stack

| Component | Description |
|------------|-------------|
| **Python** | Core programming language |
| **Streamlit** | Interactive web app framework |
| **Hugging Face Transformers** | For model (`AutoModelForSeq2SeqLM`), tokenizer (`AutoTokenizer`), and `Trainer` |
| **Hugging Face Datasets** | For loading and processing the `knkarthick/dialogsum` dataset |
| **PyTorch (torch)** | Backend for model training and inference |
| **requests** & **beautifulsoup4** | For web scraping blog content |
| **wandb** | For logging and monitoring training metrics |
| **Jupyter Notebook** | For training and experimentation |

---

## ⚙️ Setup & Installation

Follow these steps to get the project running on your local machine.

### 1️⃣ Clone the Repository

```bash
git clone https://your-repository-url.git
cd your-project-directory
```
### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```
### 3️⃣ Install Dependencies

Install all required packages using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```
## 🧩 Usage

There are two main parts to this project: **training the model** and **running the application**.

---

### 🧠 1. Training the Model (Optional)

If you want to train or fine-tune the model yourself:

#### Start Jupyter Notebook
```bash
jupyter notebook
```

#### Run the Notebook
Open and execute all cells in `Text_Summarization.ipynb`.  
This will:

- Load the **knkarthick/dialogsum** dataset
- Tokenize and preprocess the data
- Fine-tune the **facebook/bart-base** model
- Save the final model and tokenizer to a directory (e.g., `/content/your_model_directory`)

---

### 💻 2. Running the Streamlit Application

The Streamlit app requires a **trained model** to be present.

#### Step 1: Place Model Files

- Create a folder named `model_files` in the root of your project.
- Copy the saved model files (`pytorch_model.bin`, `config.json`, `tokenizer.json`, etc.) into this folder.

#### Step 2: Run the App
```bash
streamlit run app.py
```

#### Step 3: Interact with the App

- Open your browser to the local URL (usually [http://localhost:8501](http://localhost:8501)).
- Choose your input type:
  - 🧾 **Direct Text Input**, or
  - 🌍 **Blog Post URL**
- Adjust the summary length sliders.
- Click **“Summarize ✨”** to generate your summary.

---

## 📂 Project Structure
```
.
├── model_files/                <-- Contains fine-tuned model files
├── app.py                      <-- Streamlit application
├── Text_Summarization.ipynb    <-- Jupyter Notebook for training
├── requirements.txt            <-- Python dependencies
└── README.md                   <-- You are here
```

---

## 🏁 Future Improvements

- Add model performance metrics on the Streamlit dashboard  
- Support for multilingual summarization  
- Integration with Hugging Face Hub for model hosting  

---

## 🧑‍💻 Author

**Parth Zadeshwariya**  

Feel free to ⭐ this repository if you find it useful!
