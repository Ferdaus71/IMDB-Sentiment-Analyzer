# 🎬 IMDB Sentiment Analyzer 

A **deep learning-based IMDB movie review sentiment analysis system** built using **TensorFlow**, **Keras**, and **Gradio**.  
It supports **real-time text predictions** and **batch analysis via .txt or .pdf uploads (including scanned PDFs using OCR)**.  
The application also features a beautiful **Dark/Light Mode Toggle** and a clean, interactive UI. 🌙☀️

---

## 🚀 Features

✅ **Bidirectional LSTM** trained on IMDB Dataset  
✅ **Gradio-powered UI** for interactive predictions  
✅ **Text and PDF Support** (auto OCR for scanned documents)  
✅ **Dark/Light Mode Toggle** with smooth animations  
✅ **Confidence Visualization** and adjustable threshold  
✅ Developed by **Md. Ferdaus Hossen**, Junior AI/ML Engineer @ Zensoft Lab

---

## 🧠 Model Overview

The model uses a **2-layer Bidirectional LSTM** architecture for binary sentiment classification (Positive / Negative).

**Architecture:**
<img width="1269" height="2589" alt="image" src="https://github.com/user-attachments/assets/533d7f17-d463-48c4-9b13-9e6596f5101e" />



---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| Framework | TensorFlow / Keras |
| Frontend UI | Gradio |
| OCR Engine | PyPDF2, pdf2image, pytesseract |
| Data | IMDB Sentiment Dataset |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colab / Local Python 3.x |

---

## ⚙️ Installation & Setup

### 🔧 1. Clone this repository
```bash
git clone https://github.com/Ferdaus71/IMDB-Sentiment-Analyzer.git
cd IMDB-Sentiment-Analyzer


2. Install dependencies

💡 Run this inside your terminal or Colab cell.

pip install -r requirements.txt
sudo apt-get install -y poppler-utils

🧠 Run Options
▶️ Option 1 — Google Colab (Recommended)

Open IMDB_Sentiment_Analyzer.ipynb in Google Colab
.

Run all cells step-by-step (Cells 0–14).

At the final cell, click the Gradio app link to launch the interface.

▶️ Option 2 — Local Run

If you want to run the Gradio UI directly:
python app.py
Then open the local URL or public link (provided by Gradio).


💡 How to Use
🗣️ Single Review Mode

Type or paste a movie review.

Adjust the “Positive Sentiment Threshold” slider (default = 0.5).

Click 🔍 Analyze Sentiment.

View:

Predicted Label (🌟 Positive / 💔 Negative)

Confidence Score

Probability Visualization

📂 Batch Mode (File Upload)

Upload .txt or .pdf file (supports scanned PDFs with OCR).

Each paragraph/line will be treated as one review.

Adjust the threshold slider.

Get a results table with predictions for all reviews.

📁 Folder Structure

IMDB-Sentiment-Analyzer-with-OCR-UI/
│
├── IMDB_Sentiment_Analyzer.ipynb     # Full Colab-ready notebook
├── app.py                            # Optional standalone script
├── README.md                         # Documentation (this file)
├── requirements.txt                   # Project dependencies
│
├── screenshots/                      # App preview images
│   ├── light_ui.png
│   
│
└── model/                            # Pretrained model
    └── bidirectional_imdb_model.h5

🧾 Requirements
tensorflow
keras
gradio
PyPDF2
pdf2image
pytesseract
matplotlib
seaborn
scikit-learn
pandas

👨‍💻 Developer

🧑‍💻 Md. Ferdaus Hossen
Junior AI/ML Engineer @ Zensoft Lab


