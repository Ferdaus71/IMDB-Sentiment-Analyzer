# app.py
# 🎬 IMDB Sentiment Analyzer with OCR & Dark/Light UI
import gradio as gr
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract, tempfile, os, pandas as pd

# --- Load model ---
model = tf.keras.models.load_model("model/bidirectional_imdb_model.h5")

# --- Load IMDB word index ---
word_index = imdb.get_word_index()
index_from = 3
word_index = {k: (v + index_from) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

def encode_review(text, maxlen=200):
    tokens = text_to_word_sequence(text)
    encoded = [1]
    for word in tokens:
        encoded.append(word_index.get(word, 2))
    return pad_sequences([encoded], maxlen=maxlen)

# --- Single Review ---
def analyze_single_review(review_text, threshold=0.5):
    if not isinstance(review_text, str) or not review_text.strip():
        return "⚠️ Please enter a review.", 0.0
    seq = encode_review(review_text)
    prob = float(model.predict(seq)[0][0])
    sentiment = "🌟 Positive" if prob >= threshold else "💔 Negative"
    confidence = prob if prob >= threshold else 1 - prob
    label = f"{sentiment}  (probability: {prob:.3f})"
    return label, confidence

# --- Batch Analysis (PDF/TXT + OCR) ---
def analyze_file(file, threshold=0.5):
    if file is None:
        return pd.DataFrame([], columns=["Review", "Probability", "Sentiment"])
    filename = file.name.lower()
    text_data = ""
    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(file)
            pdf_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pdf_text += text + "\n"
            if not pdf_text.strip():
                with tempfile.TemporaryDirectory() as path:
                    temp_pdf = os.path.join(path, "temp.pdf")
                    with open(temp_pdf, "wb") as f:
                        f.write(file.read())
                    images = convert_from_path(temp_pdf)
                    pdf_text = "\n".join([pytesseract.image_to_string(img) for img in images])
            text_data = pdf_text
        else:
            text_data = file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return pd.DataFrame([{"Review": f"Error reading file: {e}", "Probability": None, "Sentiment": "Error"}])

    lines = [ln.strip() for ln in text_data.splitlines() if ln.strip()]
    results = []
    for review in lines:
        seq = encode_review(review)
        prob = float(model.predict(seq)[0][0])
        sentiment = "Positive" if prob >= threshold else "Negative"
        results.append({
            "Review": review[:150] + "..." if len(review) > 150 else review,
            "Probability": round(prob, 3),
            "Sentiment": sentiment
        })
    return pd.DataFrame(results)

# --- Gradio Interface ---
with gr.Blocks(title="🎬 IMDB Sentiment Analyzer (Enhanced + OCR)") as demo:
    gr.HTML("""
    <div style='text-align:center;padding:18px;background:linear-gradient(135deg,#2c5364,#203a43,#0f2027);color:white;border-radius:10px;'>
        <h1>🎬 IMDB Sentiment Analyzer (Enhanced + OCR)</h1>
        <p>Upload a <b>.txt</b> or <b>.pdf</b> (text or scanned) or type a review for instant prediction.</p>
        <button id='theme-btn' style='background:#4fc3f7;border:none;color:#000;padding:8px 16px;border-radius:6px;font-weight:600;cursor:pointer;'>🌙 Dark Mode</button>
    </div>
    <script>
    const btn=document.getElementById('theme-btn');
    btn.onclick=()=>{
      const dark=document.body.classList.toggle('dark-mode');
      btn.innerText=dark?'☀️ Light Mode':'🌙 Dark Mode';
      document.body.style.transition='all 0.4s';
      document.body.style.background=dark?'#121212':'#f7f7f7';
      document.body.style.color=dark?'#eee':'#111';
    };
    </script>
    """)

    with gr.Tab("🗣️ Single Review"):
        review_input = gr.Textbox(lines=5, placeholder="Type your movie review here...")
        threshold_slider = gr.Slider(0.1, 0.9, value=0.5, label="Threshold")
        analyze_button = gr.Button("🔍 Analyze Sentiment", variant="primary")
        output_label = gr.Textbox(label="Prediction")
        confidence_bar = gr.Slider(0, 1, value=0, label="Confidence", interactive=False)
        analyze_button.click(analyze_single_review, [review_input, threshold_slider], [output_label, confidence_bar])

    with gr.Tab("📂 File Upload"):
        file_input = gr.File(label="Upload .txt or .pdf")
        threshold_batch = gr.Slider(0.1, 0.9, value=0.5, label="Batch Threshold")
        analyze_file_button = gr.Button("📊 Analyze File")
        file_output = gr.DataFrame(headers=["Review", "Probability", "Sentiment"], label="Results")
        analyze_file_button.click(analyze_file, [file_input, threshold_batch], file_output)

    gr.HTML("""
    <div style='margin-top:30px;background:linear-gradient(90deg,#203a43,#2c5364);color:white;text-align:center;padding:15px;border-radius:8px;'>
        <h4>🧑‍💻 Developed by <span style='color:#4fc3f7;'>Md. Ferdaus Hossen</span></h4>
        <p>Junior AI/ML Engineer at <b>Zensoft Lab</b></p>
        <p>
            <a href='https://github.com/Ferdaus71' target='_blank' style='color:#fff;margin-right:10px;'>GitHub</a> |
            <a href='https://www.linkedin.com/in/ferdaus70/' target='_blank' style='color:#fff;margin-left:10px;'>LinkedIn</a>
        </p>
    </div>
    """)

demo.launch(share=True)
