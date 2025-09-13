import re
import PyPDF2
import docx2txt
import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Required skills
required_skills = ["python","sql","machine learning","deep learning","nlp","tableau","aws","azure"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text

def extract_text_from_pdf(file):
    text = ""
    with open(file.name, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file.name)

def analyze_resume(file, job_description):
    if file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        resume_text = extract_text_from_docx(file)
    else:
        return "❌ Unsupported file format!", ""

    # Clean text
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_description)

    # Embeddings
    resume_emb = model.encode([resume_clean])
    jd_emb = model.encode([jd_clean])
    score = cosine_similarity(resume_emb, jd_emb)[0][0]

    # Missing skills
    missing_skills = [s for s in required_skills if s not in resume_clean]

    result = f"✅ Match Score: {round(score*100,2)}%\n\n🚫 Missing Skills: {', '.join(missing_skills) if missing_skills else 'None 🎉'}"
    return result, resume_text[:800]  # show some extracted text

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🤖 AI-Powered Resume Macher")
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="📄 Upload Resume (PDF/DOCX)")
            jd_input = gr.Textbox(label="📝 Job Description", lines=6)
            btn = gr.Button("🚀 Analyze")
        with gr.Column():
            output = gr.Textbox(label="📊 Results", lines=8)
            preview = gr.Textbox(label="📑 Resume Extract (first 800 chars)", lines=10)

    btn.click(analyze_resume, inputs=[file_input, jd_input], outputs=[output, preview])

demo.launch()
