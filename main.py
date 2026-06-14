

from fastapi import FastAPI
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# =========================
# LOAD MODELS
# =========================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# =========================
# CLEAN TEXT FUNCTION
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# =========================
#  SMART SKILL SUGGESTIONS
# =========================
skill_suggestions = {
    "python": "Learn Python and add at least 2 projects",
    "machine": "Study Machine Learning algorithms",
    "learning": "Practice ML models using sklearn",
    "nlp": "Build NLP projects like chatbot or sentiment analysis",
    "sql": "Learn SQL and database queries",
    "deep": "Understand Deep Learning (CNN, RNN)",
    "data": "Work on data analysis projects using pandas",
    "analysis": "Improve data analysis and visualization skills",
    "excel": "Learn Excel basics and data handling",
    "communication": "Improve communication and soft skills"
}

# =========================
#  RESUME MATCH + SUGGESTIONS
# =========================
@app.post("/match")
def match(resume: str, job: str):
    resume_clean = clean_text(resume)
    job_clean = clean_text(job)

    # similarity
    vec = vectorizer.transform([resume_clean, job_clean])
    score = cosine_similarity(vec[0], vec[1])[0][0]

    # extract words
    resume_words = set(resume_clean.split())
    job_words = set(job_clean.split())

    missing_words = job_words - resume_words

    # convert to smart suggestions
    suggestions = []
    for word in missing_words:
        if word in skill_suggestions:
            suggestions.append(skill_suggestions[word])

    # remove duplicates + limit
    suggestions = list(set(suggestions))[:5]

    # default message
    if not suggestions:
        suggestions = ["Your resume looks good! Try adding more real-world projects."]

    return {
        "similarity": float(score),
        "suggestions": suggestions
    }

# =========================
#  RESUME CATEGORY PREDICTION
# =========================
@app.post("/predict")
def predict(resume: str):
    resume = clean_text(resume)

    vec = vectorizer.transform([resume])
    pred = model.predict(vec)
    category = label_encoder.inverse_transform(pred)

    return {"category": category[0]}

# =========================
#  SMART CHATBOT
# =========================
questions = [
    "hello",
    "what skills needed for data science",
    "how to improve resume",
    "how to get job",
    "what projects should i do",
    "bye"
]

answers = [
    "Hi! I can help you improve your resume and match jobs.",
    "You need Python, Machine Learning, NLP, SQL, and Data Analysis.",
    "Add strong projects, skills, and achievements to your resume.",
    "Practice coding, build projects, and apply regularly.",
    "Build projects like Resume Matcher, Chatbot, and ML models.",
    "Good luck! Keep learning."
]

q_vec = vectorizer.transform(questions)

@app.post("/chat")
def chat(query: str):
    query = clean_text(query)

    user_vec = vectorizer.transform([query])
    sim = cosine_similarity(user_vec, q_vec)

    idx = sim.argmax()

    return {"response": answers[idx]}
