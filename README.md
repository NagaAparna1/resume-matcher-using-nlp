The project is deployed and can be accessed online via this link:
https://resumematcher-so2t89unte46ueisbsyygo.streamlit.app/




# <span style="color:#FF5733; font-size:40px;">🚀 Resume Matcher + Smart Chatbot</span>

<p style="color:#2E86C1; font-size:18px;">
A web application that helps users <strong>match resumes with job descriptions</strong> using NLP and interact with a <strong>smart chatbot</strong> for guidance.<br>
Built using <strong>Streamlit</strong>, <strong>Python</strong>, and <strong>Pickle-based NLP models</strong> for fast similarity scoring.
</p>

---

## <span style="color:#28B463; font-size:28px;">📌 Features</span>

### <span style="color:#8E44AD; font-size:24px;">📝 Resume Matching</span>

<ul style="color:#1C2833; font-size:16px;">
<li>Compare resumes with job descriptions</li>
<li>Compute similarity scores using Word Vectorizer</li>
<li>Rank candidates based on keyword matches</li>
</ul>

### <span style="color:#8E44AD; font-size:24px;">💬 Smart Chatbot</span>

<ul style="color:#1C2833; font-size:16px;">
<li>Chat with users in natural language</li>
<li>Answer questions related to resumes or jobs</li>
<li>Default fallback for unknown queries</li>
</ul>

### <span style="color:#8E44AD; font-size:24px;">🖥 Web App</span>

<ul style="color:#1C2833; font-size:16px;">
<li>Streamlit frontend for interactive UI</li>
<li>Simple text input areas for resumes and job descriptions</li>
<li>Displays similarity score and matched keywords</li>
</ul>

---

## <span style="color:#28B463; font-size:28px;">🏗️ Tech Stack</span>

**Backend**

<ul style="color:#1C2833; font-size:16px;">
<li>Python 3.x</li>
<li>Streamlit</li>
<li>Scikit-learn (Word Vectorizer)</li>
<li>Pickle (Model Serialization)</li>
</ul>

**Frontend**

<ul style="color:#1C2833; font-size:16px;">
<li>Streamlit (all-in-one frontend & backend)</li>
<li>Minimal UI with text areas and buttons</li>
</ul>

---

## <span style="color:#28B463; font-size:28px;">📁 Folder Structure</span>

<pre style="color:#1C2833; font-size:16px;">
Resume_Matcher/
│── app.py              # Streamlit app
│── main.py             # Backend logic (optional)
│── models/             # Pickled NLP models (.pkl)
│── data/               # Sample resumes & job descriptions
│── requirements.txt    # Python dependencies
└── README.md           # Project overview & instructions
</pre>
## <span style="color:#28B463; font-size:28px;">📁 SYSTEM ARCHITECTURE</span

<p align="center">
  <img src="https://raw.githubusercontent.com/ambikaramireddy/RESUME_MATCHER/main/NLP%20system%20architecture%20flowchart.png" alt="NLP System Architecture" width="700"/>
</p>
## <span style="color:#28B463; font-size:28px;">🔧 Installation & Setup</span>


<li>Create virtual environment & install dependencies:
<pre>python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
pip install -r requirements.txt</pre></li>

<li>Run the app:
<pre>uvicorn main:app --reload</pre></li>

<li>New PowershellRun the app:
<pre>streamlit run app.py</pre></li>
</ol>

---


