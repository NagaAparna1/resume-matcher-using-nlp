
import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="Resume Matcher", layout="centered")

st.title("🚀 Resume Matcher + Smart Chatbot")

menu = ["Resume Matcher", "Chatbot"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Resume Matcher":
    st.subheader("📄 Resume Matching System")

    resume = st.text_area("Paste Resume", height=150)
    job = st.text_area("Paste Job Description", height=150)

    if st.button("Match Resume"):
        if resume == "" or job == "":
            st.warning("Please enter both Resume and Job Description")
        else:
            res = requests.post(f"{API}/match", params={
                "resume": resume,
                "job": job
            }).json()

            score = res["similarity"]
            st.success(f"Similarity Score: {score:.2f}")

            if score < 0.5:
                st.error("⚠️ Low Match! Your resume needs improvement")
                st.subheader("💡 Suggestions to Improve")
                for s in res["suggestions"]:
                    st.write("👉", s)
            elif score < 0.75:
                st.warning("🟡 متوسط match! You can improve further")
                st.subheader("💡 Improve these areas")
                for s in res["suggestions"]:
                    st.write("👉", s)
            else:
                st.success("🎉 Great Match! Your resume fits this job")
                st.subheader("🚀 Final Tips")
                st.write("👉 Add more real-world projects")
                st.write("👉 Highlight achievements with numbers")
                st.write("👉 Prepare for interviews")

if choice == "Chatbot":
    st.subheader("🤖 Smart Career Chatbot")

    msg = st.text_input("Ask something about jobs, skills, resume")

    if st.button("Send"):
        if msg == "":
            st.warning("Please enter a question")
        else:
            res = requests.post(f"{API}/chat", params={"query": msg}).json()
            st.write("💬", res["response"])
