import streamlit as st
import requests
import speech_recognition as sr
import tempfile
import pytesseract
from PIL import Image

File_Upload_Endpoint="http://127.0.0.1:8000/upload-pdf/"
User_Query_Endpoint="http://127.0.0.1:8000/ask/"
Quick_Response_Endpoint="http://127.0.0.1:8000/quick/"

recognizer = sr.Recognizer()

st.set_page_config(
    page_title="Knowledge AI",
    page_icon="🧠",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_path" not in st.session_state:
    st.session_state.file_path=""

if "quick_btn" not in st.session_state:
    st.session_state.quick_btn=""

st.markdown("""
    <style>

        .user {
            background: linear-gradient(135deg, #667eea, #764ba2);
            padding: 14px 18px;
            border-radius: 16px;
            max-width: 70%;
            margin-left: auto;
            margin-bottom: 12px;
            font-weight: 500;
        }

        .ai {
            background: rgba(255,255,255,0.12);
            padding: 16px 18px;
            border-radius: 16px;
            max-width: 100%;
            margin-bottom: 12px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        }

        .avatar {
            font-size: 22px;
            margin-right: 10px;
        }

        .quick > button {
            border-radius: 999px;
            padding: 10px 18px;
            background: rgba(255,255,255,0.12);
            color: white;
            border: 1px solid rgba(255,255,255,0.25);
            font-weight: 500;
        }

        .quick > button:hover {
            background: linear-gradient(135deg, #667eea, #764ba2);
            border: none;
        }

        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<h2 style='color:#03e8fc;'>Unified Knowledge Intelligence</h2>"
    "<p style='color:#017580;'>RAG-powered assistant</p>",
    unsafe_allow_html=True
)
st.markdown("---")

with st.sidebar:
    st.markdown("<h3 style='color:#16e7fa'>Turn documents into decisions.</h3>",unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("⬆ Upload to Server"):
            with st.spinner("Uploading file..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }
                response = requests.post(
                    File_Upload_Endpoint,
                    files=files,
                    timeout=1000
                )
            if response.status_code == 200:
                data = response.json()
                st.session_state.file_path=data["file_path"]
                st.success("File uploaded successfully!")
            else:
                st.error("File upload failed")

    st.divider()
    st.caption("Power by LLM+Branched & Multimodel RAG")

with st.container():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            if msg['type']=="text":
                st.markdown(
                    f"<div class='user'>🧑‍💻 {msg['content']}</div>",
                    unsafe_allow_html=True
                )
            elif msg['type']=="image":
                st.image(msg["content"])
        else:
            st.markdown(
                f"<div class='ai'>🤖 {msg['content']}</div>",
                unsafe_allow_html=True
            )

prompt = st.chat_input("Ask anything from your document...",accept_audio=True,accept_file=True, file_type=["jpg", "jpeg", "png"])

if prompt:

    if prompt.audio:
        audio_file = prompt.audio
        # save audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            audio_path = tmp.name
        # speech → text
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        try:
            prompt = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            st.error("Could not understand audio")
            prompt = None
        st.session_state.messages.append({
            "role": "user",
            "type": "text",
            "content": prompt
        })
    
    elif prompt.files:
        for file in prompt.files:
            if file.type.startswith("image"):
                st.session_state.messages.append({
                    "role": "user",
                    "type": "image",
                    "content": file
                })
                image = Image.open(file)
                prompt = pytesseract.image_to_string(image)

    elif prompt.text:
        st.session_state.messages.append({
            "role": "user",
            "type": "text",
            "content": prompt.text
        })
        prompt=prompt.text

    with st.spinner("🧠 Thinking..."):
        try:
            response = requests.post(
                url=User_Query_Endpoint,
                json={"query":prompt,"file_path":st.session_state.file_path},
                timeout=1000
            )
            
            if response.status_code == 200:
                ans = response.json()
                answer=ans['answer']
            else:
                answer = "❌ Error from backend."

        except Exception:
            answer = "❌ FastAPI server not reachable."
        
    st.session_state.messages.append(
        {"role": "assistant", "type":"text", "content": answer}
    ) 

    st.rerun()


st.markdown("### ⚡ Smart Actions")
c1, c2, c3, c4, c5, c6 = st.columns(6)

st.session_state.quick_btn="None"

with c1:
    if st.button("📌 Summary", key="q1"):
        uq="Summarize the document"
        st.session_state.messages.append(
            {"role": "user", "type":"text", "content": uq}
        )
        st.session_state.quick_btn="summary"

with c2:
    if st.button("📊 Tables", key="q2"):
        uq="Extract key tables"
        st.session_state.messages.append(
            {"role": "user", "type":"text", "content": uq}
        )
        st.session_state.quick_btn="tables"

with c3:
    if st.button("🔍 Insights", key="q3"):
        uq="Find important insights"
        st.session_state.messages.append(
            {"role": "user", "type":"text", "content": uq}
        )
        st.session_state.quick_btn="insights"

with c4:
    if st.button("🧠 Simple", key="q4"):
        uq="Explain in simple terms"
        st.session_state.messages.append(
            {"role": "user", "type":"text", "content": uq}
        )
        st.session_state.quick_btn="simple"

with c5:
    if st.button("📑 Notes", key="q5"):
        uq="Generate structured notes"
        st.session_state.messages.append(
            {"role": "user", "type":"text", "content": uq}
        )
        st.session_state.quick_btn="notes"

if st.session_state.quick_btn!="None":
    with st.spinner("🧠 Thinking..."):
        try:
            response = requests.post(
                url=Quick_Response_Endpoint,
                json={"button_pressed":st.session_state.quick_btn,"file_path":st.session_state.file_path},
                timeout=1000
            )
            
            if response.status_code == 200:
                ans = response.json()
                answer=ans['answer']
            else:
                answer = "❌ Error from backend."

        except Exception:
            answer = "❌ FastAPI server not reachable."
            
        st.session_state.messages.append(
            {"role": "assistant", "type":"text", "content": answer}
        ) 

        st.rerun()
