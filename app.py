import streamlit as st
from langdetect import detect
from googletrans import Translator, LANGUAGES
from transformers import pipeline
from io import StringIO
import docx2txt
import PyPDF2

# Initialize Streamlit app
st.set_page_config(page_title="Language-Aware AI Tutor", layout="centered")

# Load Hugging Face model
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-large")

qa_pipeline = load_model()
translator = Translator()

st.title("🌍 Language-Aware AI Tutor")

# Sort language names alphabetically
language_options = sorted([(name.title(), code) for code, name in LANGUAGES.items()], key=lambda x: x[0])
language_names = [name for name, code in language_options]
language_name_to_code = {name: code for name, code in language_options}

# Student Info Section
grade = st.selectbox("Grade", list(range(1, 13)))
language_choice = st.selectbox("Preferred Language", language_names)
selected_lang_code = language_name_to_code[language_choice]

st.markdown("---")

# File Upload Section
st.header("📄 Or Upload a File")
uploaded_file = st.file_uploader("Upload a .txt, .docx, or .pdf file", type=["txt", "docx", "pdf"])

def extract_text(file):
    file_type = file.name.split('.')[-1].lower()
    if file_type == "txt":
        return StringIO(file.getvalue().decode("utf-8")).read()
    elif file_type == "docx":
        return docx2txt.process(file)
    elif file_type == "pdf":
        reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    return ""

file_text = ""
if uploaded_file:
    try:
        file_text = extract_text(uploaded_file)
        st.text_area("📘 Extracted Text from File:", value=file_text, height=200)
    except:
        st.error("❌ Could not read the uploaded file.")

st.markdown("---")
st.header("🧠 Ask a Question")

if file_text:
    st.markdown("**📝 Ask something about your uploaded file (or type a new question):**")
else:
    st.markdown("**💬 Type your academic question below:**")

user_question = st.text_area("Question:")

if st.button("Get Answer"):
    if user_question:
        # Detect input language
        try:
            detected_lang = detect(user_question)
        except:
            st.error("Could not detect language. Please try rephrasing.")
            detected_lang = 'en'

        st.write(f"🗣 Detected Language: `{detected_lang}`")

        # Translate question to English
        if detected_lang != 'en':
            try:
                translated_question = translator.translate(user_question, src=detected_lang, dest='en').text
            except:
                st.error("Translation failed. Please check your internet connection or try a simpler question.")
                translated_question = user_question
        else:
            translated_question = user_question

        # Build prompt
        if file_text:
            instruction = f"You are an educational assistant. Given the document content below, answer the question as if teaching a grade {grade} student:\n\nDocument:\n{file_text}\n\nQuestion: {translated_question}"
        else:
            instruction = f"Answer this question as if teaching a grade {grade} student:\n{translated_question}"

        # Generate answer
        with st.spinner("Thinking..."):
            try:
                result = qa_pipeline(instruction, max_length=512, do_sample=True, temperature=0.7)[0]['generated_text']
            except Exception as e:
                st.error(f"Model failed: {e}")
                result = "Sorry, I couldn't generate an answer."

        # Translate answer back to preferred language
        if selected_lang_code != 'en':
            try:
                translated_answer = translator.translate(result, dest=selected_lang_code).text
            except:
                translated_answer = result
        else:
            translated_answer = result

        # Show both English and Translated answers
        st.success("✅ Answer:")
        st.markdown("**🔤 English:**")
        st.write(result)
        if selected_lang_code != 'en':
            st.markdown(f"**🌐 {language_choice} Translation:**")
            st.write(translated_answer)
    else:
        st.warning("Please enter a question.")
