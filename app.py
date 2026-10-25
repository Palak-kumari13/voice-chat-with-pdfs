"""Free-tier voice PDF question answering demo (Streamlit + Groq + FAISS)."""
import os
import re
import base64
from io import BytesIO

import faiss
import fitz
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title='Voice Chat with PDFs', page_icon='📚', layout='wide')
st.title('🎙️ Voice Chat with PDFs')
st.caption('Upload documents, ask a question in text or using your microphone, and listen to a cited answer.')

@st.cache_resource(show_spinner='Loading local embedding model…')
def embedder():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')


def chunks_from_pdf(raw: bytes, filename: str):
    documents = []
    with fitz.open(stream=raw, filetype='pdf') as pdf:
        for pageno, page in enumerate(pdf, start=1):
            content = re.sub(r'\s+', ' ', page.get_text()).strip()
            if len(content) < 30:
                continue
            words = content.split()
            window, overlap = 180, 35
            for start in range(0, len(words), window - overlap):
                chunk = ' '.join(words[start:start+window])
                if len(chunk) > 40:
                    documents.append({'file': filename, 'page': pageno, 'text': chunk})
    return documents


def build_index(docs):
    texts = [d['text'] for d in docs]
    vectors = embedder().encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype('float32')
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve(question, docs, index, k=4):
    vector = embedder().encode([question], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    _, indices = index.search(vector, min(k, len(docs)))
    return [docs[int(i)] for i in indices[0] if i >= 0]


def get_key():
    try:
        return st.secrets.get('GROQ_API_KEY', '') or os.environ.get('GROQ_API_KEY', '')
    except FileNotFoundError:
        return os.environ.get('GROQ_API_KEY', '')


with st.sidebar:
    st.header('Document library')
    files = st.file_uploader('Upload text-based PDF files', type=['pdf'], accept_multiple_files=True)
    st.caption('Scanned/image-only PDFs need OCR, which this demo does not include.')
    st.warning('For a public demo, upload only PDFs you are comfortable processing in the session.')
    if st.button('Clear session', use_container_width=True):
        for key in ('doc_signature', 'docs', 'index', 'history', 'audio_signature', 'transcript'):
            st.session_state.pop(key, None)
        st.rerun()

api_key = get_key()
if not api_key:
    st.error('Missing GROQ_API_KEY. Add it to Streamlit Cloud → App settings → Secrets (or set an environment variable locally).')
    st.stop()

if 'history' not in st.session_state:
    st.session_state.history = []

signature = tuple((f.name, f.size, f.file_id) for f in files) if files else ()
if signature != st.session_state.get('doc_signature'):
    st.session_state.doc_signature = signature
    st.session_state.pop('docs', None)
    st.session_state.pop('index', None)
    st.session_state.history = []

if files and 'index' not in st.session_state:
    with st.spinner('Reading PDFs and building FAISS index…'):
        try:
            documents = []
            for f in files:
                if f.size > 15 * 1024 * 1024:
                    st.warning(f'Skipped {f.name}: larger than 15 MB.')
                    continue
                documents.extend(chunks_from_pdf(f.getvalue(), f.name))
            if documents:
                st.session_state.docs = documents
                st.session_state.index = build_index(documents)
                st.success(f'Indexed {len(documents)} passages from {len(files)} uploaded PDF(s).')
            else:
                st.warning('No extractable text found. Try a text-based PDF.')
        except Exception as exc:
            st.error(f'Could not process PDF: {exc}')

for item in st.session_state.history:
    with st.chat_message(item['role']):
        st.markdown(item['content'])

st.subheader('Ask a question')
audio = st.audio_input('Record a question')
transcript = ''
if audio is not None:
    audio_sig = (audio.file_id, len(audio.getvalue()))
    if audio_sig != st.session_state.get('audio_signature'):
        try:
            with st.spinner('Transcribing audio with Whisper…'):
                audio_bytes = BytesIO(audio.getvalue())
                audio_bytes.name = 'question.wav'
                client = Groq(api_key=api_key)
                transcription = client.audio.transcriptions.create(file=('question.wav', audio_bytes.read()), model='whisper-large-v3-turbo')
                st.session_state.transcript = transcription.text
                st.session_state.audio_signature = audio_sig
        except Exception as exc:
            st.error(f'Audio transcription failed: {exc}')
    transcript = st.session_state.get('transcript', '')
    if transcript:
        st.info(f'🎤 Recognized: {transcript}')

text_question = st.chat_input('Or type your question about the PDFs…')
ask_voice = st.button('Ask recorded question', disabled=not bool(transcript))
question = transcript if ask_voice else text_question

if question:
    if 'index' not in st.session_state:
        st.warning('Please upload a text-based PDF first.')
    else:
        with st.chat_message('user'):
            st.write(question)
        st.session_state.history.append({'role': 'user', 'content': question})
        passages = retrieve(question, st.session_state.docs, st.session_state.index)
        context = '\n\n'.join(f'[Source {i}: {doc["file"]}, page {doc["page"]}]\n{doc["text"]}' for i, doc in enumerate(passages, start=1))
        system = ('You answer questions using ONLY the supplied PDF excerpts. If the evidence is insufficient, say so. '
                  'Cite supporting material inline using [Source N], and do not invent references. Keep answers concise.')
        try:
            with st.spinner('Generating an answer…'):
                response = Groq(api_key=api_key).chat.completions.create(
                    model='openai/gpt-oss-20b',
                    messages=[{'role': 'system', 'content': system},
                              {'role': 'user', 'content': f'PDF excerpts:\n{context}\n\nQuestion: {question}'}],
                    temperature=0.1,
                    max_tokens=700,
                )
                answer = response.choices[0].message.content or 'No answer received.'
            st.session_state.history.append({'role': 'assistant', 'content': answer})
            with st.chat_message('assistant'):
                st.markdown(answer)
                with st.expander('PDF evidence'):
                    for i, doc in enumerate(passages, 1):
                        st.markdown(f'**Source {i}: {doc["file"]}, page {doc["page"]}**')
                        st.write(doc['text'])
                st.caption('🔊 Click Play below to hear the answer. Playback uses your browser’s built-in speech synthesis.')
                # Pass model output into browser JS using base64, not inline HTML attributes.
                encoded = base64.b64encode(answer.encode('utf-8')).decode('ascii')
                components.html(f'''<button id="play" style="border:1px solid #999;border-radius:9px;padding:9px 18px;cursor:pointer;">▶ Play answer</button> <button id="stop" style="border:1px solid #999;border-radius:9px;padding:9px 18px;cursor:pointer;">■ Stop</button>
<script>
const data = "{encoded}";
const bytes = Uint8Array.from(atob(data), c => c.charCodeAt(0));
const content = new TextDecoder().decode(bytes);
document.getElementById('play').addEventListener('click', () => {{
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(content));
}});
document.getElementById('stop').addEventListener('click', () => window.speechSynthesis.cancel());
</script>''', height=52)

        except Exception as exc:
            st.error(f'Answer generation failed: {exc}')

st.caption('Free-tier demo: API rate limits apply. Uploaded files and indexes are session-scoped and may be lost when the app restarts.')
