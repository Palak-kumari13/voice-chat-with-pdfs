# Voice Chat with PDFs — Free GitHub Deployment

A public voice-enabled PDF Q&A demo using Groq Whisper for transcription, Groq GPT-OSS for grounded answers, local Sentence Transformers embeddings, FAISS for similarity retrieval, Streamlit for the UI, and the browser's SpeechSynthesis for spoken playback. No Gemini key required. No LangChain dependency in this lightweight hosted edition: the RAG pipeline is implemented explicitly to minimize dependencies.

## Run locally

1. Install Python 3.11 or 3.12.
2. `pip install -r requirements.txt`
3. Set `GROQ_API_KEY` as an environment variable. Get a free tier key at https://console.groq.com/keys . Windows PowerShell: `$env:GROQ_API_KEY="your_key"`; macOS/Linux: `export GROQ_API_KEY="your_key"`.
4. `streamlit run app.py`

## Publish the site for free

1. Make a public GitHub repo, e.g. `voice-chat-with-pdfs`, and upload the **contents** of this folder (not the zip).
2. Go to https://share.streamlit.io/ and connect GitHub.
3. Create app, choose the repo and `app.py`, select Python 3.11/3.12 if prompted.
4. Under **Advanced settings → Secrets**, insert `GROQ_API_KEY = "your_actual_key"` (never put the key into GitHub).
5. Deploy. Share the resulting `*.streamlit.app` link.

Note: The web frontend is hosted by Streamlit Cloud, while GitHub holds your project source and triggers redeployments. GitHub Pages alone cannot execute Python backend code.

## Limitations and safety

- Groq has free-tier request limits, which may be reached if the site becomes popular. Check https://console.groq.com/docs/rate-limits . This is not an unlimited free API.
- This is a prototype for low-traffic public demonstration, not a production security-hardened service. A single stored API key serves all visitors; consider authenticated access, upload limits, quotas, and abuse protection before broad promotion.
- PDF data is only kept in Streamlit session state while the session lives; passages sent to Groq for answer generation and audio recordings sent for transcription are processed by Groq. Do not upload sensitive documents.
- Scanned image PDFs need OCR and are not supported in this starter.
- The small local embedding model is downloaded at first startup; cold start can take some time, and free Streamlit instances have memory/compute limits.
- Free browser TTS voice quality varies by browser and OS and is not Gemini TTS.
- This deployable edition intentionally replaces LangChain with a direct, inspectable FAISS RAG pipeline; LangChain can be added after the deployment works.
