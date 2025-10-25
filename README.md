# INFO 5940 — RAG Chat Application

Welcome to your **INFO 5940** repository.  
This project implements a **Retrieval-Augmented Generation (RAG)** chat application using **Streamlit** and **LangChain**, running inside **GitHub Codespaces**.  
You can upload multiple `.txt` or `.pdf` documents and chat interactively with their content.

---

## 🧰 Getting Started

### Step 1: Fork this Repository
1. Click **Fork** (top-right corner of this page).
2. This creates a copy under **your GitHub account**, where you can safely edit and commit.

### Step 2: Open in GitHub Codespaces
1. Go to **your forked repository**.
2. Click the green **Code** button → **Codespaces** tab → **Create Codespace**.
3. Wait a few minutes for setup to finish.

### Step 3: Verify the Environment
Once the Codespace starts:
1. Select **Python 3.11.13** as the kernel (top-right → “Select Kernel”).
2. If prompted, install the **Python + Jupyter** extensions.
3. Run the setup or open `chat_with_pdf.py`.

---

## 🚀 Running the Streamlit App

### Step 1: Start the Application
Open the terminal inside your Codespace and run:
```bash
streamlit run chat_with_pdf.py
```

When prompted, click **“Open in Browser”** to launch the app.

If you miss the popup, stop it with `Ctrl + C` and rerun the command.

---

## 🔑 Setting Up Your API Key

This application uses the **Cornell OpenAI Gateway** (`https://api.ai.it.cornell.edu/`).

### Option 1 – Permanent (Recommended)
1. Open `.devcontainer/devcontainer.json`.
2. Add or update the `"remoteEnv"` section:
   ```json
   "remoteEnv": {
     "OPENAI_API_KEY": "your_api_key_here",
     "BASE_URL": "https://api.ai.it.cornell.edu/",
     "OPENAI_BASE_URL": "https://api.ai.it.cornell.edu/"
   }
   ```
3. Rebuild the container (`Ctrl+Shift+P → Rebuild Container`).

### Option 2 – Temporary (For Testing)
```bash
export OPENAI_API_KEY="your_api_key_here"
streamlit run chat_with_pdf.py
```

---

## 🧠 Model Configuration (Cornell OpenAI Gateway)

This project runs on Cornell’s internal OpenAI proxy.  
Hence, model names differ from the public OpenAI API.

| Purpose | Model Name Used | Equivalent OpenAI Model |
|----------|----------------|--------------------------|
| Embeddings | `openai.text-embedding-3-small` | `text-embedding-3-small` |
| Chat LLM | `openai.o3-mini` | `gpt-4o-mini` |

---

## 💬 App Features

- Upload multiple `.txt` and `.pdf` files.  
- Automatically extract and chunk text (default: 1000 chars, 150 overlap).  
- Store embeddings in a **Chroma vector database**.  
- Retrieve relevant chunks and generate answers using **ConversationalRetrievalChain**.  
- Multi-turn conversation memory (`chat_history`).  
- Displays **document sources with relevance scores** and short previews.

---

## 📦 Requirements

Core dependencies are already listed in `requirements.txt`.  
If you are setting up manually, ensure these packages are included:

```txt
streamlit>=1.36
langchain==0.3.27
langchain-community==0.3.31
langchain-openai==0.3.35
openai~=1.14
chromadb>=0.5.3
pdfplumber>=0.11.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## ⚙️ Running Notes

1. Each time you start a new Codespace, make sure your **API key** is set.  
2. Large PDFs may take longer to process.  
3. Temporary directories (`/tmp`) are used for Chroma persistence.  
   - If the index disappears after rerun, simply re-upload and rebuild.  

---

## 🧩 How It Works (High-Level Overview)

1. **File Upload → Extraction**  
   Extract text using `pdfplumber` or plain decoding.
2. **Chunking**  
   Split text into 1000-character chunks using `RecursiveCharacterTextSplitter`.
3. **Embedding**  
   Use `openai.text-embedding-3-small` via Cornell’s gateway.
4. **Storage**  
   Store vectors in Chroma.
5. **Retrieval + Generation**  
   A `ConversationalRetrievalChain` retrieves top-5 relevant chunks and generates responses.
6. **Chat Memory**  
   Maintains full conversation context across turns.

---

## 🧾 Environment Modifications

Added dependencies:
- `pdfplumber` — for PDF parsing  
- `chromadb` — for vector storage

Configured environment:
- `.devcontainer/devcontainer.json` updated with `"OPENAI_API_KEY"` and Cornell gateway URLs.

---

## 📚 Sources and References
- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [ChromaDB](https://docs.trychroma.com/)
- [Cornell AI Gateway](https://api.ai.it.cornell.edu/)
- [INFO 5940 Course Portal](https://canvas.cornell.edu/)
