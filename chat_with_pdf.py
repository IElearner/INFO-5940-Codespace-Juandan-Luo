import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import tempfile


st.title("RAG Chat with Your Documents")

uploaded_files = st.file_uploader(
    "Upload one or more documents (.txt or .pdf)",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

# Session states
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "documents" not in st.session_state:
    st.session_state["documents"] = {}
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = None

# Step 1. Read documents
if uploaded_files:
    for file in uploaded_files:
        name = file.name
        if name.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")
        elif name.endswith(".pdf"):
            text = ""
            try:
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                st.error(f"Failed to read {name}: {e}")
                continue
        else:
            st.warning(f"Unsupported file type: {name}")
            continue
        st.session_state["documents"][name] = text

# Step 2. Split documents
if st.session_state["documents"]:
    st.subheader("Splitting Documents into Chunks")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ".", " ", ""]
    )
    st.session_state["chunks"].clear()
    for name, text in st.session_state["documents"].items():
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            st.session_state["chunks"].append({
                "file": name,
                "chunk_id": i,
                "content": chunk
            })
        st.markdown(f"**{name}** — split into {len(chunks)} chunks")

# Step 3. Create Chroma vectorstore
if st.session_state["chunks"] and not st.session_state["vectorstore"]:
    st.subheader("Creating Vector Store (Chroma)")
    texts = [c["content"] for c in st.session_state["chunks"]]
    metadatas = [{"source": c["file"], "chunk_id": c["chunk_id"]} for c in st.session_state["chunks"]]
    with tempfile.TemporaryDirectory() as tmpdir:
        embeddings = OpenAIEmbeddings(model="openai.text-embedding-3-small")
        vectordb = Chroma.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas, persist_directory=tmpdir)
        st.session_state["vectorstore"] = vectordb
    st.success("Vector store created successfully!")

# Step 4. Build conversational RAG chain
if st.session_state["vectorstore"] and st.session_state.get("rag_chain") is None:
    retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model="openai.o3-mini", temperature=0)
    st.session_state["rag_chain"] = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, return_source_documents=True, chain_type="stuff"
    )

# Display chat history
for m in st.session_state["messages"]:
    st.chat_message(m["role"]).write(m["content"])

# Step 5. RAG conversation
if st.session_state.get("rag_chain"):
    question = st.chat_input("Ask something about your uploaded documents...")
    if question:
        st.chat_message("user").write(question)
        st.session_state["messages"].append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                try:
                    result = st.session_state["rag_chain"].invoke({
                        "question": question,
                        "chat_history": st.session_state["chat_history"]
                    })
                    answer = result["answer"]
                    st.write(answer)

                    # Sources with relevance score
                    st.markdown("** Sources (with relevance):**")
                    docs_with_scores = st.session_state["vectorstore"].similarity_search_with_score(question, k=5)
                    shown = set()
                    for doc, score in docs_with_scores:
                        src = doc.metadata.get("source", "Unknown")
                        cid = doc.metadata.get("chunk_id")
                        key = (src, cid)
                        if key in shown:
                            continue
                        shown.add(key)
                        preview = (doc.page_content or "").strip().replace("\n", " ")
                        preview = preview[:200] + ("..." if len(preview) > 200 else "")
                        st.markdown(f"- **{src}** (chunk {cid}) — relevance: {score:.4f}  \n  _{preview}_")

                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                    st.session_state["chat_history"].append((question, answer))
                except Exception as e:
                    st.error(f"RAG failed: {e}")
else:
    st.info("Please upload, split, and build the index first.")
