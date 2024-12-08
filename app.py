import streamlit as st
import fitz  
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from groq import Groq  

load_dotenv()

def query_llm_api(user_input, language):
    client = Groq()
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "user",
                "content": f"""Answer in {language} and explain in English as well, Use the given context alone to answer questions as a chatbot, if out of context, please mention: {user_input}
                             if user asks for help on what to ask next, based on the conversation in pdf, suggest what user should say in {language} and explain in english as well"""
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False  
    )
    
    response_content = completion.choices[0].message.content
    return response_content

def custom_qa_system(question, vector_store, language):
    retriever = vector_store.as_retriever()
    relevant_docs = retriever.get_relevant_documents(question)

    context_text = "\n".join([doc.page_content for doc in relevant_docs])
    combined_input = f"Context:\n{context_text}\n\nQuestion:\n{question}"

    response = query_llm_api(combined_input, language)
    return response


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf(pdf_files):
    documents = []
    for pdf in pdf_files:
        pdf_document = fitz.open(stream=pdf.read(), filetype="pdf")
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text()
            documents.append(Document(page_content=page_text, metadata={"page": page_num}))

    # Specify persist directory
    vector_store = Chroma.from_documents(documents, embedding=embeddings, persist_directory="chroma_db")
    vector_store.persist()  # Ensures database tables are created
    return vector_store

# Streamlit App
def main():
    st.set_page_config(page_title="Chat with PDF", page_icon=":books:")
    st.header("ConvoTutor")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Language dropdown in the sidebar
    with st.sidebar:
        st.subheader("Upload PDF Document")
        pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        if st.button("Process PDFs") and pdf_files:
            with st.spinner("Processing PDF..."):
                st.session_state.vector_store = process_pdf(pdf_files)
            st.success("PDFs processed successfully!")

    # Select language
    language = st.selectbox("Choose Language", [
        "English", "Spanish", "French", "German", "Chinese", "Japanese", "Russian", 
        "Italian", "Portuguese", "Dutch", "Korean", "Arabic", "Turkish", 
        "Swedish", "Hindi"
    ])

    # Input box for questions
    user_question = st.text_input("Ask a question about the document:")
    if user_question and st.session_state.vector_store:
        with st.spinner("Querying the document..."):
            answer = custom_qa_system(user_question, st.session_state.vector_store, language)
        st.write("Answer:", answer)
    elif user_question:
        st.warning("Please upload and process a PDF document first.")

if __name__ == "__main__":
    main()