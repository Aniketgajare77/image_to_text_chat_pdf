import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from datetime import datetime
import pypdfium2 as pdfium
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import os
import pytesseract
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords

# Download the stop words dataset
nltk.download('stopwords')

headers={
    "authorization": st.secrets["OPENAI_API_KEY"]
}


def save_uploaded_file(uploaded_file):
    try:
        # Create a directory to save the file
        save_path = 'uploaded_files'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Create a unique filename to prevent file overwrites
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{current_time}_{uploaded_file.name}"
        complete_path = os.path.join(save_path, file_name)

        # Write the file to the directory
        with open(complete_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return complete_path
    except Exception as e:
        return str(e)

def convert_pdf_to_images(file_path, scale=300/72):
    
    pdf_file = pdfium.PdfDocument(file_path)
    
    st.write(len(pdf_file))
    page_indices = [i for i in range(len(pdf_file))]
    
    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices, 
        scale=scale,
    )
    
    list_final_images = [] 
    
    for i, image in zip(page_indices, renderer):
        
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        list_final_images.append(dict({i: image_byte_array}))
    
    return list_final_images

def extract_text_with_pytesseract(list_dict_final_images):
    
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []
    
    for index, image_bytes in enumerate(image_list):
        
        image = Image.open(BytesIO(image_bytes))
        raw_text = str(pytesseract.image_to_string(image))  # Fix the function call here
        image_content.append(raw_text)
    
    return "\n".join(image_content)

def extract_valid_text_with_dates(text):
    
    # Patterns for URLs, emails, phone numbers, words, abbreviations, numbers, dates, and specific numericals
    url_pattern = r'\b(?:https?://)?\w+(?:\.\w+)+(?:/\w+)*\b'
    email_pattern = r'\b[\w.-]+@[\w.-]+\.\w+\b'
    phone_pattern = r'\b(?:\+?\d[\d-]{7,15})\b'
    word_pattern = r'\b[A-Z]{2,}(?:\.[A-Z]{2,})*\b|\b[a-zA-Z]+\b'
    number_pattern = r'\b\d+\b'
    date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{2,4}[/-]\d{1,2}[/-]\d{1,2})\b'
    temp_pattern = r'\b-?\d+\.?\d*\s?[°º]?[CF]\b'  # Temperature pattern, e.g., 30C, 85 F, -5°C
    currency_pattern = r'\b(?:\$\d+\.?\d*|\d+\.?\d*\s?(?:INR|Rs))\b'  # Currency pattern, e.g., $100, 500 INR, Rs 150.50
    additional_number_pattern = r'\b\d+\+\s\w+\b'  # Additional numbers, e.g., 250+ projects

    # Combined pattern
    combined_pattern = '|'.join([url_pattern, email_pattern, phone_pattern, word_pattern, number_pattern, date_pattern, temp_pattern, currency_pattern, additional_number_pattern])

    # Find all matches
    all_matches = re.findall(combined_pattern, text)

    # Joining all valid matches
    valid_text = ' '.join(all_matches)

    # Remove common stop words
    stop_words = set(stopwords.words('english'))
    filtered_text = ' '.join(word for word in valid_text.split() if word.lower() not in stop_words)

    return filtered_text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    
    #load_dotenv()
    #OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'")
        if st.button("Process"):
            file_path = save_uploaded_file(pdf_docs)
            st.success(file_path)
            with st.spinner("Processing"):
                 
                # get pdf text
                var1 = convert_pdf_to_images(file_path)  
                # 2
                var2 = extract_text_with_pytesseract(var1)
                # 3
                file_content = extract_valid_text_with_dates(var2)

                # get the text chunks
                text_chunks = get_text_chunks(file_content)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)



if __name__ == '__main__':
    main()
