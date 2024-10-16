import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_mistralai import MistralAIEmbeddings
import streamlit as st
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from streamlit_chat import message
import time
os.environ["GROQ_API_KEY"] = st.secrets["Groq_API"]
os.environ["MISTRALAI_API_KEY"] = st.secrets["Mistral_API"]
llm = ChatGroq(model="llama3-8b-8192")
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Welcome to Prestige Constructions, How can I assist you?"]
if 'requests' not in st.session_state:
    st.session_state['requests'] = []
if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)
if "vector" not in st.session_state:
    st.session_state.embedding =  MistralAIEmbeddings(model="mistral-embed", api_key=st.secrets["Mistral_API"])
    st.session_state.documents = []
    #st.session_state.urls = ["https://www.prestigeconstructions.com/residential-projects/mumbai/the-prestige-city-mulund/forest-hills","https://www.prestigeconstructions.com/residential-projects/bangalore/prestige-pine-forest","https://www.prestigeconstructions.com/residential-projects/bangalore/prestige-raintree-park","https://www.prestigeconstructions.com/residential-projects","https://www.prestigeconstructions.com/residential-projects/bangalore/prestige-kings-county","https://www.prestigeconstructions.com/residential-projects/bangalore/prestige-camden-gardens","https://www.prestigeconstructions.com/residential-projects/hyderabad/the-prestige-city-rajendra-nagar","https://www.prestigeconstructions.com/residential-projects/bangalore/prestige-somerville","https://www.prestigeconstructions.com/residential-projects/hyderabad/prestige-vaishnaoi-rainbow-waters","https://www.prestigeconstructions.com/residential-projects/hyderabad/the-prestige-city-rajendra-nagar/bellagio","https://www.prestigeconstructions.com/residential-projects/hyderabad/the-prestige-city-rajendra-nagar/apartment","https://www.prestigeconstructions.com/residential-projects/bangalore/prestige-glenbrook","https://www.prestigeconstructions.com/residential-projects/hyderabad/prestige-clairemont","https://www.prestigeconstructions.com/residential-projects/mumbai/the-prestige-city-mulund","https://www.prestigeconstructions.com/residential-projects/mumbai/the-prestige-city-mulund/bellanza","https://www.prestigeconstructions.com/residential-projects/cochin/prestige-panorama","https://www.prestigeconstructions.com/residential-projects/cochin/prestige-cityscape","https://www.prestigeconstructions.com/residential-projects/mumbai/prestige-daffodils","https://www.prestigeconstructions.com/residential-projects/mumbai/prestige-ocean-towers","https://www.prestigeconstructions.com/residential-projects/cochin/prestige-eden-garden","https://www.prestigeconstructions.com/residential-projects/hyderabad/prestige-beverly-hills"]
    #for url in st.session_state.urls:
    #time.sleep(5)
    st.session_state.loader = WebBaseLoader("https://www.prestigeconstructions.com/residential-projects")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.documents.extend(st.session_state.docs)
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    st.session_state.doc = st.session_state.text_splitter.split_documents(st.session_state.documents)
    st.session_state.db = FAISS.from_documents(documents = st.session_state.doc, embedding=st.session_state.embedding)
prompt = ChatPromptTemplate.from_template("""You are a Real Estate agent for Prestige Constructions, only suggests properties to user and if asked anything else than those things just say you cant help them and You have list of properties in context, suggest users properties strictly from context only
<context>
{context}
</context>
Question: {input}""")
document_chain = create_stuff_documents_chain(llm,prompt)
retriever = st.session_state.db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever,document_chain)
st.title("Prestige Constructions Pvt. Ltd.")
...
response_container = st.container()
textcontainer = st.container()
...
with textcontainer:
    query = st.text_input("Query: ")
    if query:
        with st.spinner("typing..."):
            response = retrieval_chain.invoke({"input":query})
        st.session_state.requests.append(query)
        st.session_state.responses.append(response['answer'])
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
