#seetting up the environment 

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'advanced-rag'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQQ_API_KEY")

#import 
import bs4
from langchain import hub 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


### indexing 

#load 

laoder = WebBaseLoader(
    web_path=("",),
    bs_kwargs=dict(
        parse_only = bs4.SoupStrainer(
            class_=("post-content","post-title","post-header")
        )
    ),
)


