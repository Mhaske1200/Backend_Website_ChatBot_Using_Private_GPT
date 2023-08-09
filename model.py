# import os
# from dotenv import load_dotenv
# load_dotenv()
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
#
# urls = [
#     'https://askadvi.org/',
#     'https://askadvi.org/students/',
#     'https://askadvi.org/counselors/',
#     'https://askadvi.org/faq/'
# ]
#
# from langchain.document_loaders import UnstructuredURLLoader
# loaders = UnstructuredURLLoader(urls=urls)
# data = loaders.load()
#
# # Text Splitter
# from langchain.text_splitter import CharacterTextSplitter
#
# text_splitter = CharacterTextSplitter(separator='\n',
#                                       chunk_size=1000,
#                                       chunk_overlap=300)
#
# docs = text_splitter.split_documents(data)
#
# import pickle
# import faiss
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
#
# embeddings = OpenAIEmbeddings()
#
# import tiktoken
# vectorStore_openAI = FAISS.from_documents(docs, embeddings)
#
# with open("faiss_store_openai.pkl", "wb") as f:
#   pickle.dump(vectorStore_openAI, f)
#
# with open("faiss_store_openai.pkl", "rb") as f:
#     VectorStore = pickle.load(f)
#
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.chains.question_answering import load_qa_chain
# from langchain import OpenAI
#
# llm=OpenAI(temperature=0)
#
# chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())
#
# output = chain({"question": "what is student resources?"}, return_only_outputs=True)
# print(output.get('answer'))
# print("DONE")

#-----------------------------------------------------------------------------

import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
import requests
from bs4 import BeautifulSoup

def create_sub_url(url):
    link_list = []

    response = requests.get(url)
    if response.status_code == 200:
        homepage_content = response.content
        soup = BeautifulSoup(homepage_content, "html.parser")
        links = soup.find_all("a")

        for link in links:
            href = link.get("href")
            if href and href.startswith("https"):
                link_list.append(href)
    print(link_list)
    return link_list
def create_pickle(urls):

    from langchain.document_loaders import UnstructuredURLLoader
    loaders = UnstructuredURLLoader(urls=urls)
    data = loaders.load()

    # Text Splitter
    from langchain.text_splitter import CharacterTextSplitter

    text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=1000,
                                      chunk_overlap=300)

    docs = text_splitter.split_documents(data)

    import pickle
    from langchain.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()

    import tiktoken
    vectorStore_openAI = FAISS.from_documents(docs, embeddings)

    with open("faiss_store_openai.pkl", "wb") as f:
        pickle.dump(vectorStore_openAI, f)

    # with open("faiss_store_openai.pkl", "rb") as f:
    #     VectorStore = pickle.load(f)
    #
    # from langchain.chains import RetrievalQAWithSourcesChain
    # from langchain.chains.question_answering import load_qa_chain
    # from langchain import OpenAI
    #
    # llm=OpenAI(temperature=0)
    #
    # chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())
    #
    # output = chain({"question": "what is student resources?"}, return_only_outputs=True)
    # print(output.get('answer'))
    print("DONE")