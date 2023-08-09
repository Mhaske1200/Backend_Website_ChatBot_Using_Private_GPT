# import pickle
# from flask import Flask, render_template, request, jsonify
# import os
# from flask_cors import CORS
#
# from dotenv import load_dotenv
# load_dotenv()
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
#
# app=Flask(__name__)
# CORS(app,origins="*")
#
# #Deserialise / Depickle
# with open("faiss_store_openai.pkl", "rb") as f:
#     vectorstore = pickle.load(f)
#
# # @app.route("/")
# # def hello():
# #     return render_template("index.html")
#
# @app.route('/check',methods=['POST','GET'])
# def predict_class():
#     data = request.get_json()
#     question = data.get("userInput")
#
#     #features=[x for x in request.form.values()]
#     #print(features[0])
#
#     from langchain.chains import RetrievalQAWithSourcesChain
#     from langchain.chains.question_answering import load_qa_chain
#     from langchain import OpenAI
#     llm = OpenAI(temperature=0)
#     chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
#
#     if question:
#         output = chain({"question": question}, return_only_outputs=True)
#         print(output.get('answer'))
#         #render_template("index.html",check=output.get('answer'))
#         sudh = jsonify(output.get('answer'))
#         print(sudh)
#         return sudh
#     else:
#         return print("Error") #("index.html",check="Please Enter Data !")
#
# if __name__ == "__main__":
#     app.run(debug=True) #create a flask local server
#-----------------------------------------------------------------------------

import pickle
from flask import Flask, render_template, request
import os

from langchain.callbacks import StreamingStdOutCallbackHandler

import model as m
from dotenv import load_dotenv
load_dotenv()
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
model_type = os.getenv('MODEL_TYPE')
model_path = os.getenv('MODEL_PATH')
model_n_ctx = os.getenv('MODEL_N_CTX')
model_n_batch = int(os.getenv('MODEL_N_BATCH', 8))
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
app=Flask(__name__)

#----------------------------------------------------

from langchain.llms import GPT4All
from langchain.chains import RetrievalQAWithSourcesChain
callbacks = [StreamingStdOutCallbackHandler()]
# from langchain import OpenAI
# llm = OpenAI(temperature=0)
llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)

#Deserialise / Depickle
# if __name__=='__main__':
#     with open("faiss_store_openai.pkl", "rb") as f:
#         VectorStore = pickle.load(f)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route('/make_model',methods=['POST','GET'])
def predict():

    features=[x for x in request.form.values()]
    print(features[0])

    if features[0]:
        urls = m.create_sub_url(features[0])
        print(urls)
        m.create_pickle(urls)
        return render_template("index.html",pickle=features[0])
    else:
        return render_template("index.html",pickle="Please Enter Data !")


@app.route('/check',methods=['POST','GET'])
def predict_class():

    with open("faiss_store_openai.pkl", "rb") as f:
        VectorStore = pickle.load(f)
    features=[x for x in request.form.values()]
    print(features[0])

    # from langchain.llms import GPT4All
    # from langchain.chains import RetrievalQAWithSourcesChain
    # callbacks = [StreamingStdOutCallbackHandler()]
    # # from langchain import OpenAI
    # # llm = OpenAI(temperature=0)
    # llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())

    if features[0]:
        output = chain({"question": features[0]}, return_only_outputs=True)
        print(output.get('answer'))
        return render_template("index.html",check=output.get('answer'))
        #sudh = jsonify(output.get('answer'))
        #print(sudh)
        #return sudh
    else:
        return render_template("index.html",check="Please Enter Data !")

if __name__ == "__main__":
    app.run(debug=True) #create a flask local server
