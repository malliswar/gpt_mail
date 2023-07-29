
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from flask import Flask, jsonify, request
from dotenv import load_dotenv
import openai
import os
import streamlit as st



OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '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')
openai_api_key = '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'#--vectorise the mails CSV
csv_loader = CSVLoader(file_path="sales_reply_mails_csv.csv")
csv_documents = csv_loader.load()



#print((csv_documents[1]))
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
faiss_openAIEmb = FAISS.from_documents(csv_documents, embeddings)

def retrieve_info(query):
    similar_resp = faiss_openAIEmb.similarity_search(query, k=2)
    page_cont = [doc.page_content for doc in similar_resp]
    #print(page_cont)
    return page_cont

#customer_msg = " Hi Paul, How are you"
#print(retrieve_info(customer_msg))

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613",openai_api_key=OPENAI_API_KEY )

template = """
You are software developer who takes up jobs from international clients.
I will share a client message with you and you will give me the best replay to that,
based on the previous messages.
and you will follow ALL of the rules below:

1/ Response should be identical to the past best practices in terms of length, tone of voice and other details.

2/ If the best practices are irrelavent, try to mimic the style of best practices.

Below is the message I recieved from client:
{message}

Here is the list of best practices of how we normally respond  to client
{best_practice}

Please write the best response that I should send to this client

"""
prompt = PromptTemplate(
input_variables= {"message","best_practice"},
template= template

)

chain = LLMChain(llm = llm, prompt= prompt )

def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message,best_practice = best_practice)
    return response

message = """
Hi Malliswar,
Last week I send my feedback regarding the PEAK tool

Regards,
Paul
"""
response = generate_response(message)
#print(response)



def main():
    st.set_page_config(
        page_title="Mail Response Generator", page_icon =":bird:")


    st.header("Mail Response Generator :bird:")
    message = st.text_area("customer message")
    if message:
        st.write("Generating the reply...")
        result = generate_response(message)
        st.info (result)

if __name__ == '__main__' :
    main()

    
# RUN above code in command propmt using below code
#python -m streamlit run app.py

