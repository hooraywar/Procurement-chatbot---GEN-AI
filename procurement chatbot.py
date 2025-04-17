from langchain_openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationChain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever
import bs4 


load_dotenv(".env")

openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACKING_V2'] = 'true'


loader = WebBaseLoader(
    web_paths= ("https://www.vaival.com/",),
    bs_kwargs = dict(
        parse_only = bs4.SoupStrainer("p")
    )
)

doc= loader.load()

print(doc)

vectorstore = FAISS.from_documents(doc, OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

output_parser = StrOutputParser()


llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory
)


chat_history = []




prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful assistant responsible for generating a comprehensive  proposal. 
                        You will be provided with a job description, and based on that, you will generate the following sections:

                        1. *Company Background*: Provide detailed information about the company, highlighting its strengths, expertise, and how it aligns with the job description. Focus on:
                            - Company's history, mission, and vision.
                            - Relevant qualifications and achievements that make the company a suitable candidate.
                            - Key differentiators that set the company apart from competitors.

                        2. *Compliance Statements*: Draft formal compliance statements ensuring that the company adheres to all necessary laws, regulations, and standards. Include:
                            - Adherence to industry-specific regulations (e.g., UVgO,VgV standards).
                            - Any certifications or accreditations held by the company.
                            - A general statement assuring legal and ethical compliance with the job requirements.

                        3. *Project Experience*: Highlight relevant project experience, focusing on:
                            - Similar projects successfully completed by the company.
                            - Key outcomes and success stories that demonstrate the company's ability to deliver results.
                            - Technologies and methodologies used that are pertinent to the job description.

                        The structure of a procurement proposel is as follows:
                                1. Company Background
                                2. Compliance Statements
                                3. Project Experience
                        The proposal should be structured clearly and professionally to align with job description expectations."""
                    "\n\n"
                    "{context}"),
        MessagesPlaceholder(variable_name="chat_history"),  
        ("user", "Question: {question}"),
    ]
)




#context = "I have a 2 years of experience in the field of development also I am multi-tasking and can adapt multi platforms and languages"
context = "Vaival have a 2 years of experience in the field of AI, blockchain , Deep laerning and Machine laerning development "


st.title("Procurement proposel Chatbot")
input_text = st.text_input("Let's AI write your procurement proposel:")


llm = OpenAI(openai_api_key=openai_api_key)



#chain = LLMChain(prompt=prompt, llm=llm)
chain = prompt | llm 


def get_response():

    if input_text:
  
        result = chain.invoke({'chat_history': chat_history,'question': input_text  , 'context': context})
        st.write(result)

        chat_history.append({"role": "user", "content": input_text})
        chat_history.append({"role": "assistant", "content": result})

# Main interaction loop
if _name_ == "_main_":
    get_response()