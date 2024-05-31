import os 
#from dotenv import load_dotenv
#from pathlib import Path
#import json
import sys
import dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder 
from langchain_core.messages import AIMessage
from langchain_openai import AzureChatOpenAI 

import ast 


def rag_query(llm, retriever, question):
    """
    Query the RAG model

    Parameters:
        llm : the LLM model
        retriever : Retrievers are used to find relevant documents or passages that contain the answer to a given query. 
        Question : Query or Question asked
    """
    try:
        #Step 2: Augment : Next, to augment the prompt with the additional context,
        #you need to prepare a prompt template. The prompt can be easily customized 
        #from a prompt template, as shown below.
        #The answer must provide the reference url. 
        
        template = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know.   
        
        Provide the top 3 answers.
        
        At the end, the answer must return source documents used.
        
        Question: {question} 
        Context: {context} 
        Answer:"""
 
        rag_prompt_template = PromptTemplate.from_template(template) 
         
        #Generate
        #Finally, you can build a chain for the RAG pipeline, chaining together the retriever, the prompt template and the LLM. 
        #Once the RAG chain is defined, you can invoke it.
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt_template
            | llm
            | StrOutputParser()
        )  
        result = rag_chain.invoke(question)  
        return result 
    
    
        # template = """Use the following context to answer the question at the end. 
        
        # If you don't know the answer, just say that you don't know, don't try to make up an answer.  
        
        # Search answer from indexed documents.
        
        # Must provide the document sources from MitreAttack.
        
        # Return the answer in the following Python dictionary format:

        # {{"question": "Question that was asked", "answer": "text", "title": "title", "url": "url"}} 

        # Return only a Python dictionary. 
        # {context}
        # Question: {question}
        # Helpful Answer:"""
        
    except Exception as e:
        print("Failed to run RAG query : ", e)
        raise e

def instantiate_model():
    """
    Instantiates a language model
    """
    try:
        #rag_query_dotenv()
        
        dotenv.load_dotenv()
 
        OPENAI_API_VERSION = "2023-05-15"
        DEPLOYMENT_NAME = "aisec-gpt35-turbo-16k"
        MODEL_VERSION = "0613"
        TEMPERATURE = 0
        SEED = 42
        
        ## Azure AISEC GPT 4 Config
        AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
        AZURE_OPENAI_ENDPOINT = os.getenv(
                "AZURE_OPENAI_ENDPOINT",
                "https://airt-report-gen.openai.azure.com/"
                )
 
        # os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        #os.environ["OPENAI_API_BASE"] ="https://airt-report-gen.openai.azure.com/"
        os.environ["OPENAI_API_VERSION"] = "2023-05-15"

        
        llm = AzureChatOpenAI(
            openai_api_version=OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_key=AZURE_OPENAI_API_KEY,
            deployment_name=DEPLOYMENT_NAME,
            model_version=MODEL_VERSION,
            temperature=TEMPERATURE,
            model_kwargs={
                "seed": SEED,
            },
        )
        
    except Exception as e:
        print("Failed to instantiate the language model:", e)
        raise e
    return llm


def get_retriever():
    """
    Get retriver 
    Retrievers are used to find relevant documents or passages that contain the answer to a given query. 
    They work by comparing the query against the indexed documents and returning the most relevant results. 
    
    Parameters:
        llm : the LLM model 
    """

    # Set the RAG index   
    rag_index="test-threat-intel-all-minilm-l6-v2"
    
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    # Create Pinecone vectorstore and LangChain retriever
    pc = Pinecone(api_key=PINECONE_API_KEY)
 
    # print("PINECONE INDEX : ", rag_index)
    # print("PINECONE KEY : ", PINECONE_API_KEY)
    
    MODEL_NAME = os.getenv("PINECONE_MODEL_NAME")
    
    embeddings = SentenceTransformerEmbeddings(model_name=MODEL_NAME)

    vectorstore = PineconeVectorStore(
            index=pc.Index(name=rag_index),
            embedding=embeddings,
            text_key="data_source",
            distance_strategy="cosine",
        )
    
    retriever = vectorstore.as_retriever(search_kwargs={"namespace":"AISecurity"})
    return retriever 



def get_vulnerability_assessment(llm,Question):
    """
    # Get the vulnerability assessment from threat intelligence database

    Parameters:
        llm : the LLM model 
        Question : Query 
    """ 
    #Step 1 : Retrieve
    # Get LangChain retriever
    rag_retriever=get_retriever()
     
    # Get the vulnerability assessment from threat intelligence database
    return  rag_query(
            llm,
            rag_retriever,
            Question,
        ) 
    
    
def initialize_llm_prompt():
        """
        Initializes the instance with the provided parameters.

        This method sets up a chat prompt template, an output parser,
        and a processing chain. It also stores the provided notes and AIBOM data.

        Args:
            llm: An instance of a language model.
            notes (str): A string of notes.
            aibom (dict): The AIBOM data.

        """
        
        # Instiantiate the Azure GPT model 
        llm = instantiate_model()  
        # Instiantiate the prompt
        generic_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a world class AI and machine learning expert 
                    and technical document writer who write clear and concise 
                    sentences.""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ]
        )

        output_parser = StrOutputParser() 
        chain = generic_prompt_template | llm | output_parser

        return llm, chain
    
    
def get_rag_query(llm, query):
    """
        Extract system vulnerabilities from the list of model names and dataset names extracted from the AIBOM
        using RAG query against VectorDB - Pinecone.
    """
    try:
        #only provide answers for open source models
        #query = "What are the top 3 AI System level Vulnerabilities associated with open-source models. " 
        #query = "What are the top 3 AI System level Vulnerabilities for open source models." 
        
        system_vul_output = []
        #query = "How do adversaries attack your system ? " 
        #query = "What are the top 3 AI System level Vulnerabilities associated with open-source models. " 
        system_vul_output = get_vulnerability_assessment(llm, query) 
        
    
        # Convert RAG outputs to dict
        #open_source_model_vul = ast.literal_eval(open_source_model_vul)  
        #print("OUTPUT OF RAG MODEL after : ",open_source_model_vul)
        # #only provide answers for open source datasets
        # #query = "What are the top 3 AI System level Vulnerabilities associated with open-source data. " 
        # query = "How do adversaries attack your system, use Mitre Attack to provide an answer" 
        # open_source_data_vul = get_vulnerability_assessment(llm, query) 
        # # Convert RAG outputs to dict
        # open_source_data_vul = ast.literal_eval(open_source_data_vul)
      
        
       #print("OUTPUT OF RAG DATA : ",open_source_data_vul)
        
     
            
        # vul_model_answer = open_source_model_vul["answer"][:-1]
        # vul_model_title  = open_source_model_vul["title"].replace("MITRE ATLAS™", "MITRE ATLAS")
        # vul_model_url    = open_source_model_vul["url"] 
    
        # vul_data_answer = open_source_data_vul["answer"][:-1]
        # vul_data_title  = open_source_data_vul["title"].replace("MITRE ATLAS™", "MITRE ATLAS")   
        # vul_data_url    = open_source_data_vul["url"] 
    
        
        # system_vul_output = [
        #     {
        #         "summary": vul_model_title,
        #         "details": vul_model_answer,
        #         "vulnerability_url": vul_model_url
        #     }, 
        #     {
        #         "summary": vul_data_title,
        #         "details": vul_data_answer,
        #         "vulnerability_url": vul_data_url
        #     } 
        # ]
    
        # First, check if the list of system culnerabilities is empty, if not, that is a vulnerability
        # if len(system_vul_output) == 0:  
        #     system_vul_output= [ 
        #         {
        #             "summary": "NA",
        #             "details":  "There were no system vulnerabilities identified (using RAG query) for the Vulnerability card. This could be a vulnerability.",
        #             "vulnerability_url": "NA"
        #         } 
        #     ]
        
        return system_vul_output

        #return open_source_model_vul, open_source_data_vul

    except Exception as e:
        print(f"Error while running RAG query : {e}")
        
        
        
def get_system_vulnerability(query):
    """
    Populate the vulnerability card with the system vulnerabilities
    """
    try:
        #Initilize llm & prompt 
        dotenv.load_dotenv()
        llm, chain = initialize_llm_prompt() 
         
        #query = "what are the top 3 ways that adversaries can attack your system ? " 
        system_vul_output=get_rag_query(llm, query)
        print("\n\nQUESTION: ",query)
        print("\nOUTPUT: ",system_vul_output)
        
        # query = "What are the top 3 AI System level Vulnerabilities associated with open-source models ? " 
        # system_vul_output=get_rag_query(llm, query)
        # print("\nOUTPUT OF RAG QUERY: ",system_vul_output)
         
        #Get top 5 System Vulnerabilities using input prompts 
        #system_vul_list = await add_system_vulnerabilities(chain, model_context, data_context, system_rag_vul_list) 
        #logger.info(f"FINAL system_vul_list {system_vul_list}")  
        
        #return open_source_model_vul, open_source_data_vul
        
        return system_vul_output
         
    except Exception as e:
        print(f"Error while running system vulnerability script : {e}")
        
        
if __name__ == "__main__": 
    query_param_1 = sys.argv[1]
    system_vul_output=get_system_vulnerability(query_param_1) 
    #print("Done working on Rag Query")
    
    #print(json.dumps(system_vul_output, indent=4))
    
    