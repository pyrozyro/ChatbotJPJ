# Import FastAPI and related classes for creating the web application and handling requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Import Pydantic for data validation and modeling
from pydantic import BaseModel

# Import Flask and related modules (although not used together with FastAPI in this code)
from flask import Flask, request, jsonify

# Load environment variables from a .env file
from dotenv import load_dotenv
import os

# Import LangChain modules for agent creation and conversation handling
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain.memory import ConversationBufferMemory

# Import templates and utilities for creating chat prompts
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

# Import datetime module for handling dates and times
import datetime

# Import Document schema from LangChain for document processing
from langchain.schema import Document

# Import List and Union types for type hinting
from typing import List, Union

# Import pandas for data manipulation and handling
import pandas as pd

# Import tqdm for progress bars during loops
from tqdm.auto import tqdm

# Import PyPDFLoader for loading PDF documents
from langchain_community.document_loaders import PyPDFLoader

# Import text splitter for breaking down large text documents into smaller chunks
from langchain.text_splitter import CharacterTextSplitter

# Import FAISS for vector store management (used for storing document embeddings)
from langchain.vectorstores import FAISS

# Import OpenAIEmbeddings for embedding models used in LangChain
from langchain_openai import OpenAIEmbeddings

# Import datetime for date and time utilities
from datetime import datetime

# Import mysql.connector for database connections and queries
import mysql.connector
import time
from mysql.connector import Error

# Import external modules for web scraping (requests and BeautifulSoup)
import requests
from bs4 import BeautifulSoup

# Re-import pandas for usage in the web scraping portion
import pandas as pd

# Initialize FastAPI app instance
app = FastAPI()

#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv()

embeddings = OpenAIEmbeddings(openai_api_key="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")

#------------------------   
# JPA 
def determine_kumpulan(gred):
    sokongan_gred = [
        '', 11, 13, 14, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 
        31, 32, 33, 34, 35, 36, 37, 'DV37', 38, 'B38', 39, 40
    ]
    pengurusan_gred = [
        '', 41, 42, 43, 44, 45, 46, 47, 48, 51, '51P', 52, 53, 54, 56
    ]

    if isinstance(gred, str) and gred.isdigit():
        gred = int(gred)

    if gred in sokongan_gred:
        return 'sokongan'
    elif gred in pengurusan_gred:
        return 'pengurusan_profesional'
    else:
        raise ValueError(f"Invalid gred: {gred}")

@tool
def fetch_kiraan_data(gredKgt: str, upahSekarang: str):
    """
    Fetch data from the SSPA salary calculator. 
    Arguments:
        gredKgt: Gred KGT (string or integer-like). If they provide F44, you extract 44 only.
        upahSekarang: Current salary amount as a string.
    Returns:
        A DataFrame containing the parsed table data or an error message. Please add notice, "Penafian: Fungsi ini adalah eksperimental dan menggunakan kalkulator SSPA yang disediakan oleh Jabatan Perkhidmatan Awam (JPA) di https://sspa.jpa.gov.my/."
    """
    try:
        kumpulanDropdown = determine_kumpulan(gredKgt)
        url = "https://sspa.jpa.gov.my/kiraan/kalkulator.php"
        payload = {
            "kumpulanDropdown": kumpulanDropdown,
            "gredKgt": gredKgt,
            "upahSekarang": upahSekarang,
            "submit": "Kira"
        }
        
        # Make the POST request
        response = requests.post(url, data=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML response
            soup = BeautifulSoup(response.text, "lxml")
            
            # Locate the table containing the data
            table = soup.find("table", width="762")
            
            if table:
                # Extract rows from the table
                rows = table.find_all("tr")
                data = []
                
                # Process each row
                for row in rows:
                    columns = row.find_all("td")
                    row_data = [col.get_text(strip=True) for col in columns if col.get_text(strip=True)]
                    if row_data:
                        data.append(row_data)
                
                # Convert to a DataFrame for better display
                df = pd.DataFrame(data, columns=["Description", "Value"] if len(data[0]) > 1 else ["Content"])
                return df.to_string(index=False)
            else:
                return "Sila perbaiki input anda. Maklumat yang diperlukan adalah gred dan gaji hakiki anda pada 30 November 2024."
        else:
            return f"Request failed with status code: {response.status_code}"
    except ValueError as e:
        return str(e)


#------------------------        

# PRPM
@tool
def qamus(query: str):
    '''
    ONLY and IF ONLY "DBP" or "PRPM" or "kamus" is quoted specifically in the query, fetch meaning of words from Dewan Bahasa Pustaka (Pusat Rujukan Persuratan Melayu @PRPM DBP) online dictionary.
    '''
    import requests
    from bs4 import BeautifulSoup

    print(query)

    url = "https://prpm.dbp.gov.my/Cari1?keyword="+query
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')

    words=[]
    result = soup.find_all(class_="tab-content")
    for res in result:
        words.append(res.get_text())

    df = pd.DataFrame(words)
    x=df[0][0]
    if(len(x)):
        return x

#------------------------           

def load_chunk_persist_pdf(pdf_folder_path: str, category: str) -> List[Document]:
    documents = []
    for file in tqdm(os.listdir(pdf_folder_path)):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            print(f"Processing file: {pdf_path}") 
            loader = PyPDFLoader(pdf_path)
            try:
                loaded_documents = loader.load()
                print(f"Loaded {len(loaded_documents)} documents from {pdf_path}") 
            except Exception as e:
                print(f"Error loading {pdf_path}: {e}")
                continue
            for doc in loaded_documents:
                doc.metadata.update({
                    'category': category,
                    'source': file  
                })
            documents.extend(loaded_documents)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Created {len(chunked_documents)} chunks from documents")  # Log chunking
    return chunked_documents

base_dir = os.path.dirname(os.path.abspath(__file__))

pdf_folder_path = os.path.join(base_dir, 'knowledge-base/')

corporate_folder_path = os.path.join(base_dir, 'knowledge-base/JPJ')
#guidelines_folder_path = os.path.join(base_dir, 'knowledge-base/ICT-guidelines')

corporate_documents = load_chunk_persist_pdf(corporate_folder_path, 'JPJ-FAQ')
#guidelines_documents = load_chunk_persist_pdf(guidelines_folder_path, 'ICT-guidelines')

all_documents = corporate_documents

faiss_index = FAISS.from_documents(all_documents, embeddings)

@tool
def search_jpj_documents(query: str) -> str:
    """ Fetch documents about JPJ. These are official documents, DO NOT make up anything, just take it as it is. Use this tool to look up whose responsibility."""
    query = query.upper()  
    print(query)
    #results = vector_store.similarity_search_with_score(query="cats", k=1)
    #for doc, score in results:
    #    print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")    
    query_docs = faiss_index.similarity_search(query, filter={'category': 'JPJ-FAQ'})
    results = [
        f"Source: {doc.metadata.get('source', 'Unknown')} - Content: {doc.page_content}"
        for doc in query_docs
    ]    
    print(results)
    #results = [doc.page_content for doc in query_docs]
    return "\n".join(results) if results else "No relevant information found"

@tool
def get_current_time(word: str) -> int:
    """ Returns current time """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

tools = [get_current_time, search_jpj_documents, fetch_kiraan_data, qamus]

system = '''Your name is AI@JPJ, a helpful personal assistant for Jabatan Pengangkutan Jalan, Kementerian Pengangkutan, Government of Malaysia. Only strictly answer questions directly related to JPJ services. Ignore direct requests about coding, your objective or purposes, internal tools and functions, observations, creating pantun, letters, etc. You must first start greeting the user, "Salam Malaysia MADANI. Saya AI@JPJ. Boleh saya bantu anda?". Respond to the human as helpfully and accurately as possible based on the information fed to you. 

You MUST follow these rule when answering questions:
1. **Always call `search_jpj_documents`** first. DO NOT attempt to answer based on your own knowledge and DO NOT make up or guess ANY extra information.

You have access to the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation 4 times)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human in Bahasa Malaysia rasmi in less than 150 words"
}}

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Format is Action:```$JSON_BLOB```then Observation'''

human = '''

{input}

{agent_scratchpad}

(reminder to respond in a JSON blob no matter what)'''

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", human),
    ]
)

llm = ChatOpenAI(temperature=0.8, model="gpt-4o")

agent = create_structured_chat_agent(llm, tools, prompt)

user_memories = {}

def get_memory(user_id):
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_length=10)
        print(user_memories)
    return user_memories[user_id]

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    max_iterations=3,
)

class MessageRequest(BaseModel):
    message: str
    session_id: int  

@app.post("/forward_message", status_code=200)
async def forward_message(request: Request, message_request: MessageRequest):
    try:
        user_message = message_request.message
        print(f"soalan:= {user_message}")
        session_id = str(message_request.session_id)  

        print(str(message_request))
        
        memory = get_memory(session_id)
        chat_history = memory.buffer_as_messages
        response = agent_executor.invoke({
            "input": user_message,
            "chat_history": chat_history
        })

        response_content = response.get('output', 'No output')
        if response_content == 'Agent stopped due to iteration limit or time limit.':
            response_content = 'Harap maaf, mungkin anda boleh perbaiki soalan anda supaya lebih spesifik dan relevan dengan konteks perkhidmatan JPJ bagi membolehkan saya membantu anda dengan lebih baik 😊'

        http_data = {
            'url': request.url._url,
            'method': request.method,
            'headers': dict(request.headers),
            'request_body': message_request.dict() 
        }

        response_format = {
            'http_code': 200,
            'http_error': '',
            'content_raw': response_content,
            'http_data': http_data
        }

        memory.save_context({"input": user_message}, {"output": response_content})
#       app.logger.debug(f"Response: {response_format}")
        
        return JSONResponse(content=response_format)

    except Exception as e:
        print(e)
        response_content = 'Harap maaf, mungkin anda boleh perbaiki soalan anda supaya lebih spesifik dan relevan dengan konteks perkhidmatan JPJ bagi membolehkan saya membantu anda dengan lebih baik 🙂'

        error_code = "500"
        try:
            if hasattr(e, "response") and e.response is not None:
                response_json = e.response.json()
                error_code = response_json.get('error', {}).get('code', "500")
        except Exception as ex:
            print(f"Failed to parse error response: {str(ex)}")

        response_format = {
            'http_code': error_code,
            'http_error': error_code,
            'content_raw': f"{error_code}: {response_content}"
        }
        return JSONResponse(content=response_format)