from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import TypedDict, Literal
import tabula
import os

from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


load_dotenv()

# Loading the Grobal variables
groq_api_keys=os.getenv("Groq_API_Key")
groq_temperature=os.getenv("Groq_Temperature")
groq_model_name=os.getenv("Groq_Model_Name")

ollama_embedding_model_name=os.getenv("Ollama_Embedding_Model_Name")
ollama_embedding_model_temperature=os.getenv("Ollama_Embedding_Model_Temperature")

# This is defination of the text.
class KnowledgeIntelligenceState(TypedDict):
    pdf_file_path:str
    query:str 
    loader:any
    splitter:any
    embedding:any
    retriever_data:any
    prompt:any
    prompt2:any
    llm:any
    answer:any

    tabular_data:any
    tabular_to_text_conversion:any
    tabular_splitted_data:any
    tabular_retriever:any
    tabular_prompt:any


# Texual loader node is here
def document_loading(state:KnowledgeIntelligenceState):
    loader=PyPDFLoader(file_path=state["pdf_file_path"])
    loaded_data=loader.load()
    return {
        "loader": loaded_data
    }

# Tabular loader node is here
def Tabular_Document_Loader(state:KnowledgeIntelligenceState):
    tfs = tabula.read_pdf(
        state["pdf_file_path"],
        pages="all"
    )
    if tfs:
        return {'tabular_data':tfs[0]}
    else:
        return {'tabular_data':"none"}
    
# Table to text conversion node
def Tabular_to_text_conversion(state:KnowledgeIntelligenceState):
    state['prompt2']=PromptTemplate(template="""
        You are an expert in converting noisy and unstructured tabular data into  clear, well-written textual format.

        You will be provided with data that originally comes from a table but may be poorly formatted or flattened. Your task is to:

            1) Identify the column names and row relationships implicitly present in the data.
            2) Preserve the semantic relationships between columns and rows.
            3) Convert each row into clear, complete, and grammatically correct 
            4) sentences using the column context.
            5) Avoid losing any information present in the table.
            Do not hallucinate or add new data.

        The final output should be a coherent textual explanation, suitable for documentation or downstream RAG usage.
        This is the tabular data:
        {tabular_data}
    """,input_variables=['tabular_data'])
    state['llm'] = ChatGroq(model=groq_model_name,temperature=groq_temperature)
    parser = StrOutputParser()
    chain = state['prompt2'] | state['llm'] | parser
    data=chain.invoke({
        "tabular_data": state["tabular_data"]
    })
    return {'tabular_to_text_conversion':data}

# splitter node is here
def text_splitting(state:KnowledgeIntelligenceState):
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    splitted_text=splitter.split_documents(state["loader"])
    return {
        "splitter": splitted_text
    }

# Table splitter node is here
def table_splitting(state:KnowledgeIntelligenceState):
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    docs = [
        Document(
            page_content=state["tabular_to_text_conversion"],
            metadata={"type": "table"}
        )
    ]

    splitted_text = splitter.split_documents(docs)
    
    return {
        "tabular_splitted_data": splitted_text
    }

# embedding node is here
def doc_embedding(state:KnowledgeIntelligenceState):
    emb=OllamaEmbeddings(model=ollama_embedding_model_name)
    return {"embedding":emb}

# vectorstore node is here
def vectorstore_node(state:KnowledgeIntelligenceState):
    db=FAISS.from_documents(
        documents=state['splitter'],
        embedding=state["embedding"]
    )
    retriever=db.as_retriever(search_type="mmr",search_kwargs={"k": 4})
    retrieved_context = retriever.invoke(state["query"])

    return {"retriever_data":retrieved_context}

# Tabular retriever node is here
def Tabular_vectorstore_node(state:KnowledgeIntelligenceState):
    db=FAISS.from_documents(
        documents=state['tabular_splitted_data'],
        embedding=state["embedding"]
    )
    retriever=db.as_retriever(search_type="mmr",search_kwargs={"k": 4})
    retrieved_context = retriever.invoke(state["query"])

    return {"tabular_retriever":retrieved_context}

def LLMGeneration(state:KnowledgeIntelligenceState):
    state['llm'] = ChatGroq(model=groq_model_name,temperature=groq_temperature)
    state['prompt'] = PromptTemplate(template="""
                        You are a helpful assistant.
                        Answer the question ONLY using the provided context.
                        Context:
                        {context}
                        {context2}
                        Question:
                        {question}
                    """,input_variables=['question','context','context2']
                    )       

    parser = StrOutputParser()
    chain = state['prompt'] | state['llm'] | parser
    context_text = "\n\n".join(doc.page_content for doc in state["retriever_data"])
    context_text2 = "\n\n".join(doc.page_content for doc in state["tabular_retriever"])

    answer = chain.invoke({
        "context": context_text,
        "context2":context_text2,
        "question": state["query"]
    })

    return {
        "answer": answer
    }

graph = StateGraph(KnowledgeIntelligenceState)

graph.add_node("load", document_loading)
graph.add_node("table_load",Tabular_Document_Loader)
graph.add_node("table_to_text",Tabular_to_text_conversion)
graph.add_node("split", text_splitting)
graph.add_node("table_split",table_splitting)
graph.add_node("emb", doc_embedding)
graph.add_node("vs",vectorstore_node)
graph.add_node("table_vs",Tabular_vectorstore_node)
graph.add_node("LLMGenerate",LLMGeneration)

graph.add_edge(START,"load")
graph.add_edge("load","split")
graph.add_edge("split","emb")
graph.add_edge("emb","vs")
graph.add_edge("vs","LLMGenerate")

graph.add_edge(START,"table_load")
graph.add_edge("table_load","table_to_text")
graph.add_edge("table_to_text","table_split")
graph.add_edge("table_split","emb")
graph.add_edge("emb","table_vs")
graph.add_edge("table_vs","LLMGenerate")

graph.add_edge("LLMGenerate", END)

workflow = graph.compile()

# initial_data={
#     "pdf_file_path": "../Input-Document/Vector_Database.pdf",
#     "query": "Which is the most reliable vector store for cloud? give answer with page number from which you extracted data and it it from table then also row and column name also."
# }

# result = workflow.invoke(initial_data)

# #main answer printing
# print(result['answer'])