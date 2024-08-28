from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from typing import Any, Dict, List

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from consts import INDEX_NAME

# Load environment variables
load_dotenv()


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create Pinecone vector store using the INDEX_NAME from consts.py
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
    # Initialize ChatOpenAI model
    chat = ChatOpenAI(verbose=True, temperature=0)

    # Load prompts from Langchain hub
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Create a chain for combining documents
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    # Create a history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    # Create a retrieval chain
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    # Invoke the chain with the query and chat history
    result = qa.invoke(input={"input": query, "chat_history": chat_history})

    # Format the result
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }
    return new_result


def format_docs(docs):
    """Format a list of documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def run_llm2(query: str, chat_history: List[Dict[str, Any]] = []):
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create Pinecone vector store using the INDEX_NAME from consts.py
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
    # Initialize ChatOpenAI model (note: 'gpt-4o' might be a typo, should be 'gpt-4' or another valid model name)
    chat = ChatOpenAI(model_name="gpt-4o", verbose=True, temperature=0)

    # Load prompts from Langchain hub
    # rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Create a RAG (Retrieval-Augmented Generation) chain
    rag_chain = (
        {
            "context": docsearch.as_retriever() | format_docs,
            "input": RunnablePassthrough(),
        }
        | retrieval_qa_chat_prompt
        | chat
        | StrOutputParser()
    )

    # Create a chain to retrieve documents
    retrieve_docs_chain = (lambda x: x["input"]) | docsearch.as_retriever()

    # Combine the retrieval and RAG chains
    chain = RunnablePassthrough.assign(context=retrieve_docs_chain).assign(
        answer=rag_chain
    )

    # Invoke the combined chain with the query and chat history
    result = chain.invoke({"input": query, "chat_history": chat_history})
    return result


if __name__ == "__main__":
    # Test the run_llm function
    res = run_llm(query="What is the Models API ?")
    print(res["result"])
