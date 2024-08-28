import time

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from consts import INDEX_NAME

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

'''
To run the ingestion script, follow these steps:
Update the `INDEX_NAME` in the `consts.py` file to match the name of the Pinecone index you want to create.
Update the `langchain_documents_base_urls` list in the `ingestion.py` file to include the URLs of the documentation 
pages you want to include in the index.

Create a Pinecone index:
   Before running the ingestion script, you need to create an index in Pinecone through their web interface:
   - Log in to your Pinecone account at https://app.pinecone.io/
   - Click on "Create Index"
   - Set the index name to match the `INDEX_NAME` in your `consts.py` file
   - Choose the appropriate environment and settings for your project
   - Click "Create Index" to finalize

Run the ingestion script to populate the Pinecone index:
   ```
   python ingestion.py
   ```
   This script will populate the Pinecone index with the required data.
'''


def ingest_docs2() -> None:
    from langchain_community.document_loaders import FireCrawlLoader

    langchain_documents_base_urls = [
        "https://developer.salesforce.com/docs/einstein/genai/guide/access-models-api-with-apex.html",
        "https://developer.salesforce.com/docs/einstein/genai/guide/access-models-api-with-rest.html",
        "https://developer.salesforce.com/docs/einstein/genai/guide/build-lwc-flow-models.html",
        "https://developer.salesforce.com/docs/einstein/genai/guide/rate-limits.html",
        "https://developer.salesforce.com/docs/einstein/genai/guide/supported-models.html",
        "https://developer.salesforce.com/docs/einstein/genai/guide/feedback.html",
        "https://developer.salesforce.com/docs/einstein/genai/guide/specify-languages-and-locales.html",
        "https://developer.salesforce.com/docs/einstein/genai/guide/data-masking.html",
        "https://developer.salesforce.com/docs/einstein/genai/guide/toxicity-scoring.html",
    ]
    langchain_documents_base_urls2 = [langchain_documents_base_urls[0]]   # test one
    for url in langchain_documents_base_urls:
        print(f"FireCrawling {url=}")
        loader = FireCrawlLoader(
            url=url,
            mode="crawl",
            params={
                "crawlerOptions": {"limit": 5},
                "pageOptions": {"onlyMainContent": True},
                "wait_until_done": True,
            },
        )
        docs = loader.load()

        print(f"Going to add {len(docs)} documents to Pinecone")
        # Add documents to Pinecone vector store
        PineconeVectorStore.from_documents(
            docs, embeddings, index_name=INDEX_NAME
        )
        print(f"****Loading {url}* to vectorstore done ***")
        # Sleep a minute to avoid rate-limits
        time.sleep(60)


if __name__ == "__main__":
    ingest_docs2()
