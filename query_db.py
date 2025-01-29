# Command line to query Chroma by passing in a query string.

import click
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

@click.command()
@click.option("--query", prompt="Enter query string", help="Query string to search Chroma")
def query_chroma(query: str):
    vector_store = Chroma(
        collection_name="personal_docs",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )
    results = vector_store.similarity_search_with_score(query, k=1)
    for result in results:
        print(result)


if __name__ == "__main__":
    query_chroma()