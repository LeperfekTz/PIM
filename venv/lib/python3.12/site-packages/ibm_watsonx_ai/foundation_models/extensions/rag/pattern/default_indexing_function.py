#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


def default_indexing_function(docs=None, params=None):
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models.extensions.rag.chunker.langchain_chunker import (
        LangChainChunker,
    )
    from ibm_watsonx_ai.foundation_models.extensions.rag import VectorStore

    client = APIClient(
        Credentials.from_dict(params["credentials"]),
        space_id=params["space_id"],
        project_id=params["project_id"],
    )

    vector_store = VectorStore.from_dict(client=client, data=params["vector_store"])
    chunker = LangChainChunker.from_dict(params["chunker"])

    chunked_docs = chunker.split_documents(docs)
    return vector_store.add_documents(chunked_docs)
