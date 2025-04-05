#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from enum import Enum
from typing import Any
from ibm_watsonx_ai.foundation_models.extensions.rag.retriever.base_retriever import (
    BaseRetriever,
)
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.base_vector_store import (
    BaseVectorStore,
)
from ibm_watsonx_ai.wml_client_error import MissingExtension

try:
    from langchain_core.documents import Document
except ImportError:
    raise MissingExtension("langchain")


class RetrievalMethod(str, Enum):
    SIMPLE = "simple"
    WINDOW = "window"


class Retriever(BaseRetriever):
    """Retriever class that handles the retrieval operation for a RAG implementation.
    Returns the `number_of_chunks` document segments using the provided `method` based on a relevant query in the ``retrieve`` method.

    :param vector_store: `VectorStore` to use for the retrieval
    :type vector_store: BaseVectorStore

    :param method: default retrieval method to use when calling `retrieve`, defaults to RetrievalMethod.SIMPLE
    :type method: RetrievalMethod, optional

    :param number_of_chunks: number of expected document chunks to be returned, defaults to 5
    :type number_of_chunks: int, optional

    You can create a repeatable retrieval and return the three nearest documents by using a simple proximity search. To do this,
    create a `VectorStore` and then define a `Retriever`.

    .. code-block:: python

        from ibm_watsonx_ai import APIClient
        from ibm_watsonx_ai.foundation_models.extensions.rag import VectorStore
        from ibm_watsonx_ai.foundation_models.extensions.rag import Retriever, RetrievalMethod
        from ibm_watsonx_ai.foundation_models.embeddings import SentenceTransformerEmbeddings

        api_client = APIClient(credentials)

        vector_store = VectorStore(
                api_client,
                connection_id='***',
                params={
                    'index_name': 'my_test_index',
                },
                embeddings=SentenceTransformerEmbeddings('sentence-transformers/all-MiniLM-L6-v2')
            )

        retriever = Retriever(vector_store=vector_store, method=RetrievalMethod.SIMPLE, number_of_chunks=3)

        retriever.retrieve("What is IBM known for?")
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        method: RetrievalMethod = RetrievalMethod.SIMPLE,
        window_size: int = 2,
        number_of_chunks: int = 5,
    ) -> None:
        super().__init__(vector_store)

        if isinstance(method, str):
            try:
                self.method = RetrievalMethod(method)
            except ValueError:
                raise ValueError(f"'{method}' is not a valid retrieval method value.")
        else:
            raise ValueError(
                "Retrieval method '{}' is not supported. Use one of {}".format(
                    self.method, (method for method in RetrievalMethod)
                )
            )

        self.window_size = window_size
        self.number_of_chunks = number_of_chunks

    def retrieve(self, query: str, **kwargs: Any) -> list[Document]:
        """Retrieve elements from the `VectorStore` by using the provided `query`.

        :param query: text query to be used for searching
        :type query: str

        :return: list of retrieved LangChain documents
        :rtype: list[langchain_core.documents.Document]
        """
        if self.method == RetrievalMethod.SIMPLE:
            return self.vector_store.search(query, k=self.number_of_chunks, **kwargs)
        elif self.method == RetrievalMethod.WINDOW:
            return self.vector_store.window_search(
                query,
                k=self.number_of_chunks,
                window_size=self.window_size,
                **kwargs,
            )
        else:
            raise ValueError(
                "Retrieval method '{}' is not supported. Use one of {}".format(
                    self.method, (method for method in RetrievalMethod)
                )
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "number_of_chunks": self.number_of_chunks,
        }

    @classmethod
    def from_vector_store(
        cls,
        vector_store: BaseVectorStore,
        init_parameters: dict[str, Any] | None = None,
    ) -> "Retriever":
        return cls(vector_store, **(init_parameters or {}))
