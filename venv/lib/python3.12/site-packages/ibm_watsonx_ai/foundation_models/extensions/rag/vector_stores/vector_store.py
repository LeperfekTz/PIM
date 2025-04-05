#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations
import contextlib
import copy
import logging
from typing import Any, Literal
from warnings import warn

from langchain_core.documents import Document

from ibm_watsonx_ai.client import APIClient
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.foundation_models.embeddings import BaseEmbeddings
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.base_vector_store import (
    BaseVectorStore,
)
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.langchain_vector_store_adapter import (
    LangChainVectorStoreAdapter,
)
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.vector_store_connector import (
    VectorStoreConnector,
    VectorStoreDataSourceType,
)

from langchain_core.vectorstores import VectorStore as LangChainVectorStore

logger = logging.getLogger(__name__)


class VectorStore(BaseVectorStore):
    """Universal vector store client for a RAG pattern.

    Instantiates the vector store connection in the Watson Machine Learning environment and handles the necessary operations.
    The parameters given by the keyword arguments are used to instantiate the vector store client in their
    particular constructor. Those parameters might be parsed differently.

    For details, refer to the VectorStoreConnector ``get_...`` methods.

    You can utilize the custom embedding function. This function can be provided in the constructor or by the ``set_embeddings`` method.
    For available embeddings, refer to the ``ibm_watsonx_ai.foundation_models.embeddings`` module.

    :param api_client: WatsonX API client required if connecting by connection_id, defaults to None
    :type api_client: APIClient, optional

    :param connection_id: connection asset ID, defaults to None
    :type connection_id: str, optional

    :param embeddings: default embeddings to be used, defaults to None
    :type embeddings: BaseEmbeddings, optional

    :param index_name: name of the vector database index, defaults to None
    :type index_name: str, optional

    :param datasource_type: data source type to use when ``connection_id`` is not provided, keyword arguments will be used to establish connection, defaults to None
    :type datasource_type: VectorStoreDataSourceType, str, optional

    :param distance_metric: metric used for determining vector distance, defaults to None
    :type distance_metric: Literal["euclidean", "cosine"], optional

    :param langchain_vector_store: use LangChain vector store, defaults to None
    :type langchain_vector_store: VectorStore, optional

    **Example:**

    To connect, provide the connection asset ID.
    You can use custom embeddings to add and search documents.

    .. code-block:: python

        from ibm_watsonx_ai import APIClient
        from ibm_watsonx_ai.foundation_models.extensions.rag import VectorStore
        from ibm_watsonx_ai.foundation_models.embeddings import SentenceTransformerEmbeddings

        api_client = APIClient(credentials)

         embedding = Embeddings(
                 model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,
                 api_client=api_client
                 )

        vector_store = VectorStore(
                api_client,
                connection_id='***',
                index_name='my_test_index',
                embeddings=embedding
            )

        vector_store.add_documents([
            {'content': 'document one content', 'metadata':{'url':'ibm.com'}},
            {'content': 'document two content', 'metadata':{'url':'ibm.com'}}
        ])

        vector_store.search('one', k=1)

    .. note::
        Optionally, like in LangChain, it is possible to use a cloud ID and API key parameters to connect to Elastic Cloud.
        The keyword arguments can be used as direct params to LangChain's ``ElasticsearchStore`` constructor.

    .. code-block:: python

        from ibm_watsonx_ai import APIClient
        from ibm_watsonx_ai.foundation_models.extensions.rag import VectorStore

        api_client = APIClient(credentials)

        vector_store = VectorStore(
                api_client,
                index_name='my_test_index',
                model_id=".elser_model_2_linux-x86_64",
                cloud_id='***',
                api_key=IAM_API_KEY
            )

        vector_store.add_documents([
            {'content': 'document one content', 'metadata':{'url':'ibm.com'}},
            {'content': 'document two content', 'metadata':{'url':'ibm.com'}}
        ])

        vector_store.search('one', k=1)


    """

    def __init__(
        self,
        api_client: APIClient | None = None,
        *,
        connection_id: str | None = None,
        embeddings: BaseEmbeddings | None = None,
        index_name: str | None = None,
        datasource_type: VectorStoreDataSourceType | str | None = None,
        distance_metric: Literal["euclidean", "cosine"] | None = None,
        langchain_vector_store: LangChainVectorStore | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self._client = api_client
        if "client" in kwargs and self._client is None:
            client_parameter_deprecated_warning = (
                "Parameter `client` is deprecated. Use `api_client` instead."
            )
            warn(client_parameter_deprecated_warning, category=DeprecationWarning)
            self._client = kwargs.pop("client")

        self._connection_id = connection_id
        self._embeddings = embeddings
        self._index_name = index_name
        self._datasource_type = datasource_type
        self._distance_metric = distance_metric
        self._vector_store: BaseVectorStore
        self._index_properties = kwargs

        if self._connection_id:
            logger.info("Connecting by connection asset.")
            if self._client:
                self._datasource_type, connection_properties = self._connect_by_type(
                    self._connection_id
                )
                logger.info(
                    f"Initializing vector store of type: {self._datasource_type}"
                )
                self._vector_store = VectorStoreConnector(
                    properties={
                        "embeddings": self._embeddings,
                        "index_name": self._index_name,
                        "distance_metric": self._distance_metric,
                        **connection_properties,
                        **self._index_properties,
                    }
                ).get_from_type(
                    self._datasource_type  # type: ignore[arg-type]
                )
                logger.info("Success. Vector store initialized correctly.")
            else:
                raise ValueError(
                    "Client is required if connecting by connection asset."
                )
        elif langchain_vector_store:
            logger.info("Connecting by already established LangChain vector store.")
            if issubclass(type(langchain_vector_store), LangChainVectorStore):
                self._vector_store = LangChainVectorStoreAdapter(langchain_vector_store)
                self._datasource_type = (
                    VectorStoreConnector.get_type_from_langchain_vector_store(
                        langchain_vector_store
                    )
                )
            else:
                raise TypeError("Langchain vector store was of incorrect type.")
        elif self._datasource_type:
            logger.info("Connecting by manually set data source type.")
            self._vector_store = VectorStoreConnector(
                properties={
                    "embeddings": self._embeddings,
                    "index_name": self._index_name,
                    "distance_metric": self._distance_metric,
                    **self._index_properties,
                }
            ).get_from_type(
                self._datasource_type  # type: ignore[arg-type]
            )
        else:
            raise TypeError(
                "To establish connection, please provide 'connection_id', 'langchain_vector_store' or 'datasource_type'."
            )

    def _get_connection_type(self, connection_details: dict[str, list | dict]) -> str:
        """Determine the connection type from the connection details by comparing it to the available list of data source types.

        :param connection_details: dict containing the connection details
        :type connection_details: dict[str, list]

        :raises KeyError: if the connection data source is invalid

        :return: name of the data source
        :rtype: str
        """
        if self._client is not None:
            with contextlib.redirect_stdout(None):
                datasource_types_df = self._client.connections.list_datasource_types()
            datasource_id_to_name_mapping = datasource_types_df.set_index(
                "DATASOURCE_ID"
            )["NAME"].to_dict()
            datasource_type = datasource_id_to_name_mapping.get(
                connection_details["entity"]["datasource_type"], None  # type: ignore[call-overload]
            )

            if datasource_type is None:
                raise WMLClientError("Connection type not found or not supported.")
            else:
                return datasource_type
        else:
            raise WMLClientError(
                "Client is required if connecting by connection asset."
            )

    def _connect_by_type(self, connection_id: str) -> tuple[str, dict]:
        """Get the datasource type and the connection properties from the connection ID.

        :param connection_id: ID of the connection asset
        :type connection_id: str

        :return: string representing the datasource type, connection properties
        :rtype: tuple[str, str]
        """
        if self._client is not None:
            connection_data = self._client.connections.get_details(connection_id)
            datasouce_type = self._get_connection_type(connection_data)
            properties = connection_data["entity"]["properties"]

            logger.info(f"Initializing vector store of type: {datasouce_type}")
            return datasouce_type, properties
        else:
            raise WMLClientError(
                "Client is required if connecting by connection asset."
            )

    def to_dict(self) -> dict:
        """Serialize ``VectorStore`` into a dict that allows reconstruction using the ``from_dict`` class method.

        :return: dict for the from_dict initialization
        :rtype: dict
        """
        return {
            "connection_id": self._connection_id,
            "embeddings": (
                self._embeddings.to_dict()
                if isinstance(self._embeddings, BaseEmbeddings)
                else {}
            ),
            "index_name": self._index_name,
            "datasource_type": (
                str(self._datasource_type) if self._datasource_type else None
            ),
            "distance_metric": self._distance_metric,
            **self._index_properties,
        }

    @classmethod
    def from_dict(
        cls, client: APIClient | None = None, data: dict | None = None
    ) -> VectorStore:
        """Creates ``VectorStore`` using only a primitive data type dict.

        :param data: dict in schema like the ``to_dict()`` method
        :type data: dict

        :return: reconstructed VectorStore
        :rtype: VectorStore
        """
        d = copy.deepcopy(data) if isinstance(data, dict) else {}

        d["embeddings"] = BaseEmbeddings.from_dict(
            data=d.get("embeddings", {}), api_client=client
        )

        return cls(client, **d)

    def get_client(self) -> Any:
        return self._vector_store.get_client()

    def set_embeddings(self, embedding_fn: BaseEmbeddings) -> None:
        setting_embeddings_deprecation_warning = "Setting embeddings after VectorStore initialization may cause issues for `langchain>=0.2.0`"
        warn(setting_embeddings_deprecation_warning, category=DeprecationWarning)
        self._embeddings = embedding_fn
        self._vector_store.set_embeddings(embedding_fn)

    async def add_documents_async(self, content: list[Any], **kwargs: Any) -> list[str]:
        return await self._vector_store.add_documents_async(content, **kwargs)

    def add_documents(self, content: list[Any], **kwargs: Any) -> list[str]:
        return self._vector_store.add_documents(content, **kwargs)

    def search(
        self,
        query: str,
        k: int,
        include_scores: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> list:
        """Searches for documents most similar to the query.

        The method is designed as a wrapper for respective LangChain VectorStores' similarity search methods.
        Therefore, additional search parameters passed in ``kwargs`` should be consistent with those methods,
        and can be found in the LangChain documentation as they may differ depending on the connection
        type: Milvus, Chroma, Elasticsearch, etc.

        :param query: text query
        :type query: str

        :param k: number of documents to retrieve
        :type k: int

        :param include_scores: whether similarity scores of found documents should be returned, defaults to False
        :type include_scores: bool

        :param verbose: whether to display a table with the found documents, defaults to False
        :type verbose: bool

        :return: list of found documents
        :rtype: list
        """
        return self._vector_store.search(
            query, k=k, verbose=verbose, include_scores=include_scores, **kwargs
        )

    def window_search(
        self,
        query: str,
        k: int,
        include_scores: bool = False,
        verbose: bool = False,
        window_size: int = 2,
        **kwargs: Any,
    ) -> list[Document]:
        if isinstance(self._vector_store, LangChainVectorStoreAdapter):
            return self._vector_store.window_search(
                query, k, include_scores, verbose, window_size, **kwargs
            )
        raise NotImplementedError("window_search is not yet implemented in VectorStore")

    def delete(self, ids: list[str], **kwargs: Any) -> None:
        return self._vector_store.delete(ids, **kwargs)

    def clear(self) -> None:
        return self._vector_store.clear()

    def count(self) -> int:
        return self._vector_store.count()

    def as_langchain_retriever(self, **kwargs: Any) -> Any:
        return self._vector_store.as_langchain_retriever(**kwargs)
