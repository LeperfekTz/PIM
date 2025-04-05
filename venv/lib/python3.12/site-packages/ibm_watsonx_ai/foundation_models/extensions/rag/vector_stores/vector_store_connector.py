#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import copy
from enum import Enum

import logging
from typing import Any

from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.base_vector_store import (
    BaseVectorStore,
)
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.langchain_vector_store_adapter import (
    LangChainVectorStoreAdapter,
)

from ibm_watsonx_ai.foundation_models.extensions.rag.utils.utils import (
    save_ssl_certificate_as_file,
)

from langchain_core.vectorstores import VectorStore as LangChainVectorStore
from ibm_watsonx_ai.wml_client_error import MissingExtension

logger = logging.getLogger(__name__)


class VectorStoreDataSourceType(str, Enum):
    ELASTICSEARCH = "elasticsearch"
    CHROMA = "chroma"
    MILVUS = "milvus"
    MILVUS_WXD = "milvuswxd"  # IBM watsonx.data Milvus
    UNDEFINED = "undefined"

    def __str__(self) -> str:
        return self.value


class VectorStoreConnector:
    """Creates a proper vector store client using the provided properties.

    Properties are arguments to the LangChain vector stores of a desired type.
    Also parses properties extracted from connection assets into one that would fit for initialization.

    Custom or connection asset properties that are parsed are:
    * `index_name`
    * `distance_metric`
    * `username`
    * `password`
    * `ssl_certificate`
    * `embeddings`

    :param properties: dictionary with all the required key values to establish the connection
    :type properties: dict
    """

    def __init__(self, properties: dict | None = None) -> None:
        def deepcopy_if_possible(obj: Any) -> Any:
            try:
                return copy.deepcopy(obj)
            except Exception:
                return obj

        self.properties: dict = (
            {key: deepcopy_if_possible(value) for key, value in properties.items()}
            if isinstance(properties, dict)
            else {}
        )

    @staticmethod
    def get_type_from_langchain_vector_store(
        langchain_vector_store: Any,
    ) -> VectorStoreDataSourceType:
        """Returns ``DataSourceType`` for concrete LangChain ``VectorStore`` class.

        :param langchain_vector_store: vector store object from LangChain
        :type langchain_vector_store: Any

        :return: DataSourceType name
        :rtype: VectorStoreDataSourceType
        """
        vs_type = langchain_vector_store.__class__.__name__

        match vs_type:
            case "ElasticsearchStore":
                return VectorStoreDataSourceType.ELASTICSEARCH
            case "Chroma":
                return VectorStoreDataSourceType.CHROMA
            case "Milvus":
                return VectorStoreDataSourceType.MILVUS
            case _:
                return VectorStoreDataSourceType.UNDEFINED

    def get_from_type(self, type: VectorStoreDataSourceType) -> BaseVectorStore:
        """Gets a vector store based on the provided type (matching from DataSource names from SDK API).

        :param type: DataSource type string from SDK API
        :type type: VectorStoreDataSourceType

        :raises TypeError: unsupported type
        :return: proper BaseVectorStore type constructed from properties
        :rtype: BaseVectorStore
        """
        match type:
            case VectorStoreDataSourceType.ELASTICSEARCH:
                return self.get_elasticsearch()
            case VectorStoreDataSourceType.CHROMA:
                return self.get_chroma()
            case (
                VectorStoreDataSourceType.MILVUS | VectorStoreDataSourceType.MILVUS_WXD
            ):
                return self.get_milvus()
            case _:
                raise TypeError("Data source type not supported.")

    def get_langchain_adapter(  # type: ignore[return]
        self, langchain_vector_store: Any
    ) -> LangChainVectorStoreAdapter | None:
        """Creates an adapter for a concrete vector store from LangChain.

        :param langchain_vector_store: object that is a subclass of the LangChain vector store
        :type langchain_vector_store: Any

        :raises ImportError: LangChain required
        :return: proper adapter for the vector store
        :rtype: LangChainVectorStoreAdapter
        """

        if isinstance(langchain_vector_store, LangChainVectorStore):
            return LangChainVectorStoreAdapter(vector_store=langchain_vector_store)

    def get_chroma(self) -> LangChainVectorStoreAdapter:
        """Creates an in-memory vector store for Chroma.

        :raises ImportError: langchain required
        :return: vector store adapter for LangChain's Chroma
        :rtype: LangChainVectorStoreAdapter
        """
        try:
            from langchain_chroma import Chroma
            from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.adapters.chroma_adapter import (
                ChromaLangchainAdapter,
            )
        except ImportError:
            raise MissingExtension("langchain_chroma")

        parsed_params = self.properties
        parsed_params.pop("datasource_type", None)

        # Parse collection name
        # 'collection_name' kwargs for Chroma has priority over generic 'index_name'
        collection_name = parsed_params.pop("index_name", None)
        if collection_name:
            parsed_params["collection_name"] = parsed_params.get(
                "collection_name", collection_name
            )

        # Parse distance metric - set it in collection_metadata
        # Distance metric for Chroma is determined by collection metadata
        # See: Chroma._select_relevance_score_fn()
        distance_metric = parsed_params.pop("distance_metric", None)
        if distance_metric == "euclidean":
            collection_metadata = {"hnsw:space": "l2"}
        elif distance_metric == "cosine":
            collection_metadata = {"hnsw:space": "cosine"}
        else:
            collection_metadata = None

        parsed_params["collection_metadata"] = parsed_params.get(
            "collection_metadata", collection_metadata
        )

        # Set embedding from params
        parsed_params["embedding_function"] = parsed_params.pop("embeddings", None)

        if parsed_params["embedding_function"] is None:
            raise ValueError("Embedding function is required for Chroma.")

        return ChromaLangchainAdapter(Chroma(**parsed_params))

    def get_milvus(self) -> LangChainVectorStoreAdapter:
        """Creates a Milvus vector store.

        :raises ImportError: langchain required
        :return: vector store adapter for LangChain's Milvus
        :rtype: LangChainVectorStoreAdapter
        """
        try:
            from langchain_milvus import Milvus
            from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.adapters.milvus_adapter import (
                MilvusLangchainAdapter,
            )
        except ImportError:
            raise MissingExtension("langchain_milvus")

        parsed_params = self.properties
        parsed_params.pop("datasource_type", None)

        # Connection 'index_name' is 'collection_name' in Milvus
        if parsed_params.get("index_name"):
            parsed_params["collection_name"] = parsed_params.pop("index_name")
        elif not parsed_params.get("collection_name"):
            raise ValueError("Provide 'index_name' or 'collection_name'.")

        # Parse distance metric
        # Distance metric is set in `index_params`.
        # Here we replace the default `index_params` with different metric type.
        # See: `Milvus._create_index()`.
        distance_metric = parsed_params.pop("distance_metric", None)
        if distance_metric == "cosine":
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64},
            }
        else:
            index_params = None

        parsed_params["index_params"] = parsed_params.get("index_params", index_params)

        # Prepare connection_args (if not present)
        if "connection_args" not in parsed_params:
            parsed_params["connection_args"] = {}

        # Set secure=True also when user set it in Connection UI
        if "ssl" in parsed_params:
            is_ssl = parsed_params.pop("ssl")
            parsed_params["secure"] = True if is_ssl == "true" else False

        # Get SSL certificate saved to file
        if "ssl_certificate" in parsed_params:
            parsed_params["server_pem_path"] = save_ssl_certificate_as_file(
                parsed_params.pop("ssl_certificate")
            )

        # Connection 'username' is 'user' in Milvus
        if "username" in parsed_params:
            parsed_params["user"] = parsed_params.pop("username")

        # Connection 'database' is 'db_name' in Milvus
        if "database" in parsed_params:
            parsed_params["db_name"] = parsed_params.pop("database")

        # Move each param that was in parsed_params to connection_args if we expect it here
        for param in [
            "uri",
            "host",
            "port",
            "user",
            "password",
            "db_name",
            "secure",
            "client_key_path",
            "client_pem_path",
            "ca_pem_path",
            "server_pem_path",
            "server_name",
        ]:
            if param in parsed_params.keys():
                parsed_params["connection_args"][param] = parsed_params.pop(param)

        parsed_params["embedding_function"] = parsed_params.pop("embeddings", None)

        def build_address(connection_args: dict) -> str:
            """Build an address string from host and port."""
            host = connection_args.get("host", "localhost")
            port = connection_args.get("port", 19530)
            return f"{host}:{port}"

        parsed_params["connection_args"]["address"] = build_address(
            parsed_params["connection_args"]
        )

        return MilvusLangchainAdapter(Milvus(**parsed_params))

    def get_elasticsearch(self) -> LangChainVectorStoreAdapter:
        """Creates an Elasticsearch vector store.

        :raises ImportError: langchain required
        :return: vector store adapter for LangChain's Elasticsearch
        :rtype: LangChainVectorStoreAdapter
        """
        try:
            from langchain_elasticsearch import (
                ElasticsearchStore,
                SparseVectorStrategy,
                DenseVectorScriptScoreStrategy,
                RetrievalStrategy,
                DistanceMetric,
            )
            from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.adapters.es_adapter import (
                ElasticsearchLangchainAdapter,
            )
        except ImportError:
            raise MissingExtension("langchain_elasticsearch")

        parsed_params = self.properties
        parsed_params.pop("datasource_type", None)

        # Always use empty es_params if not provided
        parsed_params["es_params"] = self.properties.pop("es_params", {})

        # Drop unnecessary stuff from connection asset if they are present
        parsed_params.pop("auth_method", None)
        parsed_params.pop("use_anonymous_access", None)

        # Parse ES connection data - select proper connection type
        # Connecting by 'url': username/password or api_key
        if "url" in parsed_params:
            # Get URL of ES instance
            parsed_params["es_url"] = parsed_params.pop("url")

            # Detect credentials given in connection asset
            if "username" in parsed_params and "password" in parsed_params:
                # Connect by username and password extracted from connection
                parsed_params["es_user"] = parsed_params.pop("username")
                parsed_params["es_password"] = parsed_params.pop("password")
                parsed_params.pop("api_key", None)
            elif "api_key" in parsed_params:
                # Connect by api key
                parsed_params["es_api_key"] = parsed_params.pop("api_key")

                parsed_params.pop("username", None)
                parsed_params.pop("password", None)
            else:
                raise ValueError(
                    """To connect to given hostname ['url'] provide
                                either ['username', 'password'] or ['api_key'].
                                Make sure those fields are present in connection details or parameters given
                                upon VectorStore initialization. """
                )
        elif "es_url" in parsed_params:
            if "es_user" in parsed_params and "es_password" in parsed_params:
                pass
            elif "es_api_key" in parsed_params:
                pass
            else:
                raise ValueError(
                    """To connect to given hostname ['es_url'] provide
                                either ['es_user', 'es_password'] or ['es_api_key'].
                                Make sure those fields are present in parameters given
                                upon VectorStore initialization. """
                )
        # Connecting by '(es_)cloud_id' to Elasticsearch cloud
        elif "cloud_id" in parsed_params and "api_key" in parsed_params:
            parsed_params["es_cloud_id"] = parsed_params.pop("cloud_id", None)
            parsed_params["es_api_key"] = parsed_params.pop("api_key", None)
        elif "es_cloud_id" in parsed_params and "es_api_key" in parsed_params:
            pass
        else:
            raise ValueError(
                """Connection data was not sufficent. Either provide:
                             - ['url', 'username', 'password'],
                             - ['url', 'api_key'],
                             - ['cloud_id', 'api_key']
                             or
                             - ['es_url', 'es_user', 'es_password'],
                             - ['es_url', 'es_api_key'],
                             - ['es_cloud_id', 'es_api_key'],
                             in your connection asset or in params for VectorStore."""
            )

        if not parsed_params.get("index_name"):
            raise ValueError("Provide 'index_name'.")

        # Parse SSL certificate
        ssl_certificate_content = parsed_params.pop("ssl_certificate", None)

        if ssl_certificate_content:
            parsed_params["es_params"]["ca_certs"] = save_ssl_certificate_as_file(
                ssl_certificate_content
            )

        # Parse distance metric
        # Match with ES DistanceMetric.
        # Default is cosine.
        distance_metric = parsed_params.pop("distance_metric", None)
        if distance_metric == "euclidean":
            distance_metric = DistanceMetric.EUCLIDEAN_DISTANCE
        elif distance_metric == "cosine":
            distance_metric = DistanceMetric.COSINE
        else:
            distance_metric = DistanceMetric.COSINE

        parsed_params["distance_strategy"] = parsed_params.pop(
            "distance_strategy", distance_metric
        )

        # Determine retrieval strategy type from parameters
        if "strategy" not in parsed_params or not isinstance(
            parsed_params["strategy"], RetrievalStrategy
        ):
            if "model_id" in parsed_params:
                parsed_params["strategy"] = SparseVectorStrategy(
                    model_id=parsed_params.pop("model_id")
                )
            else:
                parsed_params["strategy"] = DenseVectorScriptScoreStrategy(
                    distance=distance_metric,
                )

        # Set embedding from params
        parsed_params["embedding"] = parsed_params.pop("embeddings", None)

        return ElasticsearchLangchainAdapter(ElasticsearchStore(**parsed_params))
