#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.langchain_vector_store_adapter import (
    LangChainVectorStoreAdapter,
)
from ibm_watsonx_ai.wml_client_error import MissingExtension

try:
    from langchain_elasticsearch.vectorstores import ElasticsearchStore
    import elasticsearch
except ImportError:
    raise MissingExtension("langchain_elasticsearch")


class ElasticsearchLangchainAdapter(LangChainVectorStoreAdapter):

    def __init__(self, vector_store: ElasticsearchStore) -> None:
        super().__init__(vector_store)

    def get_client(self) -> ElasticsearchStore:
        return super().get_client()

    def clear(self) -> None:
        es_vs = self.get_client()._store
        es = self.get_client().client
        try:
            es.delete_by_query(
                index=es_vs.index, body={"query": {"match_all": {}}}, refresh=True
            )
        except elasticsearch.NotFoundError:
            pass

    def count(self) -> int:
        es = self.get_client().client
        return es.count(index=self.get_client()._store.index)["count"]
