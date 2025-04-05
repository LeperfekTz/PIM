#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from typing import Any
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.langchain_vector_store_adapter import (
    LangChainVectorStoreAdapter,
)
from ibm_watsonx_ai.wml_client_error import MissingExtension

try:
    from langchain_chroma import Chroma
except ImportError:
    raise MissingExtension("langchain_chroma")


class ChromaLangchainAdapter(LangChainVectorStoreAdapter):

    def __init__(self, vector_store: Chroma) -> None:
        super().__init__(vector_store)

    def get_client(self) -> Chroma:
        return super().get_client()

    def clear(self) -> None:
        client = self.get_client()
        all_docs_ids = client.get()["ids"]
        if len(all_docs_ids) > 0:
            self.delete(all_docs_ids)

    def count(self) -> int:
        client = self.get_client()
        return len(client.get()["ids"])

    def add_documents(
        self, content: list[str] | list[dict] | list, **kwargs: Any
    ) -> list[str]:
        max_batch_size = kwargs.get("max_batch_size")
        if max_batch_size is None:
            try:
                max_batch_size = self._langchain_vector_store._client.get_max_batch_size()  # type: ignore[attr-defined]
            except AttributeError:
                max_batch_size = 10_000

        ids, docs = self._process_documents(content)
        if len(docs) > max_batch_size:
            batch_ids = []

            for batch_start in range(0, len(docs), max_batch_size):
                batch_ids.extend(
                    self._langchain_vector_store.add_documents(
                        docs[batch_start : batch_start + max_batch_size],
                        ids=ids[batch_start : batch_start + max_batch_size],
                        **kwargs,
                    )
                )
            return batch_ids
        else:
            return self._langchain_vector_store.add_documents(docs, ids=ids, **kwargs)
