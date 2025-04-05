#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import asyncio
from typing import Any

from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.langchain_vector_store_adapter import (
    LangChainVectorStoreAdapter,
)
from ibm_watsonx_ai.wml_client_error import InvalidValue, MissingExtension

from langchain_core.documents import Document

try:
    from langchain_milvus import Milvus
except ImportError:
    raise MissingExtension("langchain_milvus")

try:
    from pymilvus import MilvusException
except ImportError:
    raise MissingExtension("pymilvus")


class MilvusLangchainAdapter(LangChainVectorStoreAdapter):

    def __init__(self, vector_store: Milvus) -> None:
        super().__init__(vector_store)

    def get_client(self) -> Milvus:
        return super().get_client()

    def clear(self) -> None:
        ids = self.get_client().get_pks("pk != ''")
        if ids:
            self.delete(ids)  # type: ignore[arg-type]

    def count(self) -> int:
        ids = self.get_client().get_pks("pk != ''")
        return len(ids) if ids else 0

    def add_documents(
        self, content: list[str] | list[dict] | list, **kwargs: Any
    ) -> list[str]:
        ids, docs = self._process_documents(content)
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        embeddings = self._langchain_vector_store.embedding_func.embed_documents(texts)  # type: ignore[attr-defined]

        return self._fallback_add_documents(
            ids, docs, texts, metadatas, embeddings, **kwargs
        )

    async def add_documents_async(
        self, content: list[str] | list[dict] | list, **kwargs: Any
    ) -> list[str]:
        ids, docs = self._process_documents(content)
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        embeddings = self._langchain_vector_store.embedding_func.embed_documents(texts)  # type: ignore[attr-defined]

        return await asyncio.to_thread(
            self._fallback_add_documents,
            ids,
            docs,
            texts,
            metadatas,
            embeddings,
            **kwargs,
        )

    def _fallback_add_documents(
        self,
        ids: list[str],
        docs: list[Document],
        texts: list[str],
        metadatas: list[Any],
        embeddings: list[Any],
        batch_size: int = 1024,  # default set to 1024
        **kwargs: Any,
    ) -> list[str]:
        if batch_size <= 0:
            raise InvalidValue(
                "batch_size",
                "`batch_size` reached 0 in fallback method for Milvus database. Either documents are too large or `batch_size` was set incorrectly.",
            )
        try:
            return self._upsert(
                ids=ids,
                docs=docs,
                texts=texts,
                metadatas=metadatas,
                embeddings=embeddings,
                batch_size=batch_size,
                **kwargs,
            )
        except MilvusException as e:
            if (
                e.code == 65535
            ):  # handle MilvusException: (code=65535, message=Broker: Message size too large)
                return self._fallback_add_documents(
                    ids=ids,
                    docs=docs,
                    texts=texts,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    batch_size=batch_size // 4,
                    **kwargs,
                )
            else:
                raise e

    def _upsert(
        self,
        ids: list[str],
        docs: list[Document],
        texts: list[str],
        metadatas: list[Any],
        embeddings: list[Any],
        **kwargs: Any,
    ) -> list[str]:
        """Upsert with custom ids.
        Based on Milvus LangChain upsert, but passes ids to add_documents.

        :param ids: list of ids for docs to upsert, defaults to None
        :type ids: list[str]

        :param docs: list of docs, defaults to None
        :type docs: list[Document]

        :return: list of added/upserted ids
        :rtype: list[str]
        """

        if docs is None or len(docs) == 0:
            return []

        if ids is not None and len(ids) and self.get_client().col is not None:
            try:
                self.delete(ids=ids)
            except MilvusException:
                pass

        try:
            return self.get_client().add_embeddings(  # type: ignore[attr-defined]
                ids=ids,
                texts=texts,
                metadatas=metadatas,
                embeddings=embeddings,
                **kwargs,
            )
        except AttributeError:
            return self.get_client().add_documents(ids=ids, documents=docs, **kwargs)
