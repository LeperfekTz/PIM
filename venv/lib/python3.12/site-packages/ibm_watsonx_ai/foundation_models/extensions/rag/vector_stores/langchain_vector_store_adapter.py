#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from collections import defaultdict
from ibm_watsonx_ai.wml_client_error import MissingExtension, MissingMetadata
from ibm_watsonx_ai.wml_resource import WMLResource

try:
    from langchain_core.vectorstores import VectorStore as LangChainVectorStore
    from langchain_core.vectorstores import (
        VectorStoreRetriever as LangChainVectorStoreRetriever,
    )
    from langchain_core.documents import Document
except ImportError:
    raise MissingExtension("langchain")

import hashlib
from typing import Any, cast
import logging

from ibm_watsonx_ai.foundation_models.embeddings import BaseEmbeddings
from ibm_watsonx_ai.foundation_models.extensions.rag.utils.utils import verbose_search
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.base_vector_store import (
    BaseVectorStore,
)

logger = logging.getLogger(__name__)


def merge_window_into_a_document(window: list[Document]) -> Document:
    """
    Merges a list of chunks into a single document.
    If consecutive chunks have intersecting merged_text, the merged_text is merged to avoid duplications.

    :param window: ordered list of documents for merging
    :type window: list[langchain_core.documents.Document]

    :return: document that contains the merged merged_text of the window documents
    :rtype: langchain_core.documents.Document
    """

    def merge_metadata(metadatas: list[dict]) -> dict:
        """
        Merges a list of dictionaries (metadata) into one metadata
        The keys remain the same but the values are changed into lists of values from all metadata

        :param metadatas: list of metadata dictionaries to be merged
        :type metadatas: list[dict]

        :return: single merged dictionary (metadata) with unique values in lists for each key
        :rtype: Tuple[str, bool]
        """
        if len(metadatas) == 1:
            return metadatas[0]

        merged_metadata = defaultdict(set)
        for metadata in metadatas:
            for key, value in metadata.items():
                if isinstance(value, list):
                    merged_metadata[key].update(value)
                else:
                    merged_metadata[key].add(value)

        result = {}
        for key, value_set in merged_metadata.items():
            value_list = sorted(value_set)
            if len(value_list) == 1:
                result[key] = value_list[0]
            else:
                result[key] = value_list
        return result

    def get_str2_without_intersecting_text(str1: str, str2: str) -> tuple[str, bool]:
        """
        Finds the intersecting merged_text between the suffix of str1 and the prefix of str2.

        :param str1: the first string
        :type str1: str

        :param str2: the second string
        :type str2: str

        :return: a tuple of:
            1. str2 without its intersection to str1
            2. whether there was an intersection or not
        :rtype: Tuple[str, bool]
        """
        # Start checking from the longest possible overlap to the shortest
        for i in range(min(len(str1), len(str2)), 0, -1):
            if str1[-i:] == str2[:i]:
                return str2[i:], True
        return str2, False

    def merge_texts(texts: list[str]) -> str:
        """
        Merges a list of texts into a single text string.
        If consecutive texts have intersecting parts, the text is merged to avoid duplications.

        :param texts: ordered list of text strings to be merged
        :type texts: List[str]

        :return: single string that contains the merged text
        :rtype: str
        """
        merged_text = ""
        for text in texts:
            text_to_add, has_intersection = get_str2_without_intersecting_text(
                merged_text, text
            )
            if merged_text and not has_intersection:
                merged_text += " "  # Add a space between non-overlapping texts (chunks)
            merged_text += text_to_add
        return merged_text

    texts = [doc.page_content for doc in window]
    merged_text = merge_texts(texts)

    metadata = [doc.metadata for doc in window]
    merged_metadata = merge_metadata(metadata)

    return Document(page_content=merged_text, metadata=merged_metadata)


class LangChainVectorStoreAdapter(BaseVectorStore):
    """
    Adapter for LangChain ``VectorStore`` base class.

    :param vector_store: concrete LangChain vector store object
    :type vector_store: langchain_core.vectorstore.VectorStore
    """

    def __init__(self, vector_store: LangChainVectorStore) -> None:
        super().__init__()
        self._langchain_vector_store: LangChainVectorStore = vector_store

    def get_client(self) -> Any:
        return self._langchain_vector_store

    def set_embeddings(self, embedding_fn: BaseEmbeddings) -> None:
        if hasattr(self._langchain_vector_store, "embedding"):
            self._langchain_vector_store.embedding = embedding_fn
        elif hasattr(self._langchain_vector_store, "_embedding"):
            self._langchain_vector_store._embedding = embedding_fn
        elif hasattr(self._langchain_vector_store, "_embedding_function"):
            self._langchain_vector_store._embedding_function = embedding_fn
        elif hasattr(self._langchain_vector_store, "embedding_function"):
            self._langchain_vector_store.embedding_function = embedding_fn
        else:
            raise AttributeError(
                "Could not set an embedding function for this vector store."
            )

    def add_documents(
        self, content: list[str] | list[dict] | list, **kwargs: Any
    ) -> list[str]:
        ids, docs = self._process_documents(content)
        return self._langchain_vector_store.add_documents(docs, ids=ids, **kwargs)

    async def add_documents_async(
        self, content: list[str] | list[dict] | list, **kwargs: Any
    ) -> list[str]:
        ids, docs = self._process_documents(content)
        return await self._langchain_vector_store.aadd_documents(
            docs, ids=ids, **kwargs
        )

    def search(
        self,
        query: str,
        k: int,
        include_scores: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> list[Document] | list[tuple[Document, float]]:
        result: list[Document] | list[tuple[Document, float]]
        if include_scores:
            result = self._langchain_vector_store.similarity_search_with_score(
                query, k=k, **kwargs
            )
        else:
            result = self._langchain_vector_store.similarity_search(
                query, k=k, **kwargs
            )

        if verbose:
            verbose_search(query, result)
        return result

    def window_search(
        self,
        query: str,
        k: int,
        include_scores: bool = False,
        verbose: bool = False,
        window_size: int = 2,
        **kwargs: Any,
    ) -> list:

        documents = self.search(query, k, include_scores, verbose, **kwargs)
        if window_size <= 0:
            return documents

        if not include_scores:
            documents = cast(list[Document], documents)
            return [
                self._window_extend_and_merge(document, window_size)
                for document in documents
            ]
        else:
            documents_and_scores = cast(list[tuple[Document, float]], documents)
            documents = [t[0] for t in documents_and_scores]
            scores = [t[1] for t in documents_and_scores]
            extended_documents = [
                self._window_extend_and_merge(document, window_size)
                for document in documents
            ]
            return list(zip(extended_documents, scores))

    def _get_window_documents_chroma(
        self, doc_id: str, seq_nums_window: list[int]
    ) -> list[Document]:
        """
        Receives a document ID and a list of chunks' sequence_numbers,
        and searches the vector store according to the metadata.

        :param doc_id: ID of document
        :type doc_id: str

        :param seq_nums_window: list of sequence numbers
        :type seq_nums_window: list[int]

        :return: list of documents from that document with these sequence_numbers
        :rtype: list[Document]
        """
        expr = {
            "$and": [
                {"document_id": {"$eq": doc_id}},
                {"sequence_number": {"$gte": seq_nums_window[0]}},
                {"sequence_number": {"$lte": seq_nums_window[-1]}},
            ]
        }
        res = self._langchain_vector_store.get(where=expr)  # type: ignore[attr-defined]
        texts, metadatas = res["documents"], res["metadatas"]
        window_documents = [
            Document(
                page_content=text,
                metadata={
                    "sequence_number": metadata["sequence_number"],
                    "document_id": doc_id,
                },
            )
            for text, metadata in zip(texts, metadatas)
        ]
        return window_documents

    def _get_window_documents_milvus(
        self, doc_id: str, seq_nums_window: list[int]
    ) -> list[Document]:
        """
        Receives a document ID and a list of chunks' sequence_numbers,
        and searches the vector store according to the metadata.

        :param doc_id: ID of document
        :type doc_id: str

        :param seq_nums_window: list of sequence numbers
        :type seq_nums_window: list[int]

        :return: list of documents from that document with these sequence_numbers
        :rtype: list[Document]
        """
        expr = f"document_id LIKE '{doc_id}' && sequence_number in {seq_nums_window}"
        docs = self._langchain_vector_store.col.query(  # type: ignore[attr-defined]
            expr=expr,
            output_fields=["sequence_number", "text"],
            limit=len(seq_nums_window),
        )
        window_documents = [
            Document(
                page_content=doc["text"],
                metadata={
                    "sequence_number": doc["sequence_number"],
                    "document_id": doc_id,
                },
            )
            for doc in docs
        ]
        return window_documents

    def _get_window_documents_elasticsearch(
        self, doc_id: str, seq_nums_window: list[int]
    ) -> list[Document]:
        """
        Receives a document ID and a list of chunks' sequence_numbers,
        and searches the vector store according to the metadata.

        :param doc_id: ID of document
        :type doc_id: str

        :param seq_nums_window: list of sequence numbers
        :type seq_nums_window: list[int]

        :return: list of documents from that document with these sequence_numbers
        :rtype: list[Document]
        """

        query = {
            "bool": {
                "must": [{"term": {"metadata.document_id.keyword": doc_id}}],
                "filter": [
                    {
                        "range": {
                            "metadata.sequence_number": {
                                "gte": seq_nums_window[0],
                                "lte": seq_nums_window[-1],
                            }
                        }
                    }
                ],
            }
        }

        response = self._langchain_vector_store.search(
            query="*:*", search_type="similarity", filter=query, k=len(seq_nums_window)
        )

        window_documents = [
            Document(
                page_content=doc.page_content,
                metadata={
                    "sequence_number": doc.metadata["sequence_number"],
                    "document_id": doc.metadata["document_id"],
                },
            )
            for doc in response
        ]

        return window_documents

    def delete(self, ids: list[str], **kwargs: Any) -> None:
        self._langchain_vector_store.delete(ids, **kwargs)

    def clear(self) -> None:
        raise NotImplementedError(
            "Use concrete wrapper if you need to use this functionality."
        )

    def count(self) -> int:
        raise NotImplementedError(
            "Use concrete wrapper if you need to use this functionality."
        )

    def as_langchain_retriever(self, **kwargs: Any) -> Any:
        return LangChainVectorStoreRetriever(
            vectorstore=self._langchain_vector_store, **kwargs
        )

    def _process_documents(
        self, content: list[str] | list[dict] | list
    ) -> tuple[list[str], list[Document]]:
        """Processes arbitrary list of data to produce two lists: one with unique IDs, and one with LangChain documents.

        Handles duplicate documents.

        :param content: arbitrary data
        :type content: list[str] | list[dict] | list

        :return: lists with IDs and docs
        :rtype: tuple[list[str], list[langchain_core.documents.Document]
        """
        WMLResource._validate_type(content, "content", list)
        docs = self._as_langchain_documents(content)
        if docs:
            # Take only unique ID document. Get two lists, one with ids, one with documents
            return tuple(
                map(  # type: ignore[return-value]
                    list,
                    zip(
                        *{
                            hashlib.sha256(str(doc).encode()).hexdigest(): doc
                            for doc in docs
                        }.items()
                    ),
                )
            )
        else:
            return [], []

    def _as_langchain_documents(
        self, content: list[str] | list[dict] | list
    ) -> list[Document]:
        """Creates a LangChain ``Document`` list from a list of potentially unstructured data.

        :param content: list of unstructured data to be parsed
        :type content: list[str] | list[dict] | list

        :raises AttributeError: raised when data does not fit the required schema
        :return: list of LangChain Documents
        :rtype: list[langchain_core.documents.Document]
        """
        result = []
        for doc in content:
            if isinstance(doc, str):
                result.append(Document(page_content=doc))
            elif isinstance(doc, dict):
                content_str: str | None = doc.get("content", None)
                metadata = doc.get("metadata", {})

                if content_str:
                    if isinstance(metadata, dict):
                        result.append(
                            Document(page_content=content_str, metadata=metadata)
                        )
                    else:
                        logger.warning(
                            f"Document: {doc} is incorrect. Metadata needs to be given with 'metadata' attribute and it needs to be a serializable dict. Skipping."
                        )
                        continue
                else:
                    logger.warning(
                        f"Document: {doc} is incorrect. Field 'content' is required"
                    )
                    continue
            else:
                try:
                    result.append(
                        Document(page_content=doc.page_content, metadata=doc.metadata)
                    )
                except AttributeError:
                    logger.warning(
                        f"Document: {doc} is not a dict, nor string, nor LangChain Document-like object. Skipping."
                    )

        return result

    def _window_extend_and_merge(
        self, document: Document, window_size: int
    ) -> Document:
        """
        Extends a document (a chunk) to its adjacent chunks (if they exist) from the same origin document.
        Then merges the adjacent chunks into one chunk while keeping their order,
        and merges intersecting text between them (if it exists).
        This requires chunks to have "document_id" and "sequence_number" in their metadata.

        :param document: document (chunk) to be extended to its window and merged
        :type document: Document

        :param window_size: number of adjacent chunks to retrieve before and after the center, according to the sequence_number
        :type window_size: int

        :return: merged window
        :rtype: Document
        """
        if "document_id" not in document.metadata:
            raise MissingMetadata('document must have "document_id" in its metadata')
        if "sequence_number" not in document.metadata:
            raise MissingMetadata(
                'document must have "sequence_number" in its metadata'
            )
        doc_id = document.metadata["document_id"]
        seq_num = document.metadata["sequence_number"]
        seq_nums_window = [seq_num + i for i in range(-window_size, window_size + 1, 1)]

        vs_type = self._langchain_vector_store.__class__.__name__

        match vs_type:
            case "Milvus":
                try:
                    from langchain_milvus.vectorstores import Milvus
                except ImportError:
                    raise MissingExtension("langchain_milvus")

                window_documents = self._get_window_documents_milvus(
                    doc_id, seq_nums_window
                )
            case "Chroma":
                try:
                    from langchain_chroma import Chroma
                except ImportError:
                    raise MissingExtension("langchain_chroma")

                window_documents = self._get_window_documents_chroma(
                    doc_id, seq_nums_window
                )
            case "ElasticsearchStore":
                try:
                    from langchain_elasticsearch.vectorstores import ElasticsearchStore
                except ImportError:
                    raise MissingExtension("langchain_elasticsearch")
                window_documents = self._get_window_documents_elasticsearch(
                    doc_id, seq_nums_window
                )
            case _:
                raise TypeError(
                    f"Currently we only support Milvus, Chroma and Elasticsearch. "
                    f"Received {type(self._langchain_vector_store)}."
                )

        window_documents.sort(key=lambda x: x.metadata["sequence_number"])
        return merge_window_into_a_document(window_documents)
