#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from typing import Literal, Sequence, Any, Iterable

from langchain.text_splitter import TextSplitter
from langchain_core.documents import Document

from .base_chunker import BaseChunker


__all__ = [
    "LangChainChunker",
]


class LangChainChunker(BaseChunker[Document]):
    """
    Wrapper for LangChain TextSplitter.

    :param method: describes the type of TextSplitter as the main instance performing the chunking, defaults to "recursive"
    :type method: Literal["recursive", "character", "token"], optional

    :param chunk_size: maximum size of a single chunk that is returned, defaults to 4000
    :type chunk_size: int, optional

    :param chunk_overlap: overlap in characters between chunks, defaults to 200
    :type chunk_overlap: int, optional

    :param encoding_name: encoding used in the TokenTextSplitter, defaults to "gpt2"
    :type encoding_name: str, optional

    :param model_name: model used in the TokenTextSplitter
    :type model_name: str, optional

    .. code-block:: python

        from ibm_watsonx_ai.foundation_models.extensions.rag.chunker import LangChainChunker

        text_splitter = LangChainChunker(
            method="recursive",
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks_ids = []

        for i, document in enumerate(data_loader):
            chunks = text_splitter.split_documents([document])
            chunks_ids.append(vector_store.add_documents(chunks, batch_size=300))
    """

    supported_methods = ("recursive", "character", "token")

    def __init__(
        self,
        method: Literal["recursive", "character", "token"] = "recursive",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        encoding_name: str = "gpt2",
        model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.method = method
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name
        self.model_name = model_name
        self.separators = kwargs.pop("separators", ["\n\n", r"(?<=\. )", "\n", " ", ""])
        self._text_splitter = self._get_text_splitter()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LangChainChunker):
            return self.to_dict() == other.to_dict()
        else:
            return NotImplemented

    def _get_text_splitter(self) -> TextSplitter:
        """Create an instance of TextSplitter based on the settings."""

        text_splitter: TextSplitter

        match self.method:
            case "recursive":
                from langchain.text_splitter import RecursiveCharacterTextSplitter

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=self.separators,
                    length_function=len,
                    add_start_index=True,
                )

            case "character":
                from langchain.text_splitter import CharacterTextSplitter

                text_splitter = CharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    add_start_index=True,
                )

            case "token":
                from langchain.text_splitter import TokenTextSplitter

                text_splitter = TokenTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    encoding_name=self.encoding_name,
                    model_name=self.model_name,
                    add_start_index=True,
                )

            case _:
                raise ValueError(
                    "Chunker method '{}' is not supported. Use one of {}".format(
                        self.method, self.supported_methods
                    )
                )

        return text_splitter

    def to_dict(self) -> dict[str, Any]:
        """
        Return dictionary that can be used to recreate an instance of the LangChainChunker.
        """
        params = (
            "method",
            "chunk_size",
            "chunk_overlap",
            "encoding_name",
            "model_name",
        )

        ret = {k: v for k, v in self.__dict__.items() if k in params}

        return ret

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LangChainChunker":
        """Create an instance from the dictionary."""

        return cls(**d)

    def _set_document_id_in_metadata_if_missing(
        self, documents: Iterable[Document]
    ) -> None:
        """
        Sets "document_id" in the metadata if it is missing.
        The document_id is the hash of the document's content.

        :param documents: sequence of documents
        :type documents: Iterable[langchain_core.documents.Document]

        :return: None
        """
        for doc in documents:
            if "document_id" not in doc.metadata:
                doc.metadata["document_id"] = str(hash(doc.page_content))

    def _set_sequence_number_in_metadata(
        self, chunks: list[Document]
    ) -> list[Document]:
        """
        Sets "sequence_number" in the metadata, sorted by chunks' "start_index".

        :param chunks: sequence of chunks of documents that contain context in a text format
        :type chunks: list[langchain_core.documents.Document]

        :return: chunks: list of updated chunks, sorted by document_id and sequence_number
        :type chunks: list[langchain_core.documents.Document]
        """
        # sort chunks by start_index for each document_id
        sorted_chunks = sorted(
            chunks, key=lambda x: (x.metadata["document_id"], x.metadata["start_index"])
        )

        document_sequence: dict[str, int] = {}
        for chunk in sorted_chunks:
            doc_id = chunk.metadata["document_id"]
            prev_seq_num = document_sequence.get(doc_id, 0)
            seq_num = prev_seq_num + 1
            document_sequence[doc_id] = seq_num
            chunk.metadata["sequence_number"] = seq_num

        return sorted_chunks

    def split_documents(self, documents: Sequence[Document]) -> list[Document]:
        """
        Split series of documents into smaller chunks based on the provided
        chunker settings. Each chunk has metadata that includes the document_id,
        sequence_number, and start_index.

        :param documents: sequence of elements that contain context in a text format
        :type documents: Sequence[langchain_core.documents.Document]

        :return: list of documents split into smaller ones, having less content
        :rtype: list[langchain_core.documents.Document]
        """
        self._set_document_id_in_metadata_if_missing(documents)
        chunks = self._text_splitter.split_documents(documents)
        sorted_chunks = self._set_sequence_number_in_metadata(chunks)
        return sorted_chunks
