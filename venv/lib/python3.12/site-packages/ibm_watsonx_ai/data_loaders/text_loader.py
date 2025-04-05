#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

__all__ = ["TextLoader"]

import io
import logging

from typing import TYPE_CHECKING
from queue import Empty

from ibm_watsonx_ai.utils import DisableWarningsLogger
from ibm_watsonx_ai.wml_client_error import (
    MissingExtension,
    WMLClientError,
    LoadingDocumentError,
)
from ibm_watsonx_ai.helpers.remote_document import RemoteDocument

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _asynch_download(args):
    """Helper function for parallel downloading documents (full asynchronous version)."""
    (q_input, qs_output) = args

    while True:
        try:
            i, doc = q_input.get(block=False)
            try:
                doc.download()
                qs_output[i].put(TextLoader(doc).load())
            except Exception as e:
                qs_output[i].put(LoadingDocumentError(doc.document_id, e))
        except Empty:
            return


class TextLoader:
    """
    TextLoader class for extraction txt, pdf, html, docx and md file from bytearray format.

    :param documents: Documents to extraction from bytearray format
    :type documents: RemoteDocument, list[RemoteDocument]

    """

    def __init__(self, document: RemoteDocument) -> None:
        self.file = document

    def load(self) -> Document:
        """
        Load text from bytearray data.
        """
        try:
            from langchain_core.documents import Document as LCDocument
        except ImportError:
            raise MissingExtension("langchain-core")

        file_content = getattr(self.file, "content", None)
        document_id = getattr(self.file, "document_id", None)
        file_type = self.identify_file_type(document_id)

        file_type_handlers = {
            "text/plain": self._txt_to_string,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": self._docs_to_string,
            "application/pdf": self._pdf_to_string,
            "text/html": self._html_to_string,
            "text/markdown": self._md_to_string,
        }

        try:
            handler = file_type_handlers[file_type]
        except KeyError:
            raise WMLClientError(
                f"Unsupported file type: {file_type}. Supported file types: {list(file_type_handlers)}."
            )

        text = handler(file_content)

        metadata = {
            "document_id": document_id,
        }

        return LCDocument(page_content=text, metadata=metadata)

    @staticmethod
    def identify_file_type(filename: str) -> str:
        """
        Identifying file type by bytearray input data
        """
        filename = filename.lower()
        if filename.endswith(".pdf"):
            return "application/pdf"
        elif filename.endswith(".html"):
            return "text/html"
        elif filename.endswith(".docx"):
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif filename.endswith(".txt"):
            return "text/plain"
        elif filename.endswith(".md"):
            return "text/markdown"
        else:
            raise WMLClientError(f"Cannot identify file type.")

    @staticmethod
    def _txt_to_string(binary_data: bytes) -> str:
        return binary_data.decode("utf-8", errors="ignore")

    @staticmethod
    def _docs_to_string(binary_data: bytes) -> str:
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise MissingExtension("python-docx")

        with io.BytesIO(binary_data) as open_docx_file:
            doc = DocxDocument(open_docx_file)
            full_text = [para.text for para in doc.paragraphs]
            return "\n".join(full_text)

    @staticmethod
    def _pdf_to_string(binary_data: bytes) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise MissingExtension("pypdf")

        with io.BytesIO(binary_data) as open_pdf_file:
            with DisableWarningsLogger():
                reader = PdfReader(open_pdf_file)
            full_text = [page.extract_text() for page in reader.pages]
            return "\n".join(full_text)

    @staticmethod
    def _html_to_string(binary_data: bytes) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise MissingExtension("beautifulsoup4")

        soup = BeautifulSoup(binary_data, "html.parser")
        return soup.get_text()

    @staticmethod
    def _md_to_string(binary_data: bytes) -> str:
        try:
            from markdown import markdown
        except ImportError:
            raise MissingExtension("markdown")
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise MissingExtension("beautifulsoup4")

        md = binary_data.decode("utf-8", errors="ignore")
        html = markdown(md)
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()
