#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

__all__ = ["RemoteDocument"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ibm_watsonx_ai.helpers.connections import DataConnection


class RemoteDocument:
    """
    Helper class for preparing and parallel download of document content.

    :param connection: Connection to resource
    :type connection: DataConnection

    :param document_id: ID of downloaded document, file name is preferred value. Must correspond to `document_ids` column
        in benchmarking data, if context based sampling is applied.
    :type document_id: str, optional
    """

    def __init__(
        self,
        *,
        connection: DataConnection,
        document_id: str | None = None,
    ):
        self.connection = connection
        self.document_id = document_id if document_id else id(connection)

        self.content: bytes

    def download(self):
        """
        Downloads the document content and place it into `content` property of the object.
        """
        self.content = self.connection.read(binary=True)
