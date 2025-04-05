#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING
import logging

from ibm_watsonx_ai.foundation_models.ilab.helper import wait_for_run_finish, BaseRuns
from ibm_watsonx_ai.helpers.connections import (
    DataConnection,
)
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class DocumentExtraction:
    """Class of InstructLab document extraction."""

    id: str

    def __init__(self, name: str, api_client: APIClient):
        self.name = name
        self._client = api_client
        self._href_definitions = self._client.service_instance._href_definitions

    def get_run_details(self) -> dict:
        """Get document extraction details

        :return: details of document extraction
        :rtype: dict
        """
        if self.id is None:
            raise WMLClientError("Run in not started, operation cannot be performed.")

        response = self._client.httpx_client.get(
            url=self._href_definitions.get_document_extraction_href(self.id),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._client.repository._handle_response(
            200, "getting documents extraction details", response
        )

    def get_run_status(self) -> str:
        """Get document extraction status

        :return: status of document extraction
        :rtype: str
        """
        return self.get_run_details()["entity"].get("status", {}).get("state")

    def delete_run(self) -> str:
        """Delete document extraction run"""
        if self.id is None:
            raise WMLClientError("Run in not started, operation cannot be performed.")

        params = self._client._params()
        params["hard_delete"] = "true"

        response = self._client.httpx_client.delete(
            url=self._href_definitions.get_document_extraction_href(self.id),
            params=params,
            headers=self._client._get_headers(),
        )

        return self._client.repository._handle_response(
            204, "deleting of document extraction", response, json_response=False
        )

    def cancel_run(self) -> str:
        """Cancel document extraction run"""
        if self.id is None:
            raise WMLClientError("Run in not started, operation cannot be performed.")

        response = self._client.httpx_client.delete(
            url=self._href_definitions.get_document_extraction_href(self.id),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._client.repository._handle_response(
            204, "cancelation of documents extraction", response, json_response=False
        )


class DocumentExtractionsRuns(BaseRuns):
    """Class of InstructLab document extraction runs."""

    def __init__(self, api_client: APIClient):
        url = (
            api_client.service_instance._href_definitions.get_document_extractions_href()
        )

        BaseRuns.__init__(self, __name__, api_client, url)

    def get_document_extraction(
        self, document_extraction_id: str
    ) -> DocumentExtraction:
        """Get document extraction object

        :param document_extraction_id: id of document extraction object
        :type document_extraction_id: str

        :returns: document extraction object
        :rtype: DocumentExtraction
        """
        doc_extr_details = self.get_run_details(document_extraction_id)
        doc_extr = DocumentExtraction(
            doc_extr_details.get("metadata", {}).get("name"), self._client
        )
        doc_extr.id = document_extraction_id
        return doc_extr


class DocumentExtractions(WMLResource):
    """Class of InstructLab document extraction module."""

    _logger = logging.getLogger(__name__)

    def __init__(self, ilab_tuner_name: str, api_client: APIClient):
        WMLResource.__init__(self, "document extractions", api_client)
        self.ilab_tuner_name = ilab_tuner_name
        self._client = api_client
        self._href_definitions = self._client.service_instance._href_definitions

    def extract(
        self,
        *,
        name: str | None = None,
        document_references: list[DataConnection],
        results_reference: DataConnection,
        background_mode: bool = False,
    ) -> DocumentExtraction:
        """Extract .md document from given .pdf document

        :param name: document extraction run name
        :type name: str

        :param document_references: .pdf document location
        :type document_references: list[DataConnection]

        :param results_reference: .md file extraction location
        :type results_reference: DataConnection

        :param background_mode: indicator if the method will run in the background, async or sync
        :type background_mode: bool, optional

        :returns: document extraction run
        :rtype: DocumentExtraction
        """
        doc = DocumentExtraction(
            name if name else f"{self.ilab_tuner_name} - Documents Extraction",
            self._client,
        )

        payload = {
            "name": doc.name,
            "document_references": [
                doc_ref.to_dict() for doc_ref in document_references
            ],
            "results_reference": results_reference.to_dict(),
        }

        params = self._client._params()

        if self._client.default_project_id:
            payload["project_id"] = self._client.default_project_id
            params.pop("project_id")
        elif self._client.default_space_id:
            payload["space_id"] = self._client.default_space_id
            params.pop("space_id")

        response = self._client.httpx_client.post(
            url=self._href_definitions.get_document_extractions_href(),
            json=payload,
            params=params,
            headers=self._client._get_headers(),
        )

        res = self._handle_response(201, "running documents extraction", response)

        doc.id = res["metadata"]["id"]

        if not background_mode:
            wait_for_run_finish(
                asked_object=doc,
                res_name="Document extraction",
                logger=self._logger,
            )

        return doc

    def runs(self) -> DocumentExtractionsRuns:
        """Get the historical runs.

        :returns: runs object
        :rtype: DocumentExtractionsRuns
        """
        return DocumentExtractionsRuns(self._client)
