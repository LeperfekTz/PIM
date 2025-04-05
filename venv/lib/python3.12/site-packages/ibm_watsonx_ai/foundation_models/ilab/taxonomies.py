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


class Taxonomy:
    """Class of InstructLab taxonomy."""

    def __init__(self, id: str, api_client: APIClient):
        self.id = id
        self._client = api_client
        self._href_definitions = self._client.service_instance._href_definitions

    def get_details(self) -> dict:
        """Get taxonomy import details

        :return: details of taxonomy import
        :rtype: dict
        """
        response = self._client.httpx_client.get(
            url=self._href_definitions.get_data_asset_href(self.id),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._client.repository._handle_response(
            200, "getting taxonomy details", response
        )

    def get_taxonomy_import(self) -> TaxonomyImport:
        """Get taxonomy import object

        :return: taxonomy import
        :rtype: TaxonomyImport
        """
        details = self.get_details()
        taxonomy_import_id = details["entity"]["wx_taxonomy"]["job_id"]
        taxonomy_import = TaxonomyImport(details["metadata"]["name"], self._client)
        taxonomy_import.id = taxonomy_import_id
        return taxonomy_import

    def get_taxonomy_tree(self) -> dict:
        """Get taxonomy import tree

        :return: taxonomy import tree
        :rtype: dict
        """
        return self.get_details()["entity"]["wx_taxonomy"]["taxonomy_tree"]

    def update_taxonomy_tree(self, updated_taxonomy_tree: dict) -> dict:
        """Update taxonomy import tree

        :param updated_taxonomy_tree: taxonomy tree with updated nodes
        :type updated_taxonomy_tree: dict
        """
        payload = [
            {"op": "replace", "path": "/taxonomy_tree", "value": updated_taxonomy_tree}
        ]

        response = self._client.httpx_client.patch(
            url=self._href_definitions.get_wx_taxonomy_href(self.id),
            json=payload,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._client.repository._handle_response(
            200, "update taxonomy tree", response
        )

    def delete(self) -> dict:
        """Delete taxonomy import"""
        params = self._client._params()
        params["hard_delete"] = "true"

        response = self._client.httpx_client.delete(
            url=self._href_definitions.get_data_asset_href(self.id),
            params=params,
            headers=self._client._get_headers(),
        )

        return self._client.repository._handle_response(
            204, "deleting taxonomy import", response
        )

    def _get_data_reference(self) -> dict:
        return {"location": {"id": self.id, "href": ""}, "type": "taxonomy_asset"}


class TaxonomyImport:
    """Class of InstructLab taxonomy import."""

    id: str

    def __init__(self, name: str, api_client: APIClient):
        self.name = name
        self._client = api_client
        self._href_definitions = self._client.service_instance._href_definitions

    def get_run_details(self) -> dict:
        """Get details of taxonomy import run

        :returns: details of taxonomy import
        :rtype: dict
        """
        if self.id is None:
            raise WMLClientError("Run in not started, operation cannot be performed.")

        response = self._client.httpx_client.get(
            url=self._href_definitions.get_taxonomies_import_href(self.id),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._client.repository._handle_response(
            200, "getting taxonomy import details", response
        )

    def get_run_status(self) -> str:
        """Get status of taxonomy import run

        :returns: status of taxonomy import
        :rtype: str
        """
        return self.get_run_details().get("entity", {}).get("status", {}).get("state")

    def delete_run(self) -> str:
        """Delete taxonomy import run"""
        if self.id is None:
            raise WMLClientError("Run in not started, operation cannot be performed.")

        params = self._client._params()
        params["hard_delete"] = "true"

        response = self._client.httpx_client.delete(
            url=self._href_definitions.get_taxonomies_import_href(self.id),
            params=params,
            headers=self._client._get_headers(),
        )

        return self._client.repository._handle_response(
            204, "deleting taxonomy import", response, json_response=False
        )

    def cancel_run(self) -> str:
        """Cancel taxonomy import run"""
        if self.id is None:
            raise WMLClientError("Run in not started, operation cannot be performed.")

        response = self._client.httpx_client.delete(
            url=self._href_definitions.get_taxonomies_import_href(self.id),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._client.repository._handle_response(
            204, "canceling taxonomy import", response, json_response=False
        )

    def get_taxonomy(self) -> Taxonomy:
        """Get taxonomy object for given taxonomy import

        :returns: taxonomy object
        :rtype: Taxonomy
        """
        details = self.get_run_details()
        return Taxonomy(
            details["entity"]["results_reference"]["location"]["id"], self._client
        )


class TaxonomiesRuns(BaseRuns):
    """Class of InstructLab taxonomy import runs."""

    def __init__(self, api_client: APIClient):
        url = (
            api_client.service_instance._href_definitions.get_taxonomies_imports_href()
        )

        BaseRuns.__init__(self, __name__, api_client, url)

    def get_taxonomy_import(self, taxonomy_import_id: str) -> TaxonomyImport:
        """Get taxonomy import object by id.

        :param taxonomy_import_id: id of given taxonomy import
        :type taxonomy_import_id: str

        :returns: taxonomy import object
        :rtype: TaxonomyImport
        """
        taxonomy_import_details = self.get_run_details(taxonomy_import_id)
        taxonomy_import = TaxonomyImport(
            taxonomy_import_details.get("metadata", {}).get("name"), self._client
        )
        taxonomy_import.id = taxonomy_import_id
        return taxonomy_import


class Taxonomies(WMLResource):
    """Class of InstructLab taxonomy import module."""

    _logger = logging.getLogger(__name__)

    def __init__(self, ilab_tuner_name: str, api_client: APIClient):
        WMLResource.__init__(self, "taxonomies", api_client)
        self.ilab_tuner_name = ilab_tuner_name
        self._client = api_client
        self._href_definitions = self._client.service_instance._href_definitions

    def run_import(
        self,
        *,
        data_reference: DataConnection,
        name: str | None = None,
        background_mode: bool = False,
    ) -> TaxonomyImport:
        """Run a taxonomy import process using `data_reference` with taxonomy Github location to `results_reference location`.

        :param data_reference: reference to github repo where taxonomy is stored
        :type data_reference: DataConnection

        :param background_mode: indicator if the method will run in the background, async or sync
        :type background_mode: bool, optional

        :return: taxonomy import object
        :rtype: TaxonomyImport

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment
            from ibm_watsonx_ai.helpers import DataConnection, GithubLocation

            experiment = TuneExperiment(credentials, ...)
            ilab_tuner = experiment.ilab_tuner(...)

            taxonomy_import = ilab_tuner.taxonomies.run_import(
                name="my_taxonomy",
                data_reference=DataConnection(
                    location=GithubLocation(
                        secret_manager_url="...",
                        secret_id="...",
                        path="."
                    )
                ))
        """
        name = name if name else f"{self.ilab_tuner_name}_taxonomy"
        payload = {
            "name": name,  # name problem
            "data_reference": data_reference.to_dict(),
        }

        params = self._client._params()

        if self._client.default_project_id:
            payload["project_id"] = self._client.default_project_id
            params.pop("project_id")
        elif self._client.default_space_id:
            payload["space_id"] = self._client.default_space_id
            params.pop("space_id")

        response = self._client.httpx_client.post(
            url=self._href_definitions.get_taxonomies_imports_href(),
            json=payload,
            params=params,
            headers=self._client._get_headers(),
        )

        res = self._handle_response(201, "importing taxonomy", response)

        taxonomy_import = TaxonomyImport(name=name, api_client=self._client)
        taxonomy_import.id = res["metadata"]["id"]

        if not background_mode:
            wait_for_run_finish(
                asked_object=taxonomy_import,
                res_name="Taxonomy import",
                logger=self._logger,
            )

        return taxonomy_import

    def runs(self) -> TaxonomiesRuns:
        """Get the historical runs.

        :returns: runs object
        :rtype: TaxonomiesRuns
        """
        return TaxonomiesRuns(self._client)
