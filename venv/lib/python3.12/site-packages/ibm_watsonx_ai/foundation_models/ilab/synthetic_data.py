#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

import json
from typing import TYPE_CHECKING
import logging

from ibm_watsonx_ai.foundation_models.ilab.helper import wait_for_run_finish, BaseRuns
from ibm_watsonx_ai.foundation_models.ilab.taxonomies import Taxonomy
from ibm_watsonx_ai.helpers.connections import DataConnection
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class SyntheticDataGeneration:
    """Class of InstructLab synthetic data generation run."""

    id: str

    def __init__(self, name: str, api_client: APIClient) -> None:
        self.name = name
        self._client = api_client
        self._href_definitions = self._client.service_instance._href_definitions

    def get_results_reference(self) -> DataConnection:
        """Get results reference to generated synthetic data.

        :returns: data connection to generated synthetic data
        :rtype: DataConnection
        """
        return DataConnection.from_dict(
            self.get_run_details()["entity"]["results_reference"]
        )

    def get_run_details(self) -> dict:
        """Get synthetic data generation details

        :return: details of synthetic data generation
        :rtype: dict
        """
        if self.id is None:
            raise WMLClientError("Run in not started, operation cannot be performed.")

        response = self._client.httpx_client.get(
            url=self._href_definitions.get_synthetic_data_generation_href(self.id),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._client.repository._handle_response(
            200, "getting synthetic data generation details", response
        )

    def get_run_status(self) -> str:
        """Get synthetic data generation status

        :return: status of synthetic data generation
        :rtype: str
        """
        return self.get_run_details()["entity"].get("status", {}).get("state")

    def delete_run(self) -> str:
        """Delete synthetic data generation run"""
        if self.id is None:
            raise WMLClientError("Run in not started, operation cannot be performed.")

        params = self._client._params()
        params["hard_delete"] = "true"

        response = self._client.httpx_client.delete(
            url=self._href_definitions.get_synthetic_data_generation_href(self.id),
            params=params,
            headers=self._client._get_headers(),
        )

        return self._client.repository._handle_response(
            204, "deletion of synthetic data generation", response, json_response=False
        )

    def cancel_run(self) -> str:
        """Cancel synthetic data generation run"""
        if self.id is None:
            raise WMLClientError("Run in not started, operation cannot be performed.")

        response = self._client.httpx_client.delete(
            url=self._href_definitions.get_synthetic_data_generation_href(self.id),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._client.repository._handle_response(
            204,
            "cancelation of synthetic data generation",
            response,
            json_response=False,
        )


class SDGRuns(BaseRuns):
    """Class of InstructLab synthetic generation runs."""

    def __init__(self, api_client: APIClient) -> None:
        url = (
            api_client.service_instance._href_definitions.get_synthetic_data_generations_href()
        )

        BaseRuns.__init__(self, __name__, api_client, url)

    def get_synthetic_data_generation(self, sdg_id: str) -> SyntheticDataGeneration:
        """Get synthetic data generation object

        :param sdg_id: id of synthetic data generation object
        :type sdg_id: str

        :returns: synthetic data generation object
        :rtype: SyntheticDataGeneration
        """
        sdg_details = self.get_run_details(sdg_id)
        sdg = SyntheticDataGeneration(
            sdg_details.get("metadata", {}).get("name"), self._client
        )
        sdg.id = sdg_id
        return sdg


class SyntheticData(WMLResource):
    """Class of InstructLab synthetic data generation module."""

    _logger = logging.getLogger(__name__)

    def __init__(self, ilab_tuner_name: str, api_client: APIClient) -> None:
        WMLResource.__init__(self, "synthetic data generation", api_client)
        self.ilab_tuner_name = ilab_tuner_name
        self._client = api_client
        self._href_definitions = self._client.service_instance._href_definitions

    def generate(
        self,
        *,
        name: str | None = None,
        taxonomy: Taxonomy,
        background_mode: bool = False,
    ) -> SyntheticDataGeneration:
        """Generate synthetic data from updated taxonomy

        :param name: name of synthetic data generation run
        :type name: str

        :param taxonomy: taxonomy object
        :type taxonomy: Taxonomy

        :param background_mode: indicator if the method will run in the background, async or sync
        :type background_mode: bool, optional

        :returns: synthetic data generation run object
        :rtype: SyntheticDataGeneration
        """

        sdg = SyntheticDataGeneration(
            name if name else f"{self.ilab_tuner_name} - Synthetic Data Generation",
            self._client,
        )

        payload = {
            "name": sdg.name,
            "data_reference": taxonomy.get_taxonomy_import().get_run_details()[
                "entity"
            ]["results_reference"],
        }

        params = self._client._params()

        if self._client.default_project_id:
            payload["project_id"] = self._client.default_project_id
            params.pop("project_id")
        elif self._client.default_space_id:
            payload["space_id"] = self._client.default_space_id
            params.pop("space_id")

        response = self._client.httpx_client.post(
            url=self._href_definitions.get_synthetic_data_generations_href(),
            json=payload,
            params=params,
            headers=self._client._get_headers(),
        )

        res = self._handle_response(201, "running synthetic data generation", response)

        sdg.id = res["metadata"]["id"]

        if not background_mode:
            wait_for_run_finish(
                asked_object=sdg,
                res_name="Synthetic data generation",
                logger=self._logger,
            )

        return sdg

    def runs(self) -> SDGRuns:
        """Get the historical runs.

        :returns: runs object
        :rtype: SDGRuns
        """
        return SDGRuns(self._client)
