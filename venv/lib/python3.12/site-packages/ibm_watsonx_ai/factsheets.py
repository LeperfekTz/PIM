#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
from typing import TYPE_CHECKING

import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.utils import print_text_header_h2
from ibm_watsonx_ai.wml_client_error import WMLClientError, WrongMetaProps
from ibm_watsonx_ai.wml_resource import WMLResource
from .metanames import FactsheetsMetaNames

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class Factsheets(WMLResource):
    """Link WML Model to Model Entry."""

    cloud_platform_spaces: bool = False
    icp_platform_spaces: bool = False

    def __init__(self, client: APIClient) -> None:
        WMLResource.__init__(self, __name__, client)

        self.ConfigurationMetaNames = FactsheetsMetaNames()

        if client.CLOUD_PLATFORM_SPACES:
            Factsheets.cloud_platform_spaces = True

        if client.ICP_PLATFORM_SPACES:
            Factsheets.icp_platform_spaces = True

    def register_model_entry(
        self, model_id: str, meta_props: dict[str, str], catalog_id: str | None = None
    ) -> dict:
        """Link WML Model to Model Entry

        :param model_id: ID of the published model/asset
        :type model_id: str

        :param meta_props: metaprops, to see the available list of meta names use:

            .. code-block:: python

                client.factsheets.ConfigurationMetaNames.get()

        :type meta_props: dict[str, str]

        :param catalog_id: catalog ID where you want to register model
        :type catalog_id: str, optional

        :return: metadata of the registration
        :rtype: dict

        **Example:**

        .. code-block:: python

            meta_props = {
                client.factsheets.ConfigurationMetaNames.ASSET_ID: '83a53931-a8c0-4c2f-8319-c793155e7517'}

            registration_details = client.factsheets.register_model_entry(model_id, catalog_id, meta_props)

        or

        .. code-block:: python

            meta_props = {
                client.factsheets.ConfigurationMetaNames.NAME: "New model entry",
                client.factsheets.ConfigurationMetaNames.DESCRIPTION: "New model entry"}

            registration_details = client.factsheets.register_model_entry(model_id, meta_props)

        """
        Factsheets._validate_type(model_id, "model_id", str, True)
        Factsheets._validate_type(catalog_id, "catalog_id", str, False)
        metaProps = self.ConfigurationMetaNames._generate_resource_metadata(meta_props)

        params = self._client._params()

        if catalog_id is not None:
            params["catalog_id"] = catalog_id
            if "project_id" in params:
                del params["project_id"]
            elif "space_id" in params:
                del params["space_id"]

        name_in = self.ConfigurationMetaNames.NAME in metaProps
        description_in = self.ConfigurationMetaNames.DESCRIPTION in metaProps
        asset_id_in = self.ConfigurationMetaNames.ASSET_ID in metaProps

        # check for metaprops correctness
        reason = "Please provide either NAME and DESCRIPTION or ASSET_ID"
        if name_in and description_in:
            if asset_id_in:
                raise WrongMetaProps(reason=reason)

        elif asset_id_in:
            if name_in or description_in:
                raise WrongMetaProps(reason=reason)

        else:
            raise WrongMetaProps(reason=reason)

        url = (
            self._client.service_instance._href_definitions.get_wkc_model_register_href(
                model_id
            )
        )

        response = requests.post(
            url,
            json=metaProps,
            params=params,  # version is mandatory
            headers=self._client._get_headers(),
        )

        if response.status_code == 200:
            print_text_header_h2(
                f"Successfully finished linking WML Model '{model_id}' to Model Entry."
            )

        else:
            error_msg = "WML Model registration failed"
            reason = response.text
            print(reason)
            print_text_header_h2(error_msg)
            raise WMLClientError(
                error_msg + ". Error: " + str(response.status_code) + ". " + reason
            )

        return response.json()

    def list_model_entries(self, catalog_id: str | None = None) -> dict:
        """Return all WKC Model Entry assets for a catalog.

        :param catalog_id: catalog ID where you want to register model. If no catalog_id is provided, WKC Model Entry assets from all catalogs are listed.
        :type catalog_id: str, optional

        :return: all WKC Model Entry assets for a catalog
        :rtype: dict

        **Example:**

        .. code-block:: python

            model_entries = client.factsheets.list_model_entries(catalog_id)

        """
        if catalog_id is not None:
            Factsheets._validate_type(catalog_id, "catalog_id", str, True)
            url = self._client.service_instance._href_definitions.get_wkc_model_list_from_catalog_href(
                catalog_id
            )

        else:
            url = (
                self._client.service_instance._href_definitions.get_wkc_model_list_all_href()
            )

        response = requests.get(
            url,
            params=self._client._params(),  # version is mandatory
            headers=self._client._get_headers(),
        )

        if response.status_code == 200:
            return response.json()

        else:
            error_msg = "WKC Models listing failed"
            reason = response.text
            print(reason)
            print_text_header_h2(error_msg)
            raise WMLClientError(
                error_msg + ". Error: " + str(response.status_code) + ". " + reason
            )

    def unregister_model_entry(
        self, asset_id: str, catalog_id: str | None = None
    ) -> None:
        """Unregister WKC Model Entry

        :param asset_id: ID of the WKC model entry
        :type asset_id: str

        :param catalog_id: catalog ID where the asset is stored, when not provided,
            default client space or project will be taken
        :type catalog_id: str, optional

        **Example:**

        .. code-block:: python

            model_entries = client.factsheets.unregister_model_entry(asset_id='83a53931-a8c0-4c2f-8319-c793155e7517',
                                                                     catalog_id='34553931-a8c0-4c2f-8319-c793155e7517')

        or

        .. code-block:: python

            client.set.default_space('98f53931-a8c0-4c2f-8319-c793155e7517')
            model_entries = client.factsheets.unregister_model_entry(asset_id='83a53931-a8c0-4c2f-8319-c793155e7517')

        """
        Factsheets._validate_type(asset_id, "asset_id", str, True)
        Factsheets._validate_type(catalog_id, "catalog_id", str, False)
        url = self._client.service_instance._href_definitions.get_wkc_model_delete_href(
            asset_id
        )

        params = self._client._params()
        if catalog_id is not None:
            params["catalog_id"] = catalog_id

            if "space_id" in str(params):
                del params["space_id"]

            elif "project_id" in str(params):
                del params["project_id"]

        response = requests.delete(
            url,
            params=params,  # version is mandatory
            headers=self._client._get_headers(),
        )

        if response.status_code == 204:
            print_text_header_h2(
                f"Successfully finished unregistering WKC Model '{asset_id}' Entry."
            )

        else:
            error_msg = "WKC Model Entry unregistering failed"
            reason = response.text
            print(reason)
            print_text_header_h2(error_msg)
            raise WMLClientError(
                error_msg + ". Error: " + str(response.status_code) + ". " + reason
            )
