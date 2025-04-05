#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from warnings import warn
import json

import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.metanames import HwSpecMetaNames
from ibm_watsonx_ai.utils import HW_SPEC_DETAILS_TYPE
from ibm_watsonx_ai.utils.utils import _get_id_from_deprecated_uid
from ibm_watsonx_ai.wml_client_error import WMLClientError, ResourceIdByNameNotFound
from ibm_watsonx_ai.wml_resource import WMLResource


if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    from pandas import DataFrame


class HwSpec(WMLResource):
    """Store and manage hardware specs."""

    ConfigurationMetaNames = HwSpecMetaNames()
    """MetaNames for Hardware Specification."""

    def __init__(self, client: APIClient) -> None:
        WMLResource.__init__(self, __name__, client)

    def get_details(self, hw_spec_id: str | None = None, **kwargs: Any) -> dict:
        """Get hardware specification details.

        :param hw_spec_id: unique ID of the hardware spec
        :type hw_spec_id: str

        :return: metadata of the hardware specifications
        :rtype: dict

        **Example:**

        .. code-block:: python

            hw_spec_details = client.hardware_specifications.get_details(hw_spec_uid)

        """
        hw_spec_id = _get_id_from_deprecated_uid(
            kwargs, hw_spec_id, "hw_spec", can_be_none=False
        )

        HwSpec._validate_type(hw_spec_id, "hw_spec_id", str, True)

        response = requests.get(
            self._client.service_instance._href_definitions.get_hw_spec_href(
                hw_spec_id
            ),
            params=self._client._params(skip_space_project_chk=True),
            headers=self._client._get_headers(),
        )

        if response.status_code == 200:
            return self._get_required_element_from_response(
                self._handle_response(200, "get hardware spec details", response)
            )
        else:
            return self._handle_response(200, "get hardware spec details", response)

    def store(self, meta_props: dict) -> dict:
        """Create a hardware specification.

        :param meta_props: metadata of the hardware specification configuration. To see available meta names, use:

            .. code-block:: python

                client.hardware_specifications.ConfigurationMetaNames.get()

        :type meta_props: dict

        :return: metadata of the created hardware specification
        :rtype: dict

        **Example:**

        .. code-block:: python

            meta_props = {
                client.hardware_specifications.ConfigurationMetaNames.NAME: "custom hardware specification",
                client.hardware_specifications.ConfigurationMetaNames.DESCRIPTION: "Custom hardware specification creted with SDK",
                client.hardware_specifications.ConfigurationMetaNames.NODES:{"cpu":{"units":"2"},"mem":{"size":"128Gi"},"gpu":{"num_gpu":1}}
             }

            client.hardware_specifications.store(meta_props)

        """

        HwSpec._validate_type(meta_props, "meta_props", dict, True)
        hw_spec_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props, with_validation=True, client=self._client
        )

        hw_spec_meta_json = json.dumps(hw_spec_meta)
        href = self._client.service_instance._href_definitions.get_hw_specs_href()

        creation_response = requests.post(
            href,
            params=self._client._params(),
            headers=self._client._get_headers(),
            data=hw_spec_meta_json,
        )

        hw_spec_details = self._handle_response(
            201, "creating hardware specification", creation_response
        )

        return hw_spec_details

    def list(self, name: str | None = None, limit: int | None = None) -> DataFrame:
        """List hardware specifications in a table format.

        :param name: unique ID of the hardware spec
        :type name: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed hardware specifications
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.hardware_specifications.list()

        """

        params = self._client._params()

        if name is not None:
            params.update({"name": name})

        # Todo provide api to return
        href = self._client.service_instance._href_definitions.get_hw_specs_href()

        response = requests.get(href, params, headers=self._client._get_headers())

        self._handle_response(200, "list hw_specs", response)
        asset_details = self._handle_response(200, "list assets", response)["resources"]
        hw_spec_values = [
            (
                m["metadata"]["name"],
                m["metadata"]["asset_id"],
                m["metadata"]["description"] if "description" in m["metadata"] else "",
            )
            for m in asset_details
        ]
        table = self._list(hw_spec_values, ["NAME", "ID", "DESCRIPTION"], limit)
        return table

    @staticmethod
    def get_id(hw_spec_details: dict) -> str:
        """Get the ID of a hardware specifications asset.

        :param hw_spec_details: metadata of the hardware specifications
        :type hw_spec_details: dict

        :return: unique ID of the hardware specifications
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_id = client.hardware_specifications.get_id(hw_spec_details)

        """
        HwSpec._validate_type(hw_spec_details, "hw_spec_details", object, True)
        HwSpec._validate_type_of_details(hw_spec_details, HW_SPEC_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(
            hw_spec_details, "hw_spec_details", ["metadata", "asset_id"]
        )

    @staticmethod
    def get_uid(hw_spec_details: dict) -> str:
        """Get the UID of a hardware specifications asset.

        *Deprecated:* Use ``get_id(hw_spec_details)`` instead.

        :param hw_spec_details: metadata of the hardware specifications
        :type hw_spec_details: dict

        :return: unique ID of the hardware specifications
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_uid = client.hardware_specifications.get_uid(hw_spec_details)

        """
        get_uid_method_deprecated_warning = (
            "This method is deprecated, please use `get_id(hw_spec_details)` instead"
        )
        warn(get_uid_method_deprecated_warning, category=DeprecationWarning)

        return HwSpec.get_id(hw_spec_details)

    @staticmethod
    def get_href(hw_spec_details: dict) -> str:
        """Get the URL of hardware specifications.

        :param hw_spec_details: details of the hardware specifications
        :type hw_spec_details: dict

        :return: href of the hardware specifications
        :rtype: str

        **Example:**

        .. code-block:: python

            hw_spec_details = client.hw_spec.get_details(hw_spec_id)
            hw_spec_href = client.hw_spec.get_href(hw_spec_details)

        """
        HwSpec._validate_type(hw_spec_details, "hw_spec_details", object, True)
        HwSpec._validate_type_of_details(hw_spec_details, HW_SPEC_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(
            hw_spec_details, "hw_spec_details", ["metadata", "href"]
        )

    def get_id_by_name(self, hw_spec_name: str) -> str:
        """Get the unique ID of a hardware specification for the given name.

        :param hw_spec_name: name of the hardware specification
        :type hw_spec_name: str

        :return: unique ID of the hardware specification
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_id = client.hardware_specifications.get_id_by_name(hw_spec_name)

        """
        HwSpec._validate_type(hw_spec_name, "hw_spec_name", str, True)
        parameters = self._client._params(skip_space_project_chk=True)
        parameters.update(name=hw_spec_name)

        response = requests.get(
            self._client.service_instance._href_definitions.get_hw_specs_href(),
            params=parameters,
            headers=self._client._get_headers(),
        )

        if response.status_code == 200:
            total_values = self._handle_response(200, "list assets", response)[
                "total_results"
            ]
            if total_values != 0:
                hw_spec_details = self._handle_response(200, "list assets", response)[
                    "resources"
                ]
                return hw_spec_details[0]["metadata"]["asset_id"]
            else:
                raise ResourceIdByNameNotFound(hw_spec_name, "hardware spec")
        else:
            raise WMLClientError(
                "Failed to Get the hardware specification id by name. Try again."
            )

    def get_uid_by_name(self, hw_spec_name: str) -> str:
        """Get the unique ID of a hardware specification for the given name.

        *Deprecated:* Use ``get_id_by_name(hw_spec_name)`` instead.

        :param hw_spec_name: name of the hardware specification
        :type hw_spec_name: str

        :return: unique ID of the hardware specification
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_uid = client.hardware_specifications.get_uid_by_name(hw_spec_name)

        """
        get_uid_method_deprecated_warning = "This method is deprecated, please use `get_id_by_name(hw_spec_name)` instead"
        warn(get_uid_method_deprecated_warning, category=DeprecationWarning)
        return HwSpec.get_id_by_name(self, hw_spec_name)

    def delete(self, hw_spec_id: str) -> str:
        """Delete a hardware specification.

        :param hw_spec_id: unique ID of the hardware specification to be deleted
        :type hw_spec_id: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str
        """
        HwSpec._validate_type(hw_spec_id, "hw_spec_id", str, True)

        response = requests.delete(
            self._client.service_instance._href_definitions.get_hw_spec_href(
                hw_spec_id
            ),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(204, "delete hardware specification", response)

    def _get_required_element_from_response(
        self, response_data: dict[str, Any]
    ) -> dict:

        WMLResource._validate_type(response_data, "hw_spec_response", dict)
        try:
            new_el = {
                "metadata": {
                    "name": response_data["metadata"]["name"],
                    "asset_id": response_data["metadata"]["asset_id"],
                    "href": response_data["metadata"]["href"],
                    "asset_type": response_data["metadata"]["asset_type"],
                    "created_at": response_data["metadata"]["created_at"],
                },
                "entity": response_data["entity"],
            }

            if "href" in response_data["metadata"]:
                href_without_host = response_data["metadata"]["href"].split(".com")[-1]
                new_el["metadata"].update({"href": href_without_host})

            return new_el
        except Exception:
            raise WMLClientError(
                "Failed to read Response from down-stream service: "
                + str(response_data)
            )
