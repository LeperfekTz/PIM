#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

import shlex
import subprocess
import time
from typing import Any, TYPE_CHECKING, Literal
from warnings import warn

import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.metanames import VolumeMetaNames
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    import pandas


class Volume(WMLResource):
    """Store and manage volume assets."""

    ConfigurationMetaNames = VolumeMetaNames()
    """MetaNames for volume assets creation."""

    def __init__(self, client: APIClient) -> None:
        WMLResource.__init__(self, __name__, client)

    def get_details(self, volume_id: str) -> dict:
        """Get volume details.

        :param volume_id: Unique ID of the volume
        :type volume_id: str

        :return: metadata of the volume details
        :rtype: dict

        **Example:**

        .. code-block:: python

            volume_details = client.volumes.get_details(volume_id)

        """
        Volume._validate_type(volume_id, "volume_id", str, True)

        response = requests.get(
            self._client.service_instance._href_definitions.volume_href(volume_id),
            headers=self._client._get_headers(zen=True),
        )

        if response.status_code == 200:
            return response.json()
        else:
            warn(f"{response.status_code} {response.text}")
            raise WMLClientError("Failed to Get the volume details. Try again.")

    def create(self, meta_props: dict[str, Any]) -> dict:
        """Create a volume asset.

        :param meta_props: metadata of the volume asset
        :type meta_props: dict

        :return: metadata of the created volume details
        :rtype: dict

        **Examples**

        Provision new PVC volume:

        .. code-block:: python

            metadata = {
                client.volumes.ConfigurationMetaNames.NAME: 'volume-for-wml-test',
                client.volumes.ConfigurationMetaNames.NAMESPACE: 'wmldev2',
                client.volumes.ConfigurationMetaNames.STORAGE_CLASS: 'nfs-client'
                client.volumes.ConfigurationMetaNames.STORAGE_SIZE: "2G"
            }

            asset_details = client.volumes.store(meta_props=metadata)

        Provision an existing PVC volume:

        .. code-block:: python

            metadata = {
                client.volumes.ConfigurationMetaNames.NAME: 'volume-for-wml-test',
                client.volumes.ConfigurationMetaNames.NAMESPACE: 'wmldev2',
                client.volumes.ConfigurationMetaNames.EXISTING_PVC_NAME: 'volume-for-wml-test'
            }

            asset_details = client.volumes.store(meta_props=metadata)

        """

        create_meta = {}
        if (
            self.ConfigurationMetaNames.EXISTING_PVC_NAME in meta_props
            and meta_props[self.ConfigurationMetaNames.EXISTING_PVC_NAME] is not None
        ):
            if (
                self.ConfigurationMetaNames.STORAGE_CLASS in meta_props
                and meta_props[self.ConfigurationMetaNames.STORAGE_CLASS] is not None
            ):
                raise WMLClientError(
                    "Failed while creating volume. Either provide EXISTING_PVC_NAME to create a volume using existing volume or"
                    "provide STORAGE_CLASS and STORAGE_SIZE for new volume creation"
                )
            else:
                create_meta.update(
                    {
                        "existing_pvc_name": meta_props[
                            self.ConfigurationMetaNames.EXISTING_PVC_NAME
                        ]
                    }
                )
        else:
            if (
                self.ConfigurationMetaNames.STORAGE_CLASS in meta_props
                and meta_props[self.ConfigurationMetaNames.STORAGE_CLASS] is not None
            ):
                if (
                    self.ConfigurationMetaNames.STORAGE_SIZE in meta_props
                    and meta_props[self.ConfigurationMetaNames.STORAGE_SIZE] is not None
                ):
                    create_meta.update(
                        {
                            "storageClass": meta_props[
                                self.ConfigurationMetaNames.STORAGE_CLASS
                            ]
                        }
                    )
                    create_meta.update(
                        {
                            "storageSize": meta_props[
                                self.ConfigurationMetaNames.STORAGE_SIZE
                            ]
                        }
                    )
                else:
                    raise WMLClientError(
                        "Failed to create volume. Missing input STORAGE_SIZE"
                    )

        if (
            self.ConfigurationMetaNames.EXISTING_PVC_NAME in meta_props
            and meta_props[self.ConfigurationMetaNames.EXISTING_PVC_NAME] is not None
        ):
            input_meta = {
                "addon_type": "volumes",
                "addon_version": "-",
                "create_arguments": {"metadata": create_meta},
                "namespace": meta_props[self.ConfigurationMetaNames.NAMESPACE],
                "display_name": meta_props[self.ConfigurationMetaNames.NAME],
            }
        else:
            input_meta = {
                "addon_type": "volumes",
                "addon_version": "-",
                "create_arguments": {"metadata": create_meta},
                "namespace": meta_props[self.ConfigurationMetaNames.NAMESPACE],
                "display_name": meta_props[self.ConfigurationMetaNames.NAME],
            }

        try:
            if self._client.CLOUD_PLATFORM_SPACES:
                creation_response = requests.post(
                    self._client.service_instance._href_definitions.volumes_href(),
                    headers=self._client._get_headers(zen=True),
                    json=input_meta,
                )

            else:
                creation_response = requests.post(
                    self._client.service_instance._href_definitions.volumes_href(),
                    headers=self._client._get_headers(zen=True),
                    json=input_meta,
                )
            if creation_response.status_code == 200:
                volume_id_details = (
                    creation_response.json()
                )  # messy details returned for backward compability
                import copy

                volume_details = copy.deepcopy(input_meta)
                volume_details.update(volume_id_details)
                actual_details = self.get_details(self.get_id(volume_id_details))
                volume_details.update(actual_details)
                return volume_details
            else:
                warn(f"{creation_response.status_code} {creation_response.text}")
                raise WMLClientError("Failed to create a volume. Try again.")
        except Exception as e:
            warn(f"Exception: {e}")
            raise WMLClientError("Failed to create a volume. Try again.")

    def start(
        self, name: str, wait_for_available: bool = False
    ) -> Literal["SUCCESS", "FAILED"]:
        """Start the volume service.

        :param name: unique name of the volume to be started
        :type name: str

        :param wait_for_available: flag indicating if method should wait until volume service is available
        :type wait_for_available: bool

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example:**

        .. code-block:: python

            client.volumes.start(volume_name)

        """

        if self._client.CPD_version >= 4.5 and "::" not in name:
            raise WMLClientError(
                "Invalid name to start volume. Correct volume name format: `<namespace>::<name>`. Retrieve the correct name using `client.volumes.get_name(client.volumes.get_details(volume_id))` command."
            )

        start_url = self._client.service_instance._href_definitions.volume_service_href(
            name
        )
        # Start the volume  service
        start_data: dict = {}
        try:
            start_data = {}
            creation_response = requests.post(
                start_url, headers=self._client._get_headers(zen=True), json=start_data
            )
            if creation_response.status_code == 200:
                print("Volume Service started")
                if wait_for_available:
                    retries = 0
                    volume_status = False
                    while True and retries < 60 and not volume_status:
                        volume_status = self.get_volume_status(name)
                        time.sleep(5)
                        retries += 1
                    if volume_status:
                        return "SUCCESS"

                volume_service_not_started_warning = (
                    "Volume Service has been started, but it is not available yet. "
                    "Check volume availability using get_volume_status method."
                )
                warn(volume_service_not_started_warning)
                return "FAILED"
            elif creation_response.status_code == 500:
                failed_to_start_volume_warning = (
                    "Failed to start the volume. "
                    "Make sure volume is in running with status RUNNING or UNKNOWN and then re-try"
                )
                warn(failed_to_start_volume_warning)
                return "FAILED"
            else:
                warn(f"{creation_response.status_code} {creation_response.text}")
                raise WMLClientError("Failed to start the file to  volume. Try again.")
        except Exception as e:
            warn(f"Exception: {e}")
            raise WMLClientError("Failed to start the file to  volume. Try again.")

    def get_volume_status(self, name: str) -> bool:
        """Monitor a volume's file server status.

        :param name: name of the volume to retrieve status for
        :type name: str

        :return: status of the volume (True if volume is available, otherwise False)
        :rtype: bool

        **Example:**

        .. code-block:: python

            client.volumes.get_volume_status(volume_name)

        """

        if self._client.CPD_version >= 4.5 and "::" not in name:
            raise WMLClientError(
                "Invalid name to start volume. Correct volume name format: `<namespace>::<name>`. Retrieve the correct name using `client.volumes.get_name(client.volumes.get_details(volume_id))` command."
            )

        monitor_url = (
            self._client.service_instance._href_definitions.volume_monitor_href(name)
        )
        try:
            monitor_response = requests.get(
                monitor_url, headers=self._client._get_headers(zen=True)
            )
            if monitor_response.status_code == 200:
                return True
            elif monitor_response.status_code == 502:
                return False
            else:
                warn(f"{monitor_response.status_code} {monitor_response.text}")
                raise WMLClientError("Cannot retrieve status of the volume.")
        except Exception as e:
            warn(f"Exception: {e}")
            raise WMLClientError("Cannot retrieve status of the volume.")

    def upload_file(self, name: str, file_path: str) -> Literal["SUCCESS", "FAILED"]:
        """Upload the data file into stored volume.

        :param name: unique name of the stored volume
        :type name: str
        :param file_path: file to be uploaded into the volume
        :type file_path: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example:**

        .. code-block:: python

            client.volumes.upload_file('testA', 'DRUG.csv')

        """

        header_input = self._client._get_headers(zen=True)
        zen_token = header_input.get("Authorization")

        filename_to_upload = file_path.split("/")[-1]
        upload_url_file = (
            self._client.service_instance._href_definitions.volume_upload_href(name)
            + filename_to_upload
        )
        cmd_str = (
            'curl -k  -X PUT "'
            + upload_url_file
            + '"'
            + "  -H 'Content-Type: multipart/form-data' -H 'Authorization: "
            + zen_token
            + "' -F upFile='@"
            + file_path
            + "'"
        )
        args = shlex.split(cmd_str)
        upload_response = subprocess.run(args, capture_output=True, text=True)
        if upload_response.returncode == 0:
            import json

            try:
                cmd_output = json.loads(upload_response.stdout)
                if cmd_output.get("_statusCode_") == 403:
                    insufficient_permissions_warning = (
                        "It seems that you don't have the necessary permissions to perform this action. "
                        "Please review your permissions and try again once they have been updated."
                    )
                    warn(insufficient_permissions_warning)
                    return "FAILED"
                print(cmd_output.get("message"))
                return "SUCCESS"
            except Exception:
                pass

        upload_response_error_warning = f"{upload_response.returncode} {upload_response.stdout} {upload_response.stderr}"
        warn(upload_response_error_warning)
        warn("Failed to upload the file to volume. Try again.")
        return "FAILED"

    def list(self) -> pandas.DataFrame:
        """Lists stored volumes in a table format.

        :return: pandas.DataFrame with listed volumes
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.volumes.list()
        """

        href = self._client.service_instance._href_definitions.volumes_href()
        params = {}
        params.update({"addon_type": "volumes"})

        response = requests.get(
            href, params=params, headers=self._client._get_headers(zen=True)
        )

        asset_details = self._handle_response(200, "list volumes", response)
        asset_list = asset_details.get("service_instances")
        volume_values = [
            (m["display_name"], m["id"], m["provision_status"]) for m in asset_list
        ]

        table = self._list(
            volume_values,
            ["NAME", "ID", "PROVISION_STATUS"],
            None,
        )

        return table

    @staticmethod
    def get_id(volume_details: dict) -> str:
        """Get unique Id of stored volume details.

        :param volume_details: metadata of the stored volume details
        :type volume_details: dict

        :return: unique Id of stored volume asset
        :rtype: str

        **Example:**

        .. code-block:: python

            volume_id = client.volumes.get_id(volume_details)

        """

        Volume._validate_type(volume_details, "volume_details", object, True)
        if (
            "service_instance" in volume_details
            and volume_details.get("service_instance") is not None
        ):
            vol_details = volume_details.get("service_instance")
            return WMLResource._get_required_element_from_dict(
                vol_details, "volume_assets_details", ["id"]
            )
        else:
            return WMLResource._get_required_element_from_dict(
                volume_details, "volume_assets_details", ["id"]
            )

    @staticmethod
    def get_name(volume_details: dict) -> str:
        """Get unique name of stored volume asset.

        :params volume_details: metadata of the stored volume asset
        :type volume_details: dict

        :return: unique name of stored volume asset
        :rtype: str

        **Example:**

        .. code-block:: python

            volume_name = client.volumes.get_name(asset_details)

        """
        Volume._validate_type(volume_details, "asset_details", object, True)
        if (
            "service_instance" in volume_details
            and volume_details.get("service_instance") is not None
        ):
            vol_details = volume_details.get("service_instance")
            return WMLResource._get_required_element_from_dict(
                vol_details, "volume_assets_details", ["display_name"]
            )
        else:
            return WMLResource._get_required_element_from_dict(
                volume_details, "volume_assets_details", ["display_name"]
            )

    def delete(self, volume_id: str) -> Literal["SUCCESS", "FAILED"]:
        """Delete a volume.

        :param volume_id: unique ID of the volume
        :type volume_id: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example:**

        .. code-block:: python

            client.volumes.delete(volume_id)

        """
        Volume._validate_type(volume_id, "volume_id", str, True)

        response = requests.delete(
            self._client.service_instance._href_definitions.volume_href(volume_id),
            headers=self._client._get_headers(zen=True),
        )

        if response.status_code == 200 or response.status_code == 204:
            print("Successfully deleted volume service.")
            return "SUCCESS"
        else:
            warn("Failed to delete volume.")
            warn(f"{response.status_code} {response.text}")
            return "FAILED"

    def stop(self, volume_name: str) -> Literal["SUCCESS", "FAILED"]:
        """Stop the volume service.

        :param volume_name: unique name of the volume
        :type volume_name: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example:**

        .. code-block:: python

            client.volumes.stop(volume_name)

        """
        Volume._validate_type(volume_name, "volume_name", str, True)

        response = requests.delete(
            self._client.service_instance._href_definitions.volume_service_href(
                volume_name
            ),
            headers=self._client._get_headers(zen=True),
        )

        if response.status_code == 200:
            print("Successfully stopped volume service.")
            return "SUCCESS"
        else:
            warn(f"{response.status_code} {response.text}")
            return "FAILED"
