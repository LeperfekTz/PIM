#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
import os
from typing import Any, TYPE_CHECKING, TypeAlias
from warnings import warn

import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.utils import (
    DATA_ASSETS_DETAILS_TYPE,
    modify_details_for_script_and_shiny,
)
from ibm_watsonx_ai.metanames import ShinyMetaNames
from ibm_watsonx_ai.utils.utils import _get_id_from_deprecated_uid
from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    ApiRequestFailure,
    ForbiddenActionForGitBasedProject,
)


ListType: TypeAlias = list

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    from pandas import DataFrame


class Shiny(WMLResource):
    """Store and manage shiny assets."""

    ConfigurationMetaNames = ShinyMetaNames()
    """MetaNames for Shiny Assets creation."""

    def __init__(self, client: APIClient):
        WMLResource.__init__(self, __name__, client)

    def get_details(
        self,
        shiny_id: str | None = None,
        limit: int | None = None,
        get_all: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get shiny asset details. If no shiny_id is passed, details for all shiny assets are returned.

        :param shiny_id: unique ID of the shiny asset
        :type shiny_id: str, optional

        :param limit:  limit number of fetched records
        :type limit: int, optional

        :param get_all:  if True, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: metadata of the stored shiny asset
        :rtype:
          - **dict** - if shiny_id is not None
          - **{"resources": [dict]}** - if shiny_id is None

        **Example:**

        .. code-block:: python

            shiny_details = client.shiny.get_details(shiny_id)

        """
        shiny_id = _get_id_from_deprecated_uid(
            kwargs, shiny_id, "shiny", can_be_none=True
        )

        def get_required_elements(response: dict[str, Any]) -> dict[str, Any]:
            response = modify_details_for_script_and_shiny(response)
            final_response = {"metadata": response["metadata"], "entity": {}}

            return final_response

        return self._get_asset_based_resource(
            shiny_id, "shiny_asset", get_required_elements, limit=limit, get_all=get_all
        )

    def store(self, meta_props: dict, file_path: str) -> dict:
        """Create a shiny asset and upload content to it.

        :param meta_props: metadata of the shiny asset
        :type meta_props: dict

        :param file_path: path to the content file to be uploaded
        :type file_path: str

        :return: metadata of the stored shiny asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            meta_props = {
                client.shiny.ConfigurationMetaNames.NAME: "shiny app name"
            }

            shiny_details = client.shiny.store(meta_props, file_path="/path/to/file")
        """
        if self._client.project_type == "local_git_storage":
            raise ForbiddenActionForGitBasedProject(
                reason="Storing Shiny apps is not supported for git based project."
            )

        Shiny._validate_type(file_path, "file_path", str, True)

        shiny_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props, with_validation=True, client=self._client
        )

        response = self._create_asset(shiny_meta, file_path)

        final_response = {"metadata": response["metadata"], "entity": {}}

        return final_response

    def _create_asset(
        self, shiny_meta: dict[str, Any], file_path: str
    ) -> dict[str, Any]:

        # Step1: Create a shiny asset
        name = shiny_meta["metadata"]["name"]

        desc = ""
        if shiny_meta["metadata"].get("description"):
            desc = shiny_meta["metadata"]["description"]

        shiny_sw_spec_id = None
        if shiny_meta.get("software_spec_uid"):
            shiny_sw_spec_id = shiny_meta["software_spec_uid"]

        shiny_sw_specs = []
        deprecated_shiny_sw_specs = []

        if self._client.CPD_version >= 4.8:
            for sw_spec in self._client.software_specifications.get_details()[
                "resources"
            ]:
                if sw_spec.get("metadata", {}).get("life_cycle", {}):
                    if (
                        "shiny" in sw_spec["metadata"]["name"]
                        or "rstudio" in sw_spec["metadata"]["name"]
                    ):
                        if ("retired" or "deprecated" or "constricted") not in sw_spec[
                            "metadata"
                        ]["life_cycle"]:
                            shiny_sw_specs.append(sw_spec["metadata"]["name"])
                        elif "deprecated" in sw_spec["metadata"]["life_cycle"]:
                            deprecated_shiny_sw_specs.append(
                                sw_spec["metadata"]["name"]
                            )

        elif self._client.CPD_version >= 4.6:
            shiny_sw_specs = ["rstudio_r4.2"]
            deprecated_shiny_sw_specs = ["shiny-r3.6"]
        else:
            shiny_sw_specs = ["shiny-r3.6"]
            deprecated_shiny_sw_specs = []
        shiny_sw_spec_ids = [
            self._client.software_specifications.get_id_by_name(sw_name)
            for sw_name in shiny_sw_specs
        ]
        deprecated_shiny_sw_spec_ids = [
            self._client.software_specifications.get_id_by_name(sw_name)
            for sw_name in deprecated_shiny_sw_specs
        ]

        if (
            shiny_sw_spec_id
            and shiny_sw_spec_id not in shiny_sw_spec_ids + deprecated_shiny_sw_spec_ids
        ):
            raise WMLClientError(
                f"For R Shiny assets, only base software specs {','.join(shiny_sw_specs)} "
                "are supported. Specify "
                "the id you get via "
                "self._client.software_specifications.get_id_by_name(sw_name)"
            )
        elif shiny_sw_spec_id and shiny_sw_spec_id in deprecated_shiny_sw_spec_ids:
            software_spec_deprecated_warning = (
                "Provided software spec is deprecated for R Shiny assets. "
                f"Only base software specs {','.join(shiny_sw_specs)} "
                "are supported. Specify the id you get via "
                "self._client.software_specifications.get_id_by_name(sw_name)"
            )
            warn(software_spec_deprecated_warning, category=DeprecationWarning)

        asset_meta = {
            "metadata": {
                "name": name,
                "description": desc,
                "asset_type": "shiny_asset",
                "origin_country": "us",
                "asset_category": "USER",
            },
            "entity": {"shiny_asset": {"ml_version": "4.0.0"}},
        }
        if (self._client.CPD_version >= 4.6) and shiny_sw_spec_id:
            asset_meta["entity"]["shiny_asset"]["software_spec"] = {
                "base_id": shiny_sw_spec_id
            }

        # Step1: Create an asset
        print("Creating Shiny asset...")

        creation_response = requests.post(
            self._client.service_instance._href_definitions.get_data_assets_href(),
            headers=self._client._get_headers(),
            params=self._client._params(),
            json=asset_meta,
        )
        shiny_details = self._handle_response(
            201, "creating new asset", creation_response
        )

        # Step2: Create attachment
        if creation_response.status_code == 201:
            asset_id = shiny_details["metadata"]["asset_id"]
            attachment_meta = {
                "asset_type": "shiny_asset",
                "name": "attachment_" + asset_id,
            }

            attachment_response = requests.post(
                self._client.service_instance._href_definitions.get_attachments_href(
                    asset_id
                ),
                headers=self._client._get_headers(),
                params=self._client._params(),
                json=attachment_meta,
            )
            attachment_details = self._handle_response(
                201, "creating new attachment", attachment_response
            )
            if attachment_response.status_code == 201:
                attachment_id = attachment_details["attachment_id"]
                attachment_url = attachment_details["url1"]

                # Step3: Put content to attachment
                try:
                    with open(file_path, "rb") as f:
                        if not self._client.ICP_PLATFORM_SPACES:
                            put_response = requests.put(
                                attachment_url,
                                data=f.read(),
                            )
                        else:
                            put_response = requests.put(
                                self._credentials.url + attachment_url,
                                files={"file": (name, f, "application/octet-stream")},
                            )
                except Exception as e:
                    deletion_response = requests.delete(
                        self._client.service_instance._href_definitions.get_data_asset_href(
                            asset_id
                        ),
                        params=self._client._params(),
                        headers=self._client._get_headers(),
                    )
                    print(deletion_response.status_code)
                    raise WMLClientError("Failed while reading a file", e)

                if put_response.status_code == 201 or put_response.status_code == 200:
                    # Step4: Complete attachment
                    complete_response = requests.post(
                        self._client.service_instance._href_definitions.get_attachment_complete_href(
                            asset_id, attachment_id
                        ),
                        headers=self._client._get_headers(),
                        params=self._client._params(),
                    )
                    if complete_response.status_code == 200:
                        print("SUCCESS")
                        return self._get_required_element_from_response(shiny_details)
                    else:
                        try:
                            self.delete(asset_id)
                        except:
                            pass
                        raise WMLClientError(
                            "Failed while creating a shiny asset. Try again."
                        )
                else:
                    try:
                        self.delete(asset_id)
                    except:
                        pass
                    raise WMLClientError(
                        "Failed while creating a shiny asset. Try again."
                    )
            else:
                print("SUCCESS")
                return self._get_required_element_from_response(shiny_details)
        else:
            raise WMLClientError("Failed while creating a shiny asset. Try again.")

    def list(self, limit: int | None = None) -> DataFrame:
        """List stored shiny assets in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed shiny assets
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.shiny.list()

        """

        Shiny._validate_type(limit, "limit", int, False)
        href = self._client.service_instance._href_definitions.get_search_shiny_href()

        data: dict[str, Any] = {"query": "*:*"}
        if limit is not None:
            data.update({"limit": limit})

        response = requests.post(
            href,
            params=self._client._params(),
            headers=self._client._get_headers(),
            json=data,
        )
        self._handle_response(200, "list assets", response)
        shiny_details = self._handle_response(200, "list assets", response)["results"]
        space_values = [
            (
                m["metadata"]["name"],
                m["metadata"]["asset_type"],
                m["metadata"]["asset_id"],
            )
            for m in shiny_details
        ]

        table = self._list(space_values, ["NAME", "ASSET_TYPE", "ASSET_ID"], limit)
        return table

    def download(
        self,
        shiny_id: str | None = None,
        filename: str | None = None,
        rev_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Download the content of a shiny asset.

        :param shiny_id: unique ID of the shiny asset to be downloaded
        :type shiny_id: str

        :param filename: filename to be used for the downloaded file
        :type filename: str

        :param rev_id: ID of the revision
        :type rev_id: str, optional

        :return: path to the downloaded shiny asset content
        :rtype: str

        **Example:**

        .. code-block:: python

            client.shiny.download(shiny_id, "shiny_asset.zip")

        """
        if filename is None:
            raise TypeError(
                "download() missing 1 required positional argument: 'filename'"
            )

        shiny_id = _get_id_from_deprecated_uid(
            kwargs, shiny_id, "shiny", can_be_none=False
        )
        rev_id = _get_id_from_deprecated_uid(kwargs, rev_id, "rev", can_be_none=True)
        # Backward compatibility in past `rev_id` was an int.
        if isinstance(rev_id, int):
            rev_id_as_int_warning = "`rev_id` parameter type as int is deprecated, please convert to str instead"
            warn(rev_id_as_int_warning, category=DeprecationWarning)
            rev_id = str(rev_id)
        Shiny._validate_type(shiny_id, "shiny_id", str, True)
        Shiny._validate_type(rev_id, "rev_id", str, False)

        params = self._client._params()

        if rev_id is not None:
            params.update({"revision_id": rev_id})

        asset_response = requests.get(
            self._client.service_instance._href_definitions.get_data_asset_href(
                shiny_id
            ),
            params=params,
            headers=self._client._get_headers(),
        )
        shiny_details = self._handle_response(200, "get shiny assets", asset_response)

        attachment_id = shiny_details["attachments"][0]["id"]
        response = requests.get(
            self._client.service_instance._href_definitions.get_attachment_href(
                shiny_id, attachment_id
            ),
            params=params,
            headers=self._client._get_headers(),
        )
        if response.status_code == 200:
            attachment_signed_url = response.json()["url"]
            if "connection_id" in shiny_details["attachments"][0]:
                att_response = requests.get(attachment_signed_url)
            else:
                if not self._client.ICP_PLATFORM_SPACES:
                    att_response = requests.get(attachment_signed_url)
                else:
                    att_response = requests.get(
                        self._credentials.url + attachment_signed_url
                    )

            if att_response.status_code != 200:
                raise ApiRequestFailure(
                    "Failure during {}.".format("downloading asset"), att_response
                )

            downloaded_asset = att_response.content
            try:
                with open(filename, "wb") as f:
                    f.write(downloaded_asset)
                print(
                    "Successfully saved shiny asset content to file: '{}'".format(
                        filename
                    )
                )
                return os.getcwd() + "/" + filename
            except IOError as e:
                raise WMLClientError(
                    "Saving shiny asset with artifact_url to local file: '{}' failed.".format(
                        filename
                    ),
                    e,
                )
        else:
            raise WMLClientError("Failed while downloading the shiny asset " + shiny_id)  # type: ignore

    @staticmethod
    def get_id(shiny_details: dict) -> str:
        """Get the unique ID of a stored shiny asset.

        :param shiny_details: metadata of the stored shiny asset
        :type shiny_details: dict

        :return: unique ID of the stored shiny asset
        :rtype: str

        **Example:**

        .. code-block:: python

            shiny_id = client.shiny.get_id(shiny_details)

        """
        Shiny._validate_type(shiny_details, "shiny_details", object, True)
        Shiny._validate_type_of_details(shiny_details, DATA_ASSETS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(
            shiny_details, "data_assets_details", ["metadata", "guid"]
        )

    @staticmethod
    def get_uid(shiny_details: dict) -> str:
        """Get the Unique ID of a stored shiny asset.

        *Deprecated:* Use ``get_id(shiny_details)`` instead.

        :param shiny_details: metadata of the stored shiny asset
        :type shiny_details: dict

        :return: unique ID of the stored shiny asset
        :rtype: str

        **Example:**

        .. code-block:: python

            shiny_id = client.shiny.get_uid(shiny_details)

        """
        return Shiny.get_id(shiny_details)

    @staticmethod
    def get_href(shiny_details: dict) -> str:
        """Get the URL of a stored shiny asset.

        :param shiny_details: details of the stored shiny asset
        :type shiny_details: dict

        :return: href of the stored shiny asset
        :rtype: str

        **Example:**

        .. code-block:: python

            shiny_details = client.shiny.get_details(shiny_id)
            shiny_href = client.shiny.get_href(shiny_details)
        """
        Shiny._validate_type(shiny_details, "shiny_details", object, True)
        Shiny._validate_type_of_details(shiny_details, DATA_ASSETS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(
            shiny_details, "shiny_details", ["metadata", "href"]
        )

    def update(
        self,
        shiny_id: str | None = None,
        meta_props: dict | None = None,
        file_path: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update a shiny asset with metadata, attachment, or both.

        :param shiny_id: ID of the shiny asset
        :type shiny_id: str

        :param meta_props: changes to the metadata of the shiny asset
        :type meta_props: dict, optional

        :param file_path: file path to the new attachment
        :type file_path: str, optional

        :return: updated metadata of the shiny asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            script_details = client.script.update(shiny_id, meta, content_path)
        """
        shiny_id = _get_id_from_deprecated_uid(
            kwargs, shiny_id, "shiny", can_be_none=False
        )

        Shiny._validate_type(shiny_id, "shiny_id", str, True)

        if meta_props is None and file_path is None:
            raise WMLClientError(
                "At least either meta_props or file_path has to be provided"
            )

        updated_details = None

        url = self._client.service_instance._href_definitions.get_asset_href(shiny_id)

        # STEPS
        # STEP 1. Get existing metadata
        # STEP 2. If meta_props provided, we need to patch meta
        #   a. Construct meta patch string and call /v2/assets/<asset_id> to patch meta
        # STEP 3. If file_path provided, we need to patch the attachment
        #   a. If attachment already exists for the script, delete it
        #   b. POST call to get signed url for upload
        #   c. Upload to the signed url
        #   d. Mark upload complete
        # STEP 4. Get the updated script record and return

        # STEP 1
        response = requests.get(
            url, params=self._client._params(), headers=self._client._get_headers()
        )

        if response.status_code != 200:
            if response.status_code == 404:
                raise WMLClientError(
                    "Invalid input. Unable to get the details of shiny_id provided."
                )
            else:
                raise ApiRequestFailure(
                    "Failure during {}.".format("getting shiny asset to update"),
                    response,
                )

        details = self._handle_response(200, "Get shiny asset details", response)

        attachments_response = None

        # STEP 2a.
        # Patch meta if provided
        if meta_props is not None:
            self._validate_type(meta_props, "meta_props", dict, True)

            meta_patch_payload = []

            # Since we are dealing with direct asset apis, name and description is metadata patch
            if "name" in meta_props or "description" in meta_props:
                props_for_asset_meta_patch = {}

                for key in meta_props:
                    if key == "name" or key == "description":
                        props_for_asset_meta_patch.update({key: meta_props[key]})

                meta_patch_payload = (
                    self.ConfigurationMetaNames._generate_patch_payload(
                        details, props_for_asset_meta_patch, with_validation=True
                    )
                )
            if meta_patch_payload:
                meta_patch_url = (
                    self._client.service_instance._href_definitions.get_asset_href(
                        shiny_id
                    )
                )

                response_patch = requests.patch(
                    meta_patch_url,
                    json=meta_patch_payload,
                    params=self._client._params(),
                    headers=self._client._get_headers(),
                )

                updated_details = self._handle_response(
                    200, "shiny patch", response_patch
                )

        if file_path is not None:
            if "attachments" in details and details["attachments"]:
                current_attachment_id = details["attachments"][0]["id"]
            else:
                current_attachment_id = None

            # STEP 3
            attachments_response = self._update_attachment_for_assets(
                "shiny_asset", shiny_id, file_path, current_attachment_id
            )

        if attachments_response is not None and "success" not in attachments_response:
            self._update_msg(updated_details)

        # Have to fetch again to reflect updated asset and attachment ids
        url = self._client.service_instance._href_definitions.get_asset_href(shiny_id)

        response = requests.get(
            url, params=self._client._params(), headers=self._client._get_headers()
        )

        if response.status_code != 200:
            if response.status_code == 404:
                raise WMLClientError(
                    "Invalid input. Unable to get the details of shiny_id provided."
                )
            else:
                raise ApiRequestFailure(
                    "Failure during {}.".format("getting shiny to update"), response
                )

        response = self._get_required_element_from_response(
            self._handle_response(200, "Get shiny details", response)
        )

        final_response = {"metadata": response["metadata"], "entity": {}}

        return final_response

    def _update_msg(self, updated_details: Any) -> None:
        if updated_details is not None:
            print(
                "Could not update the attachment because of server error."
                " However metadata is updated. Try updating attachment again later"
            )
        else:
            raise WMLClientError(
                "Unable to update attachment because of server error. Try again later"
            )

    def delete(self, shiny_id: str | None = None, **kwargs: Any) -> str | dict:
        """Delete a stored shiny asset.

        :param shiny_id: unique ID of the shiny asset
        :type shiny_id: str

        :return: status ("SUCCESS" or "FAILED") if deleted synchronously or dictionary with response
        :rtype: str | dict

        **Example:**

        .. code-block:: python

            client.shiny.delete(shiny_id)

        """
        shiny_id = _get_id_from_deprecated_uid(
            kwargs, shiny_id, "shiny", can_be_none=False
        )
        Shiny._validate_type(shiny_id, "shiny_id", str, True)

        if self._if_deployment_exist_for_asset(shiny_id):
            raise WMLClientError(
                "Cannot delete shiny asset that has existing deployments. Please delete all associated deployments and try again"
            )

        response = requests.delete(
            self._client.service_instance._href_definitions.get_asset_href(shiny_id),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        if response.status_code == 200:
            return self._get_required_element_from_response(response.json())
        else:
            return self._handle_response(204, "delete assets", response)

    def create_revision(self, shiny_id: str | None = None, **kwargs: Any) -> dict:
        """Create a revision for the given shiny asset. Revisions are immutable once created.
        The metadata and attachment at `script_id` is taken and a revision is created out of it.

        :param shiny_id: ID of the shiny asset
        :type shiny_id: str

        :return: revised metadata of the stored shiny asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            shiny_revision = client.shiny.create_revision(shiny_id)
        """
        shiny_id = _get_id_from_deprecated_uid(
            kwargs, shiny_id, "shiny", can_be_none=False
        )

        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()
        Shiny._validate_type(shiny_id, "shiny_id", str, True)

        print("Creating shiny revision...")
        response = self._get_required_element_from_response(
            self._create_revision_artifact_for_assets(shiny_id, "Shiny")
        )

        final_response = {"metadata": response["metadata"], "entity": {}}

        return final_response

    def list_revisions(
        self, shiny_id: str | None = None, limit: int | None = None, **kwargs: Any
    ) -> DataFrame:
        """List all revisions for the given shiny asset ID in a table format.

        :param shiny_id: ID of the stored shiny asset
        :type shiny_id: str

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed shiny revisions
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.shiny.list_revisions(shiny_id)
        """
        shiny_id = _get_id_from_deprecated_uid(
            kwargs, shiny_id, "shiny", can_be_none=False
        )
        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        Shiny._validate_type(shiny_id, "shiny_id", str, True)

        url = (
            self._client.service_instance._href_definitions.get_asset_href(shiny_id)
            + "/revisions"
        )

        # /v2/assets/{asset_id}/revisions returns 'results' object
        shiny_resources = self._get_with_or_without_limit(
            url,
            None,
            "List Shiny revisions",
            summary=None,
            pre_defined=None,
            _all=self._should_get_all_values(limit),
        )["resources"]

        shiny_values = [
            (
                m["metadata"]["asset_id"],
                m["metadata"]["revision_id"],
                m["metadata"]["name"],
                m["metadata"]["commit_info"]["committed_at"],
            )
            for m in shiny_resources
        ]

        table = self._list(
            shiny_values,
            ["ID", "REVISION_ID", "NAME", "REVISION_COMMIT"],
            limit,
        )
        return table

    def get_revision_details(
        self, shiny_id: str | None = None, rev_id: str | None = None, **kwargs: Any
    ) -> ListType:
        """Get metadata of the `shiny_id` revision.

        :param shiny_id: ID of the shiny asset
        :type shiny_id: str

        :param rev_id: ID of the revision. If this parameter is not provided, it returns the latest revision. If there is no latest revision, it returns an error.
        :type rev_id: str, optional

        :return: stored shiny(s) metadata
        :rtype: list

        **Example:**

        .. code-block:: python

            shiny_details = client.shiny.get_revision_details(shiny_id, rev_id)
        """
        shiny_id = _get_id_from_deprecated_uid(
            kwargs, shiny_id, "shiny", can_be_none=False
        )
        rev_id = _get_id_from_deprecated_uid(kwargs, rev_id, "rev", can_be_none=True)

        # Backward compatibility in past `rev_id` was an int.
        if isinstance(rev_id, int):
            rev_id_as_int_warning = "`rev_id` parameter type as int is deprecated, please convert to str instead"
            warn(rev_id_as_int_warning, category=DeprecationWarning)
            rev_id = str(rev_id)

        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        Shiny._validate_type(shiny_id, "shiny_id", str, True)
        Shiny._validate_type(rev_id, "rev_id", str, False)

        if rev_id is None:
            rev_id = "latest"

        url = self._client.service_instance._href_definitions.get_asset_href(shiny_id)
        resources = self._get_with_or_without_limit(
            url,
            limit=None,
            op_name="asset_revision",
            summary=None,
            pre_defined=None,
            revision=rev_id,
        )["resources"]
        responses = [
            self._get_required_element_from_response(resource) for resource in resources
        ]

        final_responses = [
            {"metadata": response["metadata"], "entity": {}} for response in responses
        ]

        return final_responses

    def _get_required_element_from_response(self, response_data: dict) -> dict:

        WMLResource._validate_type(response_data, "shiny", dict)

        revision_id = None
        metadata = {
            "guid": response_data["metadata"]["asset_id"],
            "href": response_data["href"],
            "name": response_data["metadata"]["name"],
            "asset_type": response_data["metadata"]["asset_type"],
            "created_at": response_data["metadata"]["created_at"],
            "last_updated_at": response_data["metadata"]["usage"]["last_updated_at"],
        }
        try:
            if self._client.default_space_id is not None:
                metadata["space_id"] = response_data["metadata"]["space_id"]

            elif self._client.default_project_id is not None:
                metadata["project_id"] = response_data["metadata"]["project_id"]

            if "revision_id" in response_data["metadata"]:
                revision_id = response_data["metadata"]["revision_id"]
                metadata.update(
                    {"revision_id": response_data["metadata"]["revision_id"]}
                )

            if "attachments" in response_data and response_data["attachments"]:
                metadata.update(
                    {"attachment_id": response_data["attachments"][0]["id"]}
                )

            if "commit_info" in response_data["metadata"] and revision_id is not None:
                metadata.update(
                    {
                        "revision_commit_date": response_data["metadata"][
                            "commit_info"
                        ]["committed_at"]
                    }
                )

            new_el = {"metadata": metadata, "entity": response_data["entity"]}

            if "description" in response_data["metadata"]:
                new_el["metadata"].update(
                    {"description": response_data["metadata"]["description"]}
                )

            if "href" in response_data["metadata"]:
                href_without_host = response_data["metadata"]["href"].split(".com")[-1]
                new_el["metadata"].update({"href": href_without_host})
            return new_el
        except Exception:
            raise WMLClientError(
                f"Failed to read Response from down-stream service: {response_data}"
            )
